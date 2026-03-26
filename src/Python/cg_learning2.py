import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
import subprocess
import cupy as cp
import torch
import torch.nn as nn
import random
import math
import scipy
from enum import Enum
import statistics
from tqdm import tqdm
import random
import os
import time
from functools import wraps

class ArrayOrder(Enum):
    ROW_MAJOR = 1
    COL_MAJOR = 2
    UNDETERMINED = 3

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0 
        self.M2 = 0.0  # This will track the sum of squares of differences from the mean

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def finalize(self):
        if self.n < 2:
            return float('nan')  # Not enough data points for a valid result
        variance = self.M2 / (self.n - 1)
        return self.mean, math.sqrt(variance)

class TimerWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        # If the first argument is an instance of a class, it's likely 'self'
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {self.func.__name__} took {end_time - start_time:.4f} seconds to complete.")
        return result

def timerable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__class__'):
            return TimerWrapper(func)(*args, **kwargs)
        else:
            return TimerWrapper(func)(*args, **kwargs)
    return wrapper


def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfilt(sos, data)
    return filtered_data

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfilt(sos, data)
    return filtered_data

def support(template):
    return np.where(np.any(template != 0, axis=0))[0]

def compute_mean_std(arr, start, end, inc):
    sublist = np.asarray(arr[start:end:inc])
    
    stats = RunningStats()
    for x in sublist:
        stats.update(x)
    mean, std = stats.finalize()
    return mean, std

def covariance_distance_from_identity(matrix):
    # Compute the covariance matrix
    cov_matrix = np.cov(matrix, rowvar=False)

    # Create the identity matrix of the same size
    identity_matrix = np.eye(cov_matrix.shape[0])

    # Compute the Frobenius norm of the difference between the covariance matrix and the identity matrix
    distance = np.linalg.norm(cov_matrix - identity_matrix, ord='fro')

    # Print the distance
    print(f"Distance from the identity matrix (Frobenius norm): {distance}")

def whiten(arr, mat, start, end, inc):
    sublist = arr[start:end:inc]
    whitened = np.dot(mat, sublist)
    arr[start:end:inc] = whitened

def mean_subtract(arr, start, end, inc):
    sublist = arr[start:end:inc]
    mean_value = sum(sublist) / len(sublist)
    
    for i in range(start, len(arr), inc):
        arr[i] -= mean_value
    
    return arr

def median_subtract(arr, start, end, inc):
    sublist = arr[start:end:inc]
    median_value = np.median(sublist)
    for i in range(start, len(arr), inc):
        arr[i] -= median_value

    return arr

def transpose_flattened_array(flattened_array, rows, cols):
    # Convert the flattened array to a 2D array
    original_array = np.array(flattened_array).reshape(rows, cols)
    
    # Transpose the 2D array
    transposed_array = original_array.T
    
    # Flatten the transposed array back to 1D
    transposed_flattened_array = transposed_array.flatten()
    
    return transposed_flattened_array

def detect_memory_order(flattened_array, rows, cols):
    A_row_major = flattened_array.reshape((rows, cols))
    A_col_major = transpose_flattened_array(flattened_array, rows, cols).reshape((rows, cols))
    
    row_major_diff_avg = np.sum(np.abs(A_row_major[:, :-1] - A_row_major[:, 1:])) / rows
    col_major_diff_avg = np.sum(np.abs(A_col_major[:, :-1] - A_col_major[:, 1:])) / cols
    
    print(f"Row-major difference avg: {row_major_diff_avg}")
    print(f"Column-major difference avg: {col_major_diff_avg}")
    
    if row_major_diff_avg < col_major_diff_avg:
        print("The array is most likely row-major.")
        return ArrayOrder.ROW_MAJOR
    else:
        print("The array is most likely column-major.")
        return ArrayOrder.COL_MAJOR

def heatmap(matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')  # 'viridis' is a perceptually-uniform colormap
    plt.colorbar()  # Show color scale
    plt.title(f'Heatmap {matrix.shape}')
    plt.show()

def plot_recording(data):
    samples, channels = data.shape
    plt.figure(figsize=(15, 10))  # Set the figure size
    plt.grid(visible=None)
    plt.axis('off')
    
    # Create a time axis for the data
    time = np.arange(samples)
    # Plot each channel
    for i in range(channels):
        offset = 2 * i
        plt.plot(time, (data[:, i] + offset) * 50)

    plt.title('Recording with Multiple Channels')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (arbitrary units)')
    plt.legend(loc='upper right')
    plt.grid(True)  # Add a grid for better readability
    

def convolve(batch, template):
    # Ensure inputs are CuPy arrays for GPU acceleration
    g_batch = cp.asarray(batch)
    g_template = cp.asarray(template)

    # Compute the convolution for each channel and sum across channels
    # We assume 'batch' and 'template' are both formatted correctly (W x C and M x C respectively)
    result = cp.zeros(g_batch.shape[0])

    for channel in range(g_batch.shape[1]):
        # Convolve each channel of the batch with the corresponding channel of the template
        # 'full' mode is used initially to get all possible overlaps
        channel_result = cp.convolve(g_batch[:, channel], g_template[::-1, channel], mode='full')

        # We need to trim the result to match the input batch size
        # Start index for slicing the convolution result to align with the 'same' mode concept
        start_idx = g_template.shape[0] - 1
        result += channel_result[start_idx:start_idx + g_batch.shape[0]]
        
    result /= cp.linalg.norm(result)
    return result.get()

def find_max_projection(projections):
    # Initialize variables to keep track of the maximum value and its indices
    max_value = -np.inf
    max_template_index = -1
    max_sample_index = -1
    
    # Iterate over each projection to find the maximum value and its indices
    for i, projection in enumerate(projections):
        # Find the index of the maximum value in the current projection
        current_max_value = np.abs(np.max(projection))
        if current_max_value > max_value:
            max_value = current_max_value
            max_template_index = i
            max_sample_index = np.argmax(projection)
    
    return max_template_index, max_sample_index

def conjugate_gradients(A, y):
    eps = 0.01
    n = A.shape[1]
    x = np.zeros(n)
    r = np.dot(A.T, y)
    d = np.copy(r)
    delta_new = np.dot(r, r)
    print(f"Condition number = {LA.cond(A)}")
    print(f"Initial l2r squared = {delta_new}")
    delta_0 = delta_new
    
    i = 0
    while i < 100 and delta_new > eps * eps * delta_0:
        q = np.dot(A.T, np.dot(A, d))
        alpha = delta_new / np.dot(d, q)
        x = x + alpha * d
        r = r - alpha * q
        delta_old = delta_new
        delta_new = np.dot(r, r)
        beta = delta_new / delta_old
        d = r + beta * d
        print(f"Iteration {i}: alpha = {alpha}, beta = {beta}, residual = {pow(delta_new, 0.5)}")
        i += 1

    row_space_residual = y - np.dot(A, x)
    residual_norm = pow(np.dot(row_space_residual, row_space_residual), 0.5)
    print(f"Row space residual norm = {residual_norm}")
    return x

def conjugate_gradients_gpu(A, y):
    #print("================== CONJUGATE GRADIENTS ==================")
    #print(f"Condition number = {LA.cond(A)}")
    #print(f"Is A nonzero? {np.any(A != 0)}")
    #print(f"Norm of y: {np.linalg.norm(y)}")
    y_support = np.where(y != 0, True, False)
    support_intersect_counts = []
    for col in range(A.shape[1]):
        support = np.where(A[:, col] != 0, True, False)
        support_intersect_count = 0
        for i in range(y.shape[0]):
            if support[i] and y_support[i]:
                support_intersect_count += 1
        support_intersect_counts.append(support_intersect_count)
            
    #print(f"Size of support overlap between y and each column of A: {support_intersect_counts}")
    A = cp.asarray(A)
    y = cp.asarray(y)
    n = A.shape[1]
    x = cp.zeros(n)
    r = cp.dot(A.T, y)
    d = r.copy()
    delta_new = cp.dot(r, r)
    delta_0 = delta_new.copy()
    #print(f"Initial residual L2 squared = {delta_new}")
    i = 0
    while i < 100 and delta_new > 0.0001 * delta_0:
        q = cp.dot(A.T, cp.dot(A, d))
        alpha = delta_new / cp.dot(d, q)
        x += alpha *d
        r -= alpha * q
        delta_old = delta_new.copy()
        delta_new = cp.dot(r, r)
        beta = delta_new / delta_old
        d = r + beta * d
       # print(f"Iteration {i}: alpha = {alpha}, beta = {beta}, residual = {delta_new}")
        i += 1

    row_space_residual = y - cp.dot(A, x)
    residual_norm = pow(cp.dot(row_space_residual, row_space_residual), 0.5)
    #print(f"Final residual norm = {residual_norm}")
    #print(f"Conjugate gradients final x = {x}")
    #print("================== CONJUGATE GRADIENTS END ==================")
    return x.get()  # Convert back to NumPy array if needed

def parse_bin_meta_file(filename):
    metadata = { }
    with open(filename, 'r') as bin_meta_input:
        n_channels = -1
        for line in bin_meta_input:
            delimited = line.split('=')

            if len(delimited) != 2:
                continue

            key = delimited[0]
            value = delimited[1]
            metadata[key] = value

    return metadata

def clear_line():
    # Move the cursor to the beginning of the line and clear it
    print('\r' + ' ' * 80, end='\r')

def track_progress(message):
    # Print the progress message in the same spot
    clear_line()
    print(message, end='', flush=True)

def normalize_template(template):
    channel_norms = np.linalg.norm(template, axis=0)
    template = np.where(channel_norms != 0.0, template / channel_norms, template)

    
class OSS:
    DRIVE_PATH = Path("C:/")
    BASE_PATH = Path(DRIVE_PATH, "SGL_DATA", "05_31")
    BIN_DIR = Path(BASE_PATH, "imec_raw")
    KS_OUTPUT_DIR = Path(BIN_DIR, "kilosort4")
    DECODER_INPUT_DIR = Path(BASE_PATH, "decoder_input")
    CROPPED_OUTPUT_DIR = Path(BASE_PATH, "oss_training")
    CROPPED_OUTPUT_DIR_1 = Path(BASE_PATH, "oss_training_1")
    CROPPED_OUTPUT_DIR_2 = Path(BASE_PATH, "oss_training_2")

    BIN_FILE = "240531_g0_t0.imec0.ap.bin"
    BIN_META_FILE = "240531_g0_t0.imec0.ap.meta"
    CHANNEL_MAP_FILE = "neuropixels_NHP_channel_map_dev_staggered_v1.mat"
    CPP_MAIN_FILE = Path(r"C:\Users\Spike Sorter\source\repos\OnlineSpikes_v2\x64\RELEASE", "OnlineSpikes.exe")
    SPIKES_OUTPUT_FILE = Path(BASE_PATH, "sorter_output", "spikeOutput.txt")
    OSS_TRAINING_PATH = Path("C:/SGL_DATA/05_31/oss_training_std/")
    
    WINDOW_SIZE = 10 # multiple of # samples in template
    SAMPLING_RATE_MS = 30
    SIMULATED_NUM_SPIKES = 20
    SIMULATED_AMPLITUDE = 1
    IS_SIMULATED_NOISY = True
    IS_SIMULATED = False
    
    def __init__(self):
        config = {
            "tau": 0.2,
            "thresh": 0.99,
            "start_sample": 0,
            "W": 1500,
            "event_time_sample": 0 
        }

        self.config(config)
        
        # Seed with current time
        np.random.seed()
        random.seed()

        # Load kilosort output and cropped output
        self.templates = np.load(self.OSS_TRAINING_PATH / "templates.npy")
        self.whitening = np.load(self.OSS_TRAINING_PATH / "whiteningMat.npy")
        self.channel_mask = np.load(self.OSS_TRAINING_PATH / "channelMask.npy")
        self.template_map = np.load(self.OSS_TRAINING_PATH / "templateMap.npy")
        self.inverse_template_map = { self.template_map[i] : i for i in range(len(self.template_map)) }
        self.spike_times = np.load(self.KS_OUTPUT_DIR / "spike_times.npy")
        self.spike_templates = np.load(self.KS_OUTPUT_DIR / "spike_templates.npy")
        self.amplitudes = np.load(self.KS_OUTPUT_DIR / "amplitudes.npy")
        self.T = self.templates.shape[0]
        self.M = self.templates.shape[1]
        self.C = self.templates.shape[2]

        #for template in self.templates:
        #    plt.plot(template)
        #    plt.show()
        # Compute average amplitudes per template
        average_computers = [RunningStats() for _ in range(self.T)]
        for index, template in enumerate(self.spike_templates):
            if template not in self.inverse_template_map:
                continue
            template = self.inverse_template_map[template]
            average_computers[template].update(self.amplitudes[index])
        self.avg_amplitudes = [average_computer.finalize()[0] for average_computer in average_computers]

        # Parse metadata file for pre-cropped parameters
        metadata = parse_bin_meta_file(self.BIN_DIR / self.BIN_META_FILE)
        self.raw_C = int(metadata["nSavedChans"])   
        print(self.raw_C)
        self.raw_T = self.spike_templates.shape[0]

        # Use the Kilosort output to form the matrix A
        self.construct_A()
            
        # Learn the means and stds
        self.learn_means_stds_noise()

        # Gather the ground truth
        self.plot_data = []
        self.gather_ground_truth(self.start_sample)

        print(f"CuPy is using device: {cp.cuda.Device()}")

    def config(self, config):
        self.TAU = config["tau"]
        self.THRESH = config["thresh"]
        self.start_sample = config["start_sample"]
        self.W = config["W"]
        self.event_time_sample = config["event_time_sample"]
    
    def run(self):        
        start_offset = -300 * 30
        samples_to_read = 600 * 30
        window_size = 50 * 30
        output_dir = Path(self.BASE_PATH, f"sorter_output_half_of_events")
       # self.sort_and_psth(start_offset, samples_to_read, window_size, output_dir, self.DECODER_INPUT_DIR / "eventfile.txt")
        #self.avg_psth(output_dir, start_offset, samples_to_read)

        if self.IS_SIMULATED:
            self.simulate_batch()
        else:
            self.fetch_batch_from_file(Path("C:\\SGL_DATA\\05_31\\imec_raw") / '240531_g0_t0.imec0.ap.bin', start_sample=self.start_sample)
            self.process_signal(self.y)

            for chan_index in range(min(10, self.C)):
                mean, std = compute_mean_std(self.y, chan_index, len(self.y), self.C)
                print(f"Channel {chan_index} mean = {mean} std = {std} projected_stds = {self.stds[chan_index]}")

            self.plot_y_flat()

        x_omp, indices = self.orthogonal_matching_pursuit()
        A_indices = self.extract_submatrix(indices)
        y_omp = np.dot(A_indices, x_omp[indices])
        y_omp_mem_order = detect_memory_order(y_omp, self.W, self.C)

        plot_recording(np.reshape(y_omp, (self.W, self.C)))
        
        plt.figure(figsize=(12, 8))
        
        if self.IS_SIMULATED:
            plt.subplot(321)
            plt.stem(self.x_true, linefmt='b-', markerfmt='bo', basefmt='r-')
            plt.title('True Sparse x')
        else:
            plt.subplot(321)
            plt.stem(self.x_truth, linefmt='b-', markerfmt='bo', basefmt='r-')
            plt.title('Kilosort True x')

        plt.subplot(322)
        plt.stem(x_omp, linefmt='g-', markerfmt='go', basefmt='r-')
        plt.title('Recovered Sparse x')

        plt.subplot(323)

        plt.plot(self.y, 'r-')
        plt.title('Vector y')

        plt.subplot(324)
        plt.plot(y_omp)
        plt.title('Reconstructed y')

        if not self.IS_SIMULATED:
            plt.subplot(325)
            plt.plot(self.y_reconstructed_truth)
            plt.title('Reconstructed y from Kilosort')

        plt.tight_layout()
        plt.show()

        if not self.IS_SIMULATED:
            print(self.true_spikes)
            print(self.parse_spikes(indices))

    def fft(self, signal, channel, num_channels, title):
        fft_vals = np.fft.fft(signal[channel : len(signal) : num_channels])
        power = np.abs(fft_vals)
        frequencies = np.fft.fftfreq(len(power), 1/(self.SAMPLING_RATE_MS * 1000))
        plt.figure(figsize=(12, 6))
        plt.plot(frequencies[:len(frequencies)//2], power[:len(frequencies)//2])  # Plot only the positive frequencies
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()
        
    def psth(self, spikes, event_time, start_samp_offset=-600 * 30, end_samp_offset=500 * 30, inc=10 * 30, window_size=50 * 30):
        # The list of counts to plot for PSTH
        psth_x = []
        psth_y = []

        start = start_samp_offset
        end = start_samp_offset + window_size

        post_count = 0
        pre_count = 0
        
        while end <= end_samp_offset:
            window_spike_count = 0
            
            for (sample, template) in spikes:
                if sample >= start + event_time and sample <= end + event_time:
                    window_spike_count += 1

            if -200 < end / 30 <= -100:
                pre_count += window_spike_count

            if 100 < end / 30 <= 200:
                post_count += window_spike_count
                
            psth_x.append(end / 30)
            psth_y.append(window_spike_count)
            start += inc
            end += inc

        return psth_x, psth_y, post_count / pre_count

    def sort_and_psth(self, start_offset, psth_width, window_size, output_dir, eventfile, tau=0.20, thresh=0.11, num_events=400):
        with open(eventfile, 'r') as file:
            event_times = [int(line) for line in file]

        ### Loop through event times and perform a PSTH for each one
        self.TAU = tau
        self.THRESH = thresh
        
        for event_index, event_time in enumerate(event_times):
            self.event_time_sample = event_time
            samples_read = 0
            sorted_spikes_filename = f"spikeIndices{self.event_time_sample}tau{str(self.TAU)[2:]}.npy"
            sorted_spikes = []

            track_progress(f"Processing event at sample {event_time}\n")
            
            # Kilosort output for PSTH
            kilo_spikes = []
            for i, spike_time in enumerate(self.spike_times):
                spike_template = self.spike_templates[i]

                if spike_time >= start_offset + self.event_time_sample - window_size and spike_time <= self.event_time_sample + start_offset + psth_width and spike_template in self.template_map:
                    kilo_spikes.append((spike_time, spike_template))

            relevant_spikes = [tup for tup in kilo_spikes if start_offset <= tup[0] - self.event_time_sample <= start_offset + psth_width]
            psth_x, psth_y, modulation = self.psth(relevant_spikes, self.event_time_sample, start_samp_offset=start_offset, end_samp_offset=start_offset + psth_width)
            
            plt.title(f"KS PSTH event {self.event_time_sample} tau = {self.TAU} modulation = {modulation}")
            plt.plot(psth_x, psth_y)
            plt.savefig(output_dir / f"{self.event_time_sample}tau0{str(self.TAU)[2:]}kspsth.pdf")
            plt.clf()
            
            # Read in data for verify OSS with PSTH
            while samples_read <= psth_width:
                samples_read += self.fetch_batch_from_file(Path("C:\\SGL_DATA\\05_31\\imec_raw") / '240531_g0_t0.imec0.ap.bin', start_sample=self.event_time_sample + start_offset + samples_read)
                print(f"Read {samples_read} samples")
                self.whiten_signal(self.y)
                self.mean_subtract_signal(self.y)
                self.filter_signal(self.y, self.C)
                x_omp, support = self.orthogonal_matching_pursuit()
                sorted_spikes += self.parse_spikes(support, offset=self.event_time_sample + start_offset + samples_read - self.W)

            np.save(output_dir / sorted_spikes_filename, np.array(sorted_spikes))
            
            # Sort for PSTH
            sorted_spikes = np.load(output_dir / sorted_spikes_filename)
            psth_x, psth_y , modulation = self.psth(sorted_spikes, self.event_time_sample, start_samp_offset=start_offset, end_samp_offset=start_offset + psth_width)
            plt.title(f"LSS PSTH event {self.event_time_sample} tau = {self.TAU} modulation = {modulation}")
            plt.plot(psth_x, psth_y)
            plt.savefig(output_dir / f"{self.event_time_sample}tau0{str(self.TAU)[2:]}osspsth.pdf")
            plt.clf()

    def avg_psth(self, output_dir, start_offset, psth_width):
        # Loop through all files in the directory
        avg_x = None
        avg_y = None
        kilo_x = None
        kilo_y = None

        print(self.template_map.shape)
        
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)

            if filename[:len("spikeIndices")] != "spikeIndices":
                continue

            sorted_spikes = np.load(file_path)
            event_sample_time = int(filename.split("tau")[0][len("spikeIndices"):])
            tau_str = filename.split("tau")[1][:-len(".npy")]
            print(tau_str)



            # Kilosort
            kilo_spikes = []
            for i, spike_time in enumerate(self.spike_times):
                spike_template = self.spike_templates[i]

                if start_offset <= spike_time - event_sample_time <= start_offset + psth_width and spike_template in self.template_map:
                    kilo_spikes.append((spike_time, spike_template))
                        
            psth_x, psth_y, modulation = self.psth(kilo_spikes, event_sample_time, start_samp_offset=start_offset, end_samp_offset=start_offset + psth_width)
            
       #     if modulation < 1.2:
       #         print("Modulation too low, skipping")
       #         continue

            print(f"Performing PSTH for {event_sample_time}")
            if kilo_x == None:
                kilo_x = psth_x
                kilo_y = psth_y
            else:
                kilo_y = [kilo_y[i] + psth_y[i] for i in range(len(psth_y))]

            # OSS
            psth_x, psth_y, _ = self.psth(sorted_spikes, event_sample_time, start_samp_offset=start_offset, end_samp_offset=start_offset + psth_width)

            if avg_x == None:
                avg_x = psth_x
                avg_y = psth_y
            else:
                avg_y = [avg_y[i] + psth_y[i] for i in range(len(psth_y))]

        avg_y = [i / max(avg_y) for i in avg_y]
        kilo_y = [i / max(kilo_y) for i in kilo_y]
        plt.plot(avg_x, avg_y, label='oss')
        plt.plot(kilo_x, kilo_y, label='ks')
        plt.legend()
        plt.show()
        
    def gather_ground_truth(self, start_sample):
        print(f"Gathering ground truth from {str(self.KS_OUTPUT_DIR)} sample {start_sample} to sample {start_sample + self.W}")
        self.x_truth = np.zeros(self.T * self.W)
        self.y_reconstructed_truth = []
        self.true_spikes = []
        indices = []
        x_truth_induced = []
        for i, spike_time in enumerate(self.spike_times):
            if self.spike_templates[i] not in self.inverse_template_map:
                continue
            spike_template = self.inverse_template_map[self.spike_templates[i]]
            if spike_time >= start_sample and spike_time <= start_sample + self.W:
                spike_time -= start_sample
                indices.append(spike_template + spike_time * self.T)
                self.x_truth[spike_template + spike_time * self.T] = self.amplitudes[i]
                self.true_spikes.append((spike_time, self.spike_templates[i]))

               
        x_truth_induced = self.x_truth[indices]
        A_indices = self.extract_submatrix(indices)
        self.y_reconstructed_truth = np.dot(A_indices, x_truth_induced)
                
    def construct_A(self):
        templates_buff = self.templates.flatten()
        self.D  = np.zeros(self.C * self.M * self.T, dtype=np.float32)
        self.D2 = np.zeros(self.C * self.M * self.T, dtype=np.float32)
        self.D3 = np.zeros(self.C * self.M * self.T, dtype=np.float32)

        for sampleInd in range(self.M):
            for templateInd in range(self.T):
                for chanInd in range(self.C):
                    ind = chanInd + sampleInd * self.C + templateInd * self.C * self.M
                    temp = templates_buff[ind]
                    self.D[sampleInd + templateInd * self.M + chanInd * self.T * self.M] = temp
                    self.D2[chanInd + sampleInd * self.C + templateInd * self.C * self.M] = temp
                    self.D3[chanInd + templateInd * self.C + sampleInd * self.C * self.T] = temp
        

    def fetch_batch_from_file(self, filename, start_sample=0):
        print(f"Fetching data from {filename} from sample {start_sample} to {start_sample + self.W} on {self.C} channels")
        with open(filename, 'rb') as fidInput:
            self.y = np.fromfile(fidInput, dtype=np.int16, offset=start_sample * self.raw_C * 2, count=self.raw_C * self.W)

            if self.y.size == 0 or self.y.size != self.raw_C * self.W:
                print(f"Error: read {self.y.size} entries from {filename}, expected {self.C * self.W}")

            self.y = np.reshape(self.y, (self.raw_C, self.W), order='F')[self.channel_mask, :].flatten(order='F')
            self.y = self.y.astype(np.float32)
            return self.W       
                
    def add_plot(self, data, title):
        self.plot_data.append((data, title))
        
    def show_all_plots(self):
        if len(self.plot_data) == 0:
            return
        
        n = len(self.plot_data)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))  # Create a 1xN grid of subplots

        if n == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one plot

        for i, (data, title) in enumerate(self.plot_data):
            axes[i].plot(data)
            axes[i].set_title(title)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

        self.plot_data = []
       
    def plot_y_flat(self):
        plt.plot(self.y, 'r-')
        plt.title(f"y, shape={self.y.shape}")
        plt.show()

    def plot_y_stacked(self):
        plot_recording(np.reshape(self.y, (self.W, self.C), order='F'))
        
    def simulate_batch(self):
        nonzero_indices = np.random.choice(self.T * self.W, self.SIMULATED_NUM_SPIKES, replace=False)
        self.x_true = np.zeros(self.T * self.W)
        x_true_induced = [random.randint(1, self.SIMULATED_AMPLITUDE) for _ in range(self.SIMULATED_NUM_SPIKES)]
        self.x_true[nonzero_indices] = x_true_induced
        A_induced = self.extract_submatrix(nonzero_indices)
        self.y = np.dot(A_induced, x_true_induced)
        if self.IS_SIMULATED_NOISY:
            self.add_gaussian_iid_noise(0, 0.2)

    def add_gaussian_iid_noise(self, mean, std):
        self.y += np.random.normal(mean, std, self.W * self.C)

    def add_frequency_noise(self, noise_freq):
        """
        Generates a length W array of discrete noise at a specific frequency.

        Parameters:
        noise_freq (float): Frequency of the noise to generate in Hz.

        Returns:
        np.ndarray: Array of length W containing the generated noise.
        """
        # Generate the time vector for W samples
        t = np.arange(self.W) / self.SAMPLING_RATE_MS
        
        # Generate the noise as a sine wave at the specified frequency
        noise_signal = np.sin(2 * np.pi * noise_freq * t)

        # Add the noise to each channel
        for chan_index in range(self.C):
            self.y[chan_index : len(self.y) : self.C] += noise_signal
    
    def extract_submatrix(self, indices):
        num_cols = len(indices)
        num_rows = self.W * self.C

        g_A = np.zeros(num_rows * num_cols, dtype=np.float32)

        count = 0
        for i, index in enumerate(indices):
            sample_index = index // self.T
            template_index = index % self.T

            for chansamp_index in range(min(self.M, self.W - sample_index) * self.C):
                g_A[i * self.W * self.C + sample_index * self.C + chansamp_index] = self.D2[template_index * self.M * self.C + chansamp_index]

        return np.reshape(g_A, (num_rows, num_cols), order='F')

    @timerable
    def orthogonal_matching_pursuit(self):
        MAX_ITER = 1000

        if self.IS_SIMULATED:
            residual = cp.array(self.y.copy())
        else:
            residual = cp.array(self.y.astype(np.float32).copy())
        
        indices = []
        x_approx = cp.zeros(self.T * self.W)
        rdot_init = cp.dot(residual, residual)
        rdot_old = rdot_init
        rdot_new = rdot_init
        max_relative_improvements = []
        
     #   print(f"Initial residual norm squared: {rdot_init}"        
        for _ in range(MAX_ITER):
            # Project the residual on the columns of A and find the best match
            assert self.templates[0].shape == (self.M, self.C)
            residual_reshaped = cp.reshape(residual, (self.W, self.C))
            projections = [convolve(residual_reshaped, template) for template in self.templates]
            #projections = [2 * self.avg_amplitudes[i] * convolve(residual_reshaped, template) - self.avg_amplitudes[i] ** 2 for i, template in enumerate(self.templates)]
            #heatmap(np.reshape(projections, (self.T, self.W)))
            best_template, best_sample = find_max_projection(projections)
            indices.append(best_sample * self.T + best_template)
            print(f"Max = {projections[best_template][best_sample]}")
            
            if _ == 0:
                max_old = projections[best_template][best_sample]
                max_new = projections[best_template][best_sample]

            max_old = max_new
            max_new = projections[best_template][best_sample]

            max_relative_improvements.append(max_new / max_old)

            if _ >= 5 and sum(max_relative_improvements[-5:]) / 5 >= self.THRESH:
                break

            max_std = 0
            avg_std = 0
            for channel in range(self.C):
                std = self.max_std(residual, channel)
                avg_std = max(avg_std, self.avg_std(residual, channel))
                max_std = max(std, max_std)

            if _ > 0:
                plt.title(f"Residual iteration {_}")
                plt.plot(residual)
                plt.show()
            print(f"max_std = {max_std}, avg_std = {avg_std}")
           # print(f"The largest uncertainty of {projections[best_template][best_sample]} was found at template {best_template} and sample {best_sample}, corresponding to index {best_template + best_sample * self.T}.")

            # Update the approximation and residual
            A_selected = self.extract_submatrix(indices)
            x_sparse = conjugate_gradients_gpu(A_selected, self.y)
            print(x_sparse)
            x_approx[indices] = x_sparse
            residual = self.y - cp.dot(A_selected, x_sparse)
            rdot_old = rdot_new
            rdot_new = cp.dot(residual, residual)

            # To account for the residual jumping extremely high during the first iteration
            if _ == 0:
                rdot_init = rdot_new
                
            track_progress(f"OMP Iteration {_}, sample {self.event_time_sample}, past 5 max ratio avg = {sum(max_relative_improvements[-5:]) / 5}, max change ratio = {max_new / max_old}, initial residual {rdot_init}: {(rdot_init - rdot_old) / (len(indices) - 1) * self.TAU} > {rdot_old - rdot_new}\n")
                
            if len(indices) > 1 and (rdot_init - rdot_old) / (len(indices) - 1) * self.TAU > (rdot_old - rdot_new):
                break

        return x_approx.get(), indices

    def mean_subtract_signal(self, signal):
        for chan_index in range(self.C):
            mean_subtract(signal, chan_index, len(signal), self.C)

    def whiten_signal(self, signal):
        for samp_index in range(self.W):
            whiten(signal, self.whitening, samp_index * self.C, samp_index * self.C + self.C, 1)

    def median_subtract_signal(self, signal):
        for samp_index in range(self.W):
            median_subtract(signal, samp_index * self.C, (samp_index + 1) * self.C, 1)

    def parse_spikes(self, indices, offset=0):
        return [(offset + index // self.T, self.template_map[index % self.T]) for index in indices]
    
    def filter_signal(self, signal, num_channels):
        for chan_index in range(num_channels):
            subarray = signal[chan_index: len(signal):num_channels]
            signal[chan_index:len(signal):num_channels] = bandpass(signal[chan_index:len(signal):num_channels], [50, 4000], self.SAMPLING_RATE_MS * 1000)

    def plot_spike(self, spike_index, channel):
        spike_time = self.spike_times[spike_index]
        spike_template = self.spike_templates[spike_index]
        print(self.templates[self.inverse_template_map[spike_template]].shape)
        self.W = self.M
        self.fetch_batch_from_file(Path("C:\\SGL_DATA\\05_31\\imec_raw") / '240531_g0_t0.imec0.ap.bin', start_sample=spike_time - self.M // 2)
        self.whiten_y()
        self.mean_subtract_y()
        assert channel in range(self.C)
        plt.plot(range(-self.M // 2, self.M - self.M // 2 - 1), self.y[channel : len(self.y) : self.C])
        plt.plot(range(-self.M // 2, self.M - self.M // 2 - 1), self.templates[self.inverse_template_map[spike_template]][:, channel] * self.amplitudes[spike_index])
        plt.axvline(x = 0, color = 'b', label = 'axvline - full height')
        plt.show()

    @timerable
    def learn_means_stds_noise(self):
        old_W = self.W
        self.W = 30000
        self.fetch_batch_from_file(Path("C:\\SGL_DATA\\05_31\\imec_raw") / '240531_g0_t0.imec0.ap.bin', start_sample=0)
        self.process_signal(self.y)
        self.means = []
        self.stds = []
        
        for chan_index in range(self.C):
            mean, std = compute_mean_std(self.y, chan_index, len(self.y), self.C)
            self.means.append(mean)
            self.stds.append(std)

        #avg_stds = [self.avg_std(noise, channel) for channel in range(self.C)]
        #print(sum(avg_stds) / len(avg_stds))
        self.W = old_W
        
    def max_std(self, signal, channel):
        # Ensure signal is a CuPy array
        signal = cp.asarray(signal)

        # Calculate the standardized signal
        standardized_signal = (signal[channel : len(signal) : self.C] - self.means[channel]) / self.stds[channel]

        # Compute the max of the absolute values of the standardized signal
        return cp.max(cp.abs(standardized_signal))

    def avg_std(self, signal, channel):
        signal = cp.asarray(signal)
        standardized_signal = (signal[channel : len(signal) : self.C] - self.means[channel]) / self.stds[channel]
        return cp.sum(cp.abs(standardized_signal)) / len(standardized_signal)

    def process_signal(self, signal):
        self.whiten_signal(signal)
        self.mean_subtract_signal(signal)
        self.filter_signal(signal, self.C)

def main() -> int:
    sorter = OSS()
    sorter.run()
    return 0

if __name__ == '__main__':
    sys.exit(main())
    
    

    
        
