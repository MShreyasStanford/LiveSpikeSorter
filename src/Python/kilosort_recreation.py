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
import logging
from colorama import Fore, Style, init
from cupyx.scipy.signal import convolve2d as gpu_convolve2d
import time
from matplotlib.widgets import TextBox
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from pprint import pprint

def stonum(s):
    if '.' in s or 'e' in s or 'E' in s:
        return float(s)
    else:
        return int(s)

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
            value = value.rstrip('\n')
            metadata[key] = value

    return metadata

def covariance_distance_from_identity(matrix):
    # Compute the covariance matrix
    cov_matrix = np.cov(matrix, rowvar=False)

    # Create the identity matrix of the same size
    identity_matrix = np.eye(cov_matrix.shape[0])

    # Compute the Frobenius norm of the difference between the covariance matrix and the identity matrix
    distance = np.linalg.norm(cov_matrix - identity_matrix, ord='fro')

    # Print the distance
    return distance

def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfilt(sos, data)
    return filtered_data

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfilt(sos, data)
    return filtered_data

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfilt(sos, data)
    return filtered_data

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

def reduce(projections, mode='max'):
    if mode == 'max':
        projections_reduced = cp.max(projections, axis=2)
    elif mode == 'sum':
        projections_reduced = cp.sum(projections, axis=2)
    elif mode == 'mean':
        projections_reduced = cp.mean(projections, axis=2)
    elif mode == 'window15':
        # Initialize an array to store the maximum of the sliding window sums
        projections_reduced = cp.zeros((projections.shape[0], projections.shape[1]))
        
        # Slide a window of 15 channels across the original channels
        # We will sum the contents of the window and keep track of the maximum sum encountered
        for start_idx in range(projections.shape[2] - 14):  # Ensure the window doesn't go out of bounds
            window_sum = cp.sum(projections[:, :, start_idx:start_idx+15], axis=2)
            projections_reduced = cp.maximum(projections_reduced, window_sum)
    else:
        raise ValueError("Unsupported mode")

    norms = cp.linalg.norm(projections_reduced, axis=1, keepdims=True)
    norms = cp.where(norms == 0, 1, norms)  # Prevent division by zero
    projections_normalized = projections_reduced / norms

    return projections_normalized


def find_max_projection(projections):
    # Initialize variables to keep track of the maximum value and its indices
    max_value = -cp.inf
    max_template_index = -1
    max_sample_index = -1
    
    # Iterate over each projection to find the maximum value and its indices
    for i, projection in enumerate(projections):
        # Find the index of the maximum value in the current projection
        current_max_value = cp.abs(cp.max(projection))
        if current_max_value > max_value:
            max_value = current_max_value
            max_template_index = i
            max_sample_index = cp.argmax(projection)
    
    return max_template_index, max_sample_index

def conjugate_gradients_gpu(A, y):
    A = cp.asarray(A)
    y = cp.asarray(y)
    n = A.shape[1]
    x = cp.zeros(n)
    r = cp.dot(A.T, y)
    print(f"r = {r.get()}")
    d = r.copy()
    print(d.get())
    delta_new = cp.dot(r, r)
    delta_0 = delta_new.copy()
    print(f"Initial CG residual norm squared = {delta_new}")
    i = 0
    while i < 100 and delta_new > 0.0001 * delta_0:
        q = cp.dot(A.T, cp.dot(A, d))
        alpha = delta_new / cp.dot(d, q)

        x += alpha * d
        #r = cp.dot(A.T, cp.dot(A, x))
        r -= alpha * q
        delta_old = delta_new.copy()
        delta_new = cp.dot(r, r)
        beta = delta_new / delta_old
        d = r + beta * d
        i += 1

        print(f"alpha = {alpha}")
        print(f"beta = {beta}")

    row_space_residual = y - cp.dot(A, x)
    residual_norm = pow(cp.dot(row_space_residual, row_space_residual), 0.5)

    return x

def support(template):
    return np.where(np.any(template != 0, axis=0))[0]

def heatmap(matrix, ax=None):
    matrix **= 3
    if ax is None:
        ax = plt.gca()  # Get current axes if none provided
    cax = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.figure.colorbar(cax, ax=ax)  # Attach colorbar to the specific subplot
    ax.set_title(f'Heatmap {matrix.shape}')


def cosine_similarity_flattened(X_flat, Y_flat):
    dot_product = np.dot(X_flat, Y_flat)
    norm_X = np.linalg.norm(X_flat)
    norm_Y = np.linalg.norm(Y_flat)
    return dot_product / (norm_X * norm_Y)

# Create custom formatter
class ColorFormatter(logging.Formatter):
    level_colors = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        color = self.level_colors.get(record.levelno, Fore.WHITE)
        formatter = logging.Formatter(f"{color}%(levelname)s: %(message)s{Style.RESET_ALL}")
        return formatter.format(record)

class OSS:
    def __init__(self, config):
        # Setup logger
        self.logger = logging.getLogger('LSS Logger')
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(ColorFormatter())

        self.logger.addHandler(ch)

        # Class members that are part of the config
        self.tau: float
        self.thresh: float
        self.start_sample: int
        self.W : int
        self.bin_file: str
        self.oss_training_path: Path
        self.ks_output_dir: Path
        self.bin_meta_file: str
        self.dtype: np.dtype
        self.debug_plots: bool
        self.eventfile: str
        self.output_dir: str

        for key, value in config.items():
            setattr(self, key, value)

        # Class members that are dependent on config
        self.last_processed_sample = self.start_sample

        # Load data from Kilosort
        self.templates = np.load(self.oss_training_path / "templates.npy")
        self.whitening = np.load(self.oss_training_path / "whiteningMat.npy")
        self.channel_mask = np.load(self.oss_training_path / "channelMask.npy")
        self.template_map = np.load(self.oss_training_path / "templateMap.npy")
        self.inverse_template_map = { self.template_map[i] : i for i in range(len(self.template_map)) }
        self.spike_times = np.load(self.ks_output_dir / "spike_times.npy")
        self.spike_templates = np.load(self.ks_output_dir / "spike_templates.npy")
        self.spike_detection_templates = np.load(self.ks_output_dir / "spike_detection_templates.npy")
        self.amplitudes = np.load(self.ks_output_dir / "amplitudes.npy")
        self.ops = np.load(self.ks_output_dir / "ops.npy", allow_pickle=True).item()
        self.logger.debug(f"Loaded ops with keys {self.ops.keys()}")
        self.wPCA = self.ops['wPCA']
        self.n_pcs = self.ops['n_pcs']
        self.T = self.templates.shape[0]
        self.M = self.templates.shape[1]
        self.C = self.templates.shape[2]
        self.logger.info(f"Loaded data from {self.oss_training_path} and {self.ks_output_dir} with parameters T = {self.templates.shape[0]} M = {self.templates.shape[1]} C = {self.templates.shape[2]}")
        self.logger.info(f"Sorting will be performed on channels: {min(self.channel_mask)} to {max(self.channel_mask)}")
        self.logger.info(f"Sorting will be performed on templates: {min(self.template_map)} to {max(self.template_map)}")
        self.logger.info(f"Kilosort sorted spikes from sample {min(self.spike_times)} to {max(self.spike_times)}")

        # GPU copies of Kilosort objects
        self.d_whitening = cp.array(self.whitening)
        self.d_templates = cp.array(self.templates)
        self.logger.debug(f"d_templates.shape = {self.d_templates.shape}")

        # Form the matrix A
        self.construct_A()

        # Parse metadata file for pre-cropped parameters
        metadata = parse_bin_meta_file(self.bin_meta_file)
        self.raw_C = int(metadata["nSavedChans"])   
        self.sampling_rate_hz = int(float(metadata["imSampRate"]))
        x_values = [ -51.5, 35.5, -35.5, 51.5 ]
        self.channel_positions = [ ( x_values[self.channel_mask[i] % 4], (self.channel_mask[i] // 2) * 20 ) for i in range(self.C) ]
        self.logger.info(f"Detected {self.raw_C} channels at a sampling rate of {self.sampling_rate_hz} hz from the binary file")
        self.logger.info(f"Detected channel positions from {self.channel_positions[0:5]} to {self.channel_positions[-5:]}")

        # Additional class members that will be populated later
        self.batch: list
        self.d_batch: cp.array

    def get_events(self):
        self.logger.info(f"Reading events from {self.eventfile}")
        with open(self.eventfile, 'r') as file:
            return [int(line.split(' ')[0]) for line in file]

    def update_config(self, config):
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.logger.warning(f"{key} is not a valid attribute of {self.__class__.__name__}")

    # Returns spike_times, spike_template
    def sort_spikes(self, start, end):
        samples_read = 0
        sorted_spikes = []

        while start + samples_read <= end:
            samples_read += self.fetch_batch_from_file(start_sample=start + samples_read)
            self.logger.info(f"Read {samples_read} samples.")
            self.preprocess_batch()
            _, spike_indices = self.orthogonal_matching_pursuit()

            offset = start + samples_read - self.W

            sorted_spikes += [ (offset + spike_index.get() // self.T, self.template_map[spike_index.get() % self.T]) for spike_index in spike_indices ]

        with open(output_dir / "spikeOutput.txt", 'w') as f:
            for spike_time, spike_template in sorted_spikes:
                f.write(f"{spike_time} {spike_template} 0.0")

        return sorted_spikes

    def kilosort_spikes(self, start, end):
        kilo_spikes = []
        for i, spike_time in enumerate(self.spike_times):
            spike_template = self.spike_templates[i]

            if start <= spike_time <= end and spike_template in self.template_map:
                kilo_spikes.append((spike_time, spike_template))

        kilo_spikes = [ tup for tup in kilo_spikes if start <= tup[0] <= end ]
        return kilo_spikes

    def run(self):
        self.output_f = open(self.output_dir / "spikeOutput.txt", 'w')

        while True:
            self.fetch_batch_from_file()
            self.log_batch_statistics("Batch statistics prior to preprocessing")

            if self.debug_plots:
                self.plot_batch("Before preprocessing")

            self.preprocess_batch()
            self.log_batch_statistics("Batch statistics post preprocessing")

            if self.debug_plots:
                self.plot_batch("After preprocessing")

            _, spike_indices = self.orthogonal_matching_pursuit()
            
            offset = self.last_processed_sample - self.W
            spikes = [ (offset + spike_index.get() // self.T, self.template_map[spike_index.get() % self.T]) for spike_index in spike_indices ]

            for spike_time, spike_template in spikes:
                self.output_f.write(f"{spike_time} {spike_template} 0.0")

    def fetch_batch_from_file(self, start_sample=None):
        if start_sample:
            start = start_sample
        else:
            start = self.last_processed_sample

        self.logger.debug(f"Fetching data from {self.bin_file} from sample {start} to {start + self.W} on {self.C} channels")
        with open(self.bin_file, 'rb') as fidInput:
            self.batch = np.fromfile(fidInput, dtype=np.int16, offset=start * self.raw_C * 2, count=self.raw_C * self.W)

            if self.batch.size == 0 or self.batch.size != self.raw_C * self.W:
                self.logger.error(f"Read {self.batch.size} entries from {self.bin_file}, expected {self.C * self.W}")

            self.last_processed_sample = start + len(self.batch) // self.raw_C
            self.batch = np.reshape(self.batch, (self.raw_C, self.W), order='F')[self.channel_mask, :]
            self.batch = self.batch.flatten(order='F')
            self.batch = self.batch.astype(self.dtype)
            self.d_batch = cp.array(self.batch)
            samples_read = self.batch.size // self.C
            return samples_read

    def construct_A(self):
        # Assume templates_buff is already a flat CuPy array
        templates_buff = self.d_templates.flatten()
        self.D = cp.zeros((self.C, self.M, self.T), dtype=cp.float32)
        self.D2 = cp.zeros((self.C, self.M, self.T), dtype=cp.float32)
        self.D3 = cp.zeros((self.C, self.M, self.T), dtype=cp.float32)

        # Reshape templates_buff to match the dimensions needed for D, D2, D3
        temp_reshaped = templates_buff.reshape(self.C, self.M, self.T)

        # Assign the reshaped template directly to D, D2, D3 with appropriate transpose operations
        self.D = temp_reshaped.transpose(1, 2, 0).flatten()
        self.D2 = temp_reshaped.flatten()
        self.D3 = temp_reshaped.transpose(2, 0, 1).flatten()

    def reconstruct_signal(self, spike_indices, amplitudes):
        return cp.dot(self.extract_submatrix(spike_indices), cp.array(amplitudes)).get()

    def get_batch(self):
        return self.d_batch.get()

    def extract_submatrix(self, indices):
        # Ensure all inputs are CuPy arrays or scalars
        self.D2 = cp.asarray(self.D2)
        indices = cp.asarray(indices)
        T = int(self.T)  # Assuming T is a scalar
        M = int(self.M)  # Assuming M is a scalar
        W = int(self.W)  # Assuming W is a scalar
        C = int(self.C)  # Assuming C is a scalar

        num_cols = len(indices)
        num_rows = W * C
        
        # Calculate sample_indices and template_indices for all indices at once
        sample_indices = indices // T
        template_indices = indices % T
        
        # Create a 2D array of channel-sample indices
        chansamp_indices = cp.arange(M * C, dtype=cp.int32).reshape(M, C)
        
        # Create the output array
        g_A = cp.zeros((num_rows, num_cols), dtype=cp.float32)
        
        # Vectorized operation to fill g_A
        for i, (sample_index, template_index) in enumerate(zip(sample_indices, template_indices)):
            max_m = cp.minimum(M, W - sample_index)
            slice_height = max_m * C
            
            src_indices = template_index * M * C + chansamp_indices[:max_m].ravel()
            dst_indices = cp.arange(int(sample_index) * C, int(sample_index) * C + int(slice_height), dtype=cp.int32)
            
            g_A[dst_indices, i] = self.D2[src_indices]
        
        return g_A

    def orthogonal_matching_pursuit(self):
        residual = self.d_batch
        indices = []
        x_approx = cp.zeros(self.T * self.W)
        rdot_init = cp.dot(residual, residual)
        rdot_old = rdot_init
        rdot_new = rdot_init
        self.logger.info(f"Initial residual norm squared: {rdot_init}")
        max_relative_improvements = []
             
        for iter in range(1000):
            # Project the residual on the columns of A and find the best match
            residual_reshaped = cp.reshape(residual, (self.W, self.C))

            # Compute the convolution, save result for CUDA comparing
            projections = cp.array([convolve(residual_reshaped, d_template) for d_template in self.d_templates])

            if iter == 0:
                self.logger.debug(f"Saving convolution result to convResult_{self.last_processed_sample - self.W}_{self.last_processed_sample}.npy")

            np.save(self.output_dir / f"convResult_{self.last_processed_sample - self.W}_{self.last_processed_sample}.npy", projections.get())

            # Find the maximum so we know which template to add to our reconstruction
            best_template, best_sample = find_max_projection(projections)
            self.logger.debug(f"Highest correlation is found at template {best_template} and sample {best_sample} with correlation {projections[best_template][best_sample]}.")
            indices.append(best_sample * self.T + best_template)

            if iter == 0:
                max_old = projections[best_template][best_sample]
                max_new = projections[best_template][best_sample]

            max_old = max_new
            max_new = projections[best_template][best_sample]       

            max_relative_improvements.append(max_new / max_old)

            # Update the approximation and residual
            A_selected = self.extract_submatrix(indices)
            x_sparse = conjugate_gradients_gpu(A_selected, self.d_batch)
            x_approx[indices] = x_sparse
            self.logger.debug(f"max(x_approx) = {max(x_approx.get())}")
            self.logger.debug(f"x_sparse = {x_sparse.get()}")
            residual = self.d_batch - cp.dot(A_selected, x_sparse)
            rdot_old = rdot_new
            rdot_new = cp.dot(residual, residual)
                
            self.logger.debug(f"OMP Iteration {iter}")
            self.logger.debug(f"Max = {projections[best_template][best_sample]}")
            self.logger.debug(f"Past 5 max ratio avg = {sum(max_relative_improvements[-5:]) / 5}")
            self.logger.debug(f"Max change ratio = {max_new / max_old}")
            self.logger.debug(f"Expression: {(rdot_init - rdot_old) / (len(indices) - 1) * self.tau} > {rdot_old - rdot_new}\n")
                
            if iter >= 5 and sum(max_relative_improvements[-5:]) / 5 >= self.thresh:
                self.logger.debug("Projection exit condition met.")
                break

            if len(indices) > 1 and (rdot_init - rdot_old) / (len(indices) - 1) * self.tau > (rdot_old - rdot_new):
                self.logger.debug("Residual exit condition met.")
                break

        plt.show()
        return x_approx, indices

    ### PREPROCESSING ###
    def preprocess_batch(self):
        self.mean_subtract_batch()
        self.median_subtract_batch()
        self.whiten_batch()
        self.filter_batch()
       # self.mean_subtract_batch()

    def mean_subtract_batch(self):
        means = []
        for chan_index in range(self.C):
            chan = self.d_batch[chan_index : len(self.d_batch) : self.C]
            mean = cp.mean(chan)
            means.append(float(mean.get()))
            chan -= mean

        return means
    
    def median_subtract_batch(self):
        for samp_index in range(self.W):
            samp = self.d_batch[samp_index * self.C : (samp_index + 1) * self.C]
            median = np.median(cp.asnumpy(samp))
            samp -= median

    def whiten_batch(self):
        self.logger.debug(f"Whitening batch")
        for samp_index in range(self.W):
            samp = self.d_batch[samp_index * self.C : (samp_index + 1) * self.C]
            samp = cp.dot(self.d_whitening, samp)
            self.d_batch[samp_index * self.C : (samp_index + 1) * self.C] = samp

    def filter_batch(self):
        self.logger.debug(f"Performing lowpass filter on frequencies below 2000 hz")
        for chan_index in range(self.C):
            chan = self.d_batch[chan_index : len(self.d_batch) : self.C]
            self.d_batch[chan_index : len(self.d_batch) : self.C] = cp.array(lowpass(chan.get(), 2500, self.sampling_rate_hz))

    ### DEBUG METHODS ###
    def plot_batch(self, text, save=False):
        self.batch = self.d_batch.get()
        plt.title(f"Batch [{self.last_processed_sample - self.W}, {self.last_processed_sample}], shape = {self.batch.shape} " + text)
        plt.plot(self.batch)
        if save:
            plt.savefig(Path("C:/", "SGL_DATA", "05_31", "cuda_output") / f"{self.last_processed_sample - self.W}_pybatch.pdf")
            plt.clf()
        else:
            plt.show()

    def fftplot_batch(self, channel, text):
        self.batch = self.d_batch.get()
        fft_vals = np.fft.fft(self.batch[channel : len(self.batch) : self.C])
        power = np.abs(fft_vals)
        frequencies = np.fft.fftfreq(len(power), 1/(self.sampling_rate_hz))
        plt.figure(figsize=(12, 6))
        plt.plot(frequencies[:len(frequencies)//2], power[:len(frequencies)//2])  # Plot only the positive frequencies
        plt.title(text)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()

    def log_batch_statistics(self, text):
        self.logger.debug(text)
        self.logger.debug(f"Batch [{self.last_processed_sample - self.W}, {self.last_processed_sample}]")
        for chan_index in range(min(5, self.C)):
            sublist = self.d_batch[chan_index : len(self.batch) : self.C]
            self.logger.debug(f"mean, std on channel {chan_index}: {cp.mean(sublist)}, {cp.std(sublist)}")

    ### VISUALIZATION METHODS ###
    def plot_channel_around_events(self, events, channel, padding_samples):
        start_sample = max(min(events) - padding_samples, 0)
        end_sample = max(events) + padding_samples
        offset_events = [ event - start_sample for event in events ]
        old_W = self.W
        self.W = end_sample - start_sample
        self.fetch_batch_from_file(start_sample)
        plt.title(f"Plot of raw data from sample {start_sample} to {start_sample + self.W} on channel {self.channel_mask[channel]}")
        channel_amps = np.reshape(self.batch, (self.C, self.W), order='F')[channel]
        plt.ylim((min(channel_amps) - 5, max(channel_amps) + 5))
        plt.plot(channel_amps)
        y_mins = [ -100 ] * len(events)
        y_maxs = [ 100 ] * (len(events))
        plt.vlines(offset_events, y_mins, y_maxs, linestyles="dashed")
        plt.show()
        self.W = old_W

    def plot_templates(self, templates):
        for i, template in enumerate(templates):
            plt.subplot((len(templates) + 1)// 2, 2, i + 1)
            plt.title(f"Template {template}")
            plt.plot(self.templates[self.inverse_template_map[template]])

        if len(templates) == 2:
            distance = cosine_similarity_flattened(self.templates[self.inverse_template_map[templates[0]]].flatten(), self.templates[self.inverse_template_map[templates[1]]].flatten())
            self.logger.info(f"Cosine similarity: {distance}")
            first_template = self.templates[self.inverse_template_map[templates[0]]]
            second_template = self.templates[self.inverse_template_map[templates[1]]]
            first_template_normalized = first_template / np.linalg.norm(first_template)
            second_template_normalized = second_template / np.linalg.norm(second_template)

            for channel in range(self.C):
                self.logger.debug(f"First template channel {channel} norm = {np.linalg.norm(first_template_normalized[:, channel])}")
                self.logger.debug(f"Second template channel {channel} norm = {np.linalg.norm(second_template_normalized[:, channel])}")

        plt.show()

    def bar_graph_ks_spike_templates(self):
        template_counts = { template : 0 for template in self.template_map }
        for spike_template in self.spike_templates:
            if spike_template in self.template_map:
                template_counts[spike_template] += 1

        plt.bar(range(len(template_counts)), list(template_counts.values()), align='center')
        plt.xticks(range(len(template_counts)), list(template_counts.keys()))
        plt.show()

class PSTHGenerator:
    EXTENSION = ".npy"

    def __init__(self, sorter, directory, start_offset_ms, end_offset_ms):
        # Initialize PSTH logger
        self.logger = logging.getLogger('PSTH Logger')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(ColorFormatter())
        self.logger.addHandler(ch)

        # Store parameters
        self.sorter = sorter
        self.directory = directory
        self.start_offset_ms = start_offset_ms
        self.end_offset_ms = end_offset_ms

        self.logger.info(f"PSTH's will be created from {start_offset_ms}ms to {end_offset_ms}ms w.r.t. event times.")
        
        # Create working directory if doesn't exist
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
            self.logger.info(f"Creating working directory {self.directory}")
        else:
            self.logger.info(f"Loading spikes from {self.directory}")
            
        # Load spikes that are existing in the directory
        self.oss_spikes = []
        self.ks_spikes = []
        self.events = []
        self.oss_event_spike_indices = {}
        self.ks_event_spike_indices = {}

        for filename in os.listdir(self.directory):
            # If doesn't end in ".npy"
            if not filename[-len(self.EXTENSION):] == self.EXTENSION:
                continue

            # If filename is not a number
            if not filename[:-len(self.EXTENSION)].isdigit():
                continue

            self.logger.info(f"Loading spikes from {filename}.")

            # Gather spikes
            file_path = self.directory / filename

            event = int(filename[:-len(self.EXTENSION)])
            self.events.append(event)

            oss_spikes_read = np.load(file_path)
            self.oss_spikes.extend(oss_spikes_read)
            self.oss_event_spike_indices[event] = oss_spikes_read

            ks_spikes_read = sorter.kilosort_spikes(event + self.start_offset_ms * 30, event + self.end_offset_ms * 30)
            self.ks_spikes.extend(ks_spikes_read)
            self.ks_event_spike_indices[event] = ks_spikes_read

        self.logger.info(f"Loaded {len(self.events)} events into PSTH object.")
        self.logger.info(f"{len(self.oss_spikes)} spikes detected.")
        
        # Load spike times into dictionary for future computations
        self.oss_template_spike_times = { template : [] for template in sorter.template_map }
        self.ks_template_spike_times = { template : [] for template in sorter.template_map }
        for (sample, template) in self.oss_spikes:
            self.oss_template_spike_times[template].append(sample)
        for (sample, template) in self.ks_spikes:
            self.ks_template_spike_times[template].append(sample)

    def plot_avg(self, templates=None, show=False):
        oss_x, oss_y, ks_x, ks_y = self.avg_psth(templates)
        oss_y = [ x / max(oss_y) if max(oss_y) != 0 else x for x in oss_y ]
        ks_y =  [ x / max(ks_y) if max(ks_y) != 0 else x for x in ks_y]
        l1_dist = sum(abs(a - b) for a, b in zip(oss_y, ks_y))
        plt.title(f"Average PSTH over {len(self.events)} events for templates = {templates}, L1 = {l1_dist}")
        plt.plot(oss_x, oss_y, label='oss')
        plt.plot(ks_x, ks_y, label='ks')
        plt.legend()

        if templates:
            plt.savefig(Path(self.directory, str(templates[0])))
        else:
            plt.savefig(Path(self.directory, "total_average"))

        if show:
            plt.show()
        else:
            plt.clf()       

    def oss_ks_L1(self, templates=None):
        oss_x, oss_y, ks_x, ks_y = self.avg_psth(templates)
        total = sum(oss_y) + sum(ks_y)
        oss_y = [ x / max(oss_y) if max(oss_y) != 0 else x for x in oss_y ]
        ks_y =  [ x / max(ks_y) if max(ks_y) != 0 else x for x in ks_y]
        l1_dist = sum(abs(a - b) for a, b in zip(oss_y, ks_y))
        return l1_dist, total

    def avg_psth(self, templates=None):
        oss_x = None
        oss_y = None
        ks_x = None
        ks_y = None
        for event in self.events:
            # Perform PSTH and update averages
            psth_x, psth_y = self.psth(event, 'oss', templates=templates)

            if oss_x:
                oss_y = [ oss_y[i] + psth_y[i] for i in range(len(psth_y)) ]
            else:
                oss_x = psth_x
                oss_y = psth_y

            psth_x, psth_y = self.psth(event, 'ks', templates=templates)

            if ks_y:
                ks_y = [ ks_y[i] + psth_y[i] for i in range(len(psth_y)) ]
            else:
                ks_x = psth_x
                ks_y = psth_y
        return oss_x, oss_y, ks_x, ks_y

    def psth(self, event_time, source, templates=None, inc=10*30, window_size=50*30, self_spikes=None):
        if source == 'oss':
            spikes = self.oss_event_spike_indices[event_time]
        elif source == 'ks':
            spikes = self.ks_event_spike_indices[event_time]
        elif source == 'self' and self_spikes:
            spikes = self_spikes
        else:
            self.logger.error(f"{source} not a valid spike source for PSTH::psth().")
            exit(0)

        # The list of counts to plot for PSTH
        psth_x = []
        psth_y = []

        start = self.start_offset_ms * 30
        end = self.start_offset_ms * 30 + window_size

        post_count = 0
        pre_count = 0
        
        while end <= self.end_offset_ms * 30:
            window_spike_count = 0
            
            for (sample, template) in spikes:
                if templates and (template not in templates):
                    continue
                
                if sample >= start + event_time and sample <= end + event_time:
                    window_spike_count += 1
                
            psth_x.append(end / 30)
            psth_y.append(window_spike_count)
            start += inc
            end += inc

        return psth_x, psth_y

class CUDAVerifier:
    cuda_raw_file = "raw_output.bin"
    def __init__(self, sorter, directory):
        # Setup logger
        self.logger = logging.getLogger('CUDA Verifier Logger')
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(ColorFormatter())

        self.logger.addHandler(ch)

        # Copy constructor parameters
        self.directory = directory
        self.sorter = sorter

        # Parameters to be populated later
        self.channel_map: list = None
        self.C: int = None
        self.W: int = None

    def verify(self):
        for file in os.listdir(self.directory):
            self.logger.debug(file)
            file_without_extension = file.split(".")[0]
            tokens = file_without_extension.split('_')

            if len(tokens) > 2:
                continue

            start_sample = None
            end_sample = None
            for token in tokens:
                if token.isdigit():
                    if not start_sample:
                        start_sample = int(token)
                    elif not end_sample:
                        end_sample = int(token)

            if start_sample and end_sample:
                self.verify_file(file, start_sample, end_sample)

    def verify_file(self, file, start_sample, end_sample):
        with open(self.directory / file, 'r') as f:
            for i, line in enumerate(f):
                if i > 0:
                    break

                data = [ float(token) for token in line.split(' ') if len(token) > 0 ]
        '''
        with open(self.directory / f"{start_sample}_{end_sample}_means.txt") as means_file:
            for j, line in enumerate(means_file):
                if j > 0:
                    break
                means = [ float(token) for token in line.split(' ') if len(token) > 0 ]
                self.logger.debug(f"CUDA means = {means}")
        '''

        config = {
            "W": end_sample - start_sample
        }

        #heatmap(np.reshape(data, (end_sample - start_sample, len(data) // (end_sample - start_sample))))
        
        plt.plot(data)
        plt.title("CUDA Post-preprocess")
        plt.savefig(self.directory / f"{start_sample}_cudabatch.pdf")
        plt.clf()
        self.sorter.update_config(config)
        self.sorter.fetch_batch_from_file(start_sample=start_sample)
        #python_means = self.sorter.mean_subtract_batch()
        self.sorter.median_subtract_batch()
        self.sorter.whiten_batch()
        self.sorter.filter_batch()
        self.sorter.plot_batch("real", save=True)

        #self.logger.debug(f"Python means = {python_means}")
       # plt.plot([python_means[i] - means[i] for i in range(len(means))])
        plt.show()
                

    def psth(self, event_time, spikes, templates=None, inc=10*30, window_size=100*30):
        if len(spikes) == 0:
            return None, None

        if min([spike[0] for spike in spikes]) >= event_time + 400 * 30:
            return None, None

        if max([spike[0] for spike in spikes]) <= event_time - 400 * 30:
            return None, None

        # The list of counts to plot for PSTH
        psth_x = []
        psth_y = []

        start = -300 * 30
        end = start + window_size
        
        while end <= 400 * 30:
            window_spike_count = 0
            
            for (sample, template) in spikes:
                if templates and (template not in templates):
                    continue
                
                if sample >= start + event_time and sample <= end + event_time:
                    window_spike_count += 1
                
            psth_x.append(end / 30)
            psth_y.append(window_spike_count)
            start += inc
            end += inc

        return psth_x, psth_y

    def verify_psth(self, file):
        with open(self.directory / file, 'r') as f:
            spikes = [(int(line.split(' ')[0]), int(line.split(' ')[1])) for line in f]
            events = self.sorter.get_events()

            for event in events:
                psth_x, psth_y = self.psth(event, spikes)

                if not psth_x or not psth_y:
                    continue

                print(psth_x)
                print(psth_y)

                plt.title("CUDA")
                plt.plot(psth_x, psth_y)
                plt.show()

    def find_residual_threshold(self, file, thresholds=[0]):
        with open(self.directory / file, 'r') as f:
            self.logger.info(f"Reading in LSS spikes from {self.directory / file}.")
            spikes = [(stonum(line.split(',')[0]), stonum(line.split(',')[1]), stonum(line.split(',')[2])) for line in f]

        events = self.sorter.get_events()

        self.logger.info(f"Reading in KS spikes from {self.sorter.ks_output_dir}.")
        spike_templates = np.load(self.sorter.ks_output_dir / "spike_templates.npy")
        spike_times = np.load(self.sorter.ks_output_dir / "spike_times.npy")
        #spikes = tuple(zip(spike_times, spike_detection_templates, [0.0] * len(spike_times)))
        kilo_spikes = tuple(zip(spike_times, spike_templates))
        
        events = [ event for event in events if 59998208 < event < 83984384 ]
        first_event = events[0]
        last_event = events[len(events) - 1]
        min_index = min([ i for i in range(len(spike_times)) if spike_times[i] > first_event - 1000 * 30 and spike_times[i] < last_event + 1000 * 30])
        max_index = max([ i for i in range(len(spike_times)) if spike_times[i] > first_event - 1000 * 30 and spike_times[i] < last_event + 1000 * 30])
        kilo_spikes = kilo_spikes[min_index : max_index]
        #kilo_spikes = [ spike for spike in kilo_spikes if spike[1] in self.sorter.template_map ]

        ks_x = None
        ks_y = None

        for event in events:
            psth_x, psth_y = self.psth(event, kilo_spikes)

            if not psth_y:
                continue

            if ks_y:
                ks_y = [ ks_y[i] + psth_y[i] for i in range(len(psth_y)) ]
            else:
                #ks_x = psth_x
                ks_x = [ psth_x[i] for i in range(len(psth_x)) ]
                ks_y = psth_y

        for i, threshold in enumerate(thresholds):
            spikes_cropped = [(spike[0], spike[1]) for spike in spikes if spike[2] >= threshold]
            self.logger.info(f"Threshold of {threshold} has {len(spikes_cropped)} spikes.")
            oss_x = None
            oss_y = None

            for event in events:
                self.logger.info(f"Processing event {event}")

                psth_x, psth_y = self.psth(event, spikes_cropped)
                if not psth_y:
                    continue
                    
                if oss_x:
                    oss_y = [ oss_y[i] + psth_y[i] for i in range(len(psth_y)) ]
                else:
                    oss_x = psth_x
                    oss_y = psth_y

            ks_max = max(ks_y)
            oss_max = max(oss_y)
            ks_y = [ x / ks_max for x in ks_y ]
            oss_y = [ x / oss_max for x in oss_y ]
            self.logger.debug("Adding subplot")
            #plt.subplot(len(thresholds), 1, i + 1)
            plt.subplot(len(thresholds), 2, 2 * i + 1)
            plt.plot(oss_x, oss_y)
            plt.title(f"Live Spike Sorter Avg PSTH, Normalized L_infty = 1")
            plt.subplot(len(thresholds), 2, 2 * i + 2)
            plt.plot(ks_x, ks_y)
            plt.title("Kilosort Avg PSTH, Normalized L_infty = 1")
        self.logger.debug("Showing plot!")
        plt.show()

    def verify_avg_psth(self, file, residual_thresh):
        with open(self.directory / file, 'r') as f:
            spikes = [(int(line.split(' ')[0]), int(line.split(' ')[1])) for line in f if float(line.split(' ')[2]) >= residual_thresh]
            events = self.sorter.get_events()

        KS_OUTPUT_DIR = Path("C:/", "SGL_DATA", "05_31", "imec_raw", "kilosort4")
        spike_templates = np.load(KS_OUTPUT_DIR / "spike_templates.npy")
        spike_times = np.load(KS_OUTPUT_DIR / "spike_times.npy")
        kilo_spikes = tuple(zip(spike_times, spike_templates))

        k = 50
        first_event = events[len(events) - k]
        last_event = events[len(events) - 1]
        min_index = min([ i for i in range(len(spike_times)) if spike_times[i] > first_event - 1000 * 30 and spike_times[i] < last_event + 1000 * 30])
        max_index = max([ i for i in range(len(spike_times)) if spike_times[i] > first_event - 1000 * 30 and spike_times[i] < last_event + 1000 * 30])
        kilo_spikes = kilo_spikes[min_index : max_index]

        oss_x = None
        oss_y = None
        ks_x = None
        ks_y = None

        for event in events[len(events) - k : ]:
            print(event)
            # Perform PSTH and update averages
            psth_x, psth_y = self.psth(event, spikes)

            if not psth_y:
                continue
                
            if oss_x:
                oss_y = [ oss_y[i] + psth_y[i] for i in range(len(psth_y)) ]
            else:
                oss_x = psth_x
                oss_y = psth_y

            psth_x, psth_y = self.psth(event, kilo_spikes)

            if not psth_y:
                continue

            if ks_y:
                ks_y = [ ks_y[i] + psth_y[i] for i in range(len(psth_y)) ]
            else:
                ks_x = psth_x
                ks_y = psth_y
        
        return oss_x, oss_y, ks_x, ks_y
        plt.subplot(211)
        plt.plot(oss_x, oss_y)
        plt.title("CUDA")
        plt.subplot(212)
        plt.plot(ks_x, ks_y)
        plt.title("KS")
        plt.show()

    def verify_single_neuron_psths(self, file, thresh):
        import os
        import pickle
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import TextBox

        # Define pickle file name based on the file argument.
        psth_pickle_file = self.directory / f"{file}_psth_data.pkl"

        # If the pickle file exists, load the PSTH data.
        if os.path.exists(psth_pickle_file):
            self.logger.info(f"Loading PSTH data from pickle: {psth_pickle_file}")
            with open(psth_pickle_file, 'rb') as pf:
                psth_data = pickle.load(pf)
        else:
            psth_data = {}  # Dictionary to store PSTH data for each template

            # -----------------------------
            # Process OSS spike data
            with open(self.directory / file, 'r') as f:
                print(f"Loading LSS spikes from {self.directory / file}.")
                spikes = [
                    (stonum(line.split(',')[0]),
                     stonum(line.split(',')[1]),
                     stonum(line.split(',')[2]))
                    for line in f
                ]
            # Keep only the first two values (spike time and template)
            spikes = [(s, t) for s, t, _ in spikes]

            # -----------------------------
            # Process events and Kilosort spike data
            events = self.sorter.get_events()
            self.logger.info(f"Reading in KS spikes from {self.sorter.ks_output_dir}.")
            KS_OUTPUT_DIR = self.sorter.ks_output_dir
            spike_detection_templates = np.load(KS_OUTPUT_DIR / "spike_detection_templates.npy")
            spike_templates = np.load(KS_OUTPUT_DIR / "spike_templates.npy")
            spike_times = np.load(KS_OUTPUT_DIR / "spike_times.npy")
            kilo_spikes = list(zip(spike_times, spike_detection_templates)) # NOTICE CHANGE THIS FOR UNCLUSTERED VS CLUSTERED
            min_oss_spike = min([s for s, t in spikes])
            max_oss_spike = max([s for s, t in spikes])
            print(f"# kilo spikes = {len(kilo_spikes)}")
            print(f"kilo spikes range = {min(spike_times)} to {max(spike_times)}")
            print(f"kilo template range = {min(spike_detection_templates)} to {max(spike_detection_templates)}")
            print(f"oss spike time range = {min_oss_spike} to {max_oss_spike}")

            # Filter events to a specific range
            events = [event for event in events if min_oss_spike < event < max_oss_spike]
            events = sorted(random.sample(events, min(25, len(events))))
            print(f"Processing {len(events)} events.")
            first_event = events[0]
            last_event = events[-1]
            print(f"first_event = {first_event}")
            print(f"last_event = {last_event}")
            min_time = first_event - 1000 * 30
            max_time = last_event + 1000 * 30
            kilo_spikes = [(s, t) for s, t in kilo_spikes if min_time < s < max_time]

            # -----------------------------
            # Loop over each template to compute PSTH data
            for template in range(1, max(spike_detection_templates) + 1):
                self.logger.info(f"Processing template {template}")

                # Process Kilosort spikes for the current template
                kilo_spikes_cropped = [(s, t) for s, t in kilo_spikes if t == template]
                ks_x, ks_y = None, None
                for event in events:
                    psth_x, psth_y = self.psth(event, kilo_spikes_cropped)
                    if psth_y is None:
                        continue
                    if ks_y:
                        ks_y = [ks_y[i] + psth_y[i] for i in range(len(psth_y))]
                    else:
                        ks_x, ks_y = psth_x, psth_y

                # Process OSS spikes for the current template
                oss_x, oss_y = None, None
                spikes_cropped = [(s, t) for s, t in spikes if t == template]
                for event in events:
                    psth_x, psth_y = self.psth(event, spikes_cropped)
                    if psth_y is None:
                        continue
                    if oss_y:
                        oss_y = [oss_y[i] + psth_y[i] for i in range(len(psth_y))]
                    else:
                        oss_x, oss_y = psth_x, psth_y

                # Determine a standard x-axis from the first valid PSTH
                standard_x = None
                if ks_x is not None:
                    standard_x = ks_x.copy()
                elif oss_x is not None:
                    standard_x = oss_x.copy()

                # Handle cases where PSTH returns None by plotting zeros
                if ks_x is None or ks_y is None:
                    if standard_x is not None:
                        ks_x = standard_x.copy()
                        ks_y = [0] * len(standard_x)
                        self.logger.warning(f"Kilosort PSTH for template {template} is None. Plotting zeros.")
                    else:
                        self.logger.warning(f"Kilosort PSTH for template {template} is None and standard_x is not set. Skipping.")
                        continue

                if oss_x is None or oss_y is None:
                    if standard_x is not None:
                        oss_x = standard_x.copy()
                        oss_y = [0] * len(standard_x)
                        self.logger.warning(f"LSS PSTH for template {template} is None. Plotting zeros.")
                    else:
                        self.logger.warning(f"LSS PSTH for template {template} is None and standard_x is not set. Skipping.")
                        continue

                if ks_x != standard_x:
                    self.logger.warning(f"Kilosort PSTH x-axis for template {template} does not match standard_x. Skipping.")
                    continue

                if oss_x != standard_x:
                    self.logger.warning(f"LSS PSTH x-axis for template {template} does not match standard_x. Skipping.")
                    continue

                # Store the computed PSTH data for this template.
                psth_data[template] = {'standard_x': standard_x, 'oss_y': oss_y, 'ks_y': ks_y}

            # After processing all templates, pickle the PSTH data.
            with open(psth_pickle_file, 'wb') as pf:
                pickle.dump(psth_data, pf)

        # -----------------------------
        # Plotting: create 3 subplots (OSS, Kilosort, and Average PSTH)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 15), sharex=True)
        ax1.set_title("Live Spike Sorter")
        ax2.set_title("Kilosort")
        ax3.set_title("Average PSTH (Normalized)")

        lines_to_templates = {}
        all_oss = []
        all_ks = []
        standard_x = None

        # Iterate over stored PSTH data sorted by template.
        for template in sorted(psth_data.keys()):
            data = psth_data[template]
            curr_x = data['standard_x']
            curr_oss_y = data['oss_y']
            curr_ks_y = data['ks_y']

            if standard_x is None:
                standard_x = curr_x.copy()

            # Plot OSS data on ax1 and Kilosort data on ax2.
            line1, = ax1.plot(curr_x, curr_oss_y, label=f"Template {template}", picker=5, alpha=0.7)
            lines_to_templates[line1] = str(template)
            line2, = ax2.plot(curr_x, curr_ks_y, label=f"Template {template}", picker=5, alpha=0.7)
            lines_to_templates[line2] = str(template)

            # Collect the PSTH values for average computation.
            all_oss.append(curr_oss_y)
            all_ks.append(curr_ks_y)

        # Compute the average PSTH across templates.
        if all_oss and all_ks:
            avg_oss = np.mean(all_oss, axis=0)
            avg_ks = np.mean(all_ks, axis=0)
            # Normalize so that each average PSTH has maximum = 1.
            if np.max(avg_oss) != 0:
                avg_oss = avg_oss / np.max(avg_oss)
            if np.max(avg_ks) != 0:
                avg_ks = avg_ks / np.max(avg_ks)
            ax3.plot(standard_x, avg_oss, label="Average OSS", alpha=0.9)
            ax3.plot(standard_x, avg_ks, label="Average KS", alpha=0.9)
            ax3.legend()
        else:
            self.logger.warning("No valid PSTH data for average computation.")

        # -----------------------------
        # Add a TextBox and annotation for interactive template selection.
        plt.subplots_adjust(right=0.8)
        text_box_ax = fig.add_axes([0.85, 0.9, 0.1, 0.05])
        text_box = TextBox(text_box_ax, 'Enter Template:', initial='All')
        annotation = ax1.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annotation.set_visible(False)

        def on_pick(event):
            clicked_line = event.artist
            clicked_template = lines_to_templates.get(clicked_line, None)
            if clicked_template is None:
                return
            if hasattr(event, 'mouseevent') and event.mouseevent.xdata and event.mouseevent.ydata:
                x_val = event.mouseevent.xdata
                y_val = event.mouseevent.ydata
                annotation.xy = (x_val, y_val)
                annotation.set_text(f"Template: {clicked_template}")
                annotation.set_visible(True)
                fig.canvas.draw_idle()
            isolate_template(clicked_template)

        def isolate_template(template):
            for line, tmpl in lines_to_templates.items():
                if tmpl == template:
                    line.set_alpha(1.0)
                else:
                    line.set_alpha(0.1)
            text_box.set_val(str(template))
            fig.canvas.draw_idle()

        def reset_visibility():
            for line in lines_to_templates.keys():
                line.set_alpha(0.7)
            annotation.set_visible(False)
            fig.canvas.draw_idle()

        def submit(text):
            label = text.strip()
            if label.lower() == 'all' or label == '':
                reset_visibility()
            elif label in lines_to_templates.values():
                isolate_template(label)
            else:
                self.logger.warning(f"Template '{label}' not found.")
                annotation.xy = (0, 0)
                annotation.set_text(f"Template '{label}' not found.")
                annotation.set_visible(True)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('pick_event', on_pick)
        text_box.on_submit(submit)

        self.logger.debug("Showing plot!")
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show()


    def plot_residual_statistics(self, file):
        with open(self.directory / file, 'r') as f:
            data = [float(line) for line in f]
        data = sorted(data)
        plt.hist(data, bins=150)
        plt.show()
        '''
        cdf = np.arange(1, len(data) + 1) / len(data)
        plt.figure(figsize=(8, 6))
        plt.plot(cdf, data, marker='.', linestyle='none')
        plt.xlabel('Data')
        plt.ylabel('CDF')
        plt.title('Cumulative Distribution Function (CDF)')
        plt.grid(True)
        plt.show()
        '''

    def plot_cosine_sim_statistics(self, file):
        with open(self.directory / file, 'r') as f:
            data = [float(line.split()[1]) for line in f]
        data = sorted(data)
        plt.hist(data, bins=1000)
        plt.show()

    def compare_cosine_sims(self, file):
        with open(self.directory / file, 'r') as f:
            data = [ (int(line.split()[0]), float(line.split()[1]), float(line.split()[2])) for line in f ]

        T = len(self.sorter.template_map)
        cos_sims = { template: [] for template in self.sorter.template_map }
        dots = { template: [] for template in self.sorter.template_map }
        for template, cos_sim, dot in data:
            cos_sims[template].append(cos_sim)
            dots[template].append(dot)


        for i, template in enumerate(self.sorter.template_map):
            plt.subplot(T, 2, 2 * i + 1)
            plt.title(f"Template {template} cosine similarities")
            plt.hist(cos_sims[template], bins=1000)

            plt.subplot(T, 2, 2 * i + 2)
            plt.title(f"Template {template} dots")
            plt.hist(dots[template], bins=1000)

        plt.show()

    def plot_spike_statistics(self, file):
        with open(self.directory / file, 'r') as f:
            data = [(int(line.split(' ')[0]), int(line.split(' ')[1])) for line in f if len(line.split(' ')) == 2]

        numSpikes = [datum[0] for datum in data]
        numSamples = [datum[1] for datum in data]
        plt.scatter(numSamples, numSpikes)
        plt.show()

    def plot_conv_result(self):
        for file in os.listdir(self.directory):
            self.logger.debug(file)
            file_without_extension = file.split(".")[0]
            tokens = file_without_extension.split('_')

            if tokens[0] != "convResult":
                continue

            start_sample = None
            end_sample = None
            for token in tokens:
                if token.isdigit():
                    if not start_sample:
                        start_sample = int(token)
                    elif not end_sample:
                        end_sample = int(token)

            if start_sample and end_sample:
                with open(self.directory / file, 'r') as f:
                    for i, line in enumerate(f):
                        if i > 0:
                            break

                        data = [ float(token) for token in line.split(' ') if len(token) > 0 ]
                        if (len(data) // self.sorter.T) * self.sorter.T != len(data):
                            continue

                        data = np.array(data)
                        data = np.reshape(data, ( len(data) // self.sorter.T , self.sorter.T))
                        heatmap(data)

    def plot_ax(self):
        for file in os.listdir(self.directory):
            self.logger.debug(file)
            file_without_extension = file.split(".")[0]
            tokens = file_without_extension.split('_')

            if tokens[0] != "Ax":
                continue

            start_sample = None
            end_sample = None
            for token in tokens:
                if token.isdigit():
                    if not start_sample:
                        start_sample = int(token)
                    elif not end_sample:
                        end_sample = int(token)

            if start_sample and end_sample:
                with open(self.directory / file, 'r') as f:
                    for i, line in enumerate(f):
                        if i > 0:
                            break
                        data = [ float(token) for token in line.split(' ') if len(token) > 0 ]

                        plt.plot(data)
                        plt.show()

    def plot_A(self):
        for file in os.listdir(self.directory):
            self.logger.debug(file)
            file_without_extension = file.split(".")[0]
            tokens = file_without_extension.split('_')

            if tokens[0] != "A":
                continue

            start_sample = None
            end_sample = None
            numRows = None
            numCols = None
            for token in tokens:
                if token.isdigit():
                    if not start_sample:
                        start_sample = int(token)
                    elif not end_sample:
                        end_sample = int(token)
                    elif not numRows:
                        numRows = int(token)
                    elif not numCols:
                        numCols = int(token)

            if start_sample and end_sample:
                with open(self.directory / file, 'r') as f:
                    for i, line in enumerate(f):
                        if i > 0:
                            break
                        data = [ float(token) for token in line.split(' ') if len(token) > 0 ]
                        data = np.array(data)
                        numRows = self.sorter.C * (end_sample - start_sample)
                        data = np.reshape(data, (numRows, numCols), order='F')
                        plt.plot(data)
                        plt.show()
    def verify_conv(self):
        for file in os.listdir(self.directory):
            self.logger.debug(file)
            file_without_extension = file.split(".")[0]
            tokens = file_without_extension.split('_')

            if tokens[0] != "convResult":
                continue

            start_sample = None
            end_sample = None
            for token in tokens:
                if token.isdigit():
                    if not start_sample:
                        start_sample = int(token)
                    elif not end_sample:
                        end_sample = int(token)

            if not start_sample or not end_sample:
                continue

            config = {
                "W": end_sample - start_sample
            }

            self.sorter.update_config(config)
            self.sorter.fetch_batch_from_file(start_sample=start_sample)
            self.sorter.preprocess_batch()
            batch_reshaped = cp.reshape(self.sorter.d_batch, (self.sorter.W, self.sorter.C))
            projections = cp.array([convolve(batch_reshaped, d_template) for d_template in self.sorter.d_templates])
            
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # Plot Python heatmap
            ax = axs[0, 0]
            ax.set_title("Python")
            heatmap(projections.get(), ax=ax)

            # Plot CUDA heatmap
            with open(self.directory / file, 'r') as f:
                for i, line in enumerate(f):
                    if i > 0:
                        break

                    data = [ float(token) for token in line.split(' ') if len(token) > 0 ]
                    if (len(data) // self.sorter.T) * self.sorter.T != len(data):
                        continue

                    data = np.array(data)
                    data = np.reshape(data, ( len(data) // self.sorter.T , self.sorter.T)).T
                    ax = axs[0, 1]
                    ax.set_title("CUDA")
                    heatmap(data, ax=ax)

            # Plot Fetched batch
            ax = axs[1, 0]
            ax.set_title("Fetched batch")
            for i, chan_index in enumerate(range(self.sorter.C)):
                batch_channel = self.sorter.get_batch()[ chan_index::self.sorter.C ]
                ax.plot(batch_channel ** 3 + 50 * i)

            # Optionally hide the unused subplot
            axs[1, 1].axis('off')

            plt.tight_layout()
            plt.show()

    def compare_ks_oss_spikes_in_range(self, file, start_samp, end_samp):
        end_samp = int(end_samp)
        start_samp = int(start_samp)
        self.logger.debug(f"Comparing KS and LSS in interval of size {end_samp - start_samp} samples.")

        with open(self.directory / file, 'r') as f:
            oss_spikes = [
                (int(line.split(' ')[0]), int(line.split(' ')[1]), float(line.split(' ')[2]), float(line.split(' ')[3]))
                for line in f
            ]

        oss_spikes_filtered = [ spike for spike in oss_spikes if start_samp <= spike[0] <= end_samp ]
        kilo_spikes = list(zip(self.sorter.spike_times, self.sorter.spike_templates, self.sorter.amplitudes))
        kilo_spikes = [spike for spike in kilo_spikes if start_samp < spike[0] < end_samp]
        kilo_spikes = [spike for spike in kilo_spikes if spike[1] in self.sorter.template_map]
        self.logger.debug(oss_spikes_filtered[0:5])
        self.logger.debug(kilo_spikes[0:5])
        self.logger.info(f"LSS found {len(oss_spikes_filtered)} spikes from samples {start_samp} to {end_samp}.")
        self.logger.info(f"KS found {len(kilo_spikes)} spikes from samples {start_samp} to {end_samp}.")

        # Get spike template counts
        oss_templates = { template : 0 for template in self.sorter.template_map }
        ks_templates = { template : 0 for template in self.sorter.template_map }

        for _, spike_template, _, _ in oss_spikes_filtered:
            oss_templates[spike_template] += 1
        for _, spike_template, _ in kilo_spikes:
            ks_templates[spike_template] += 1

        # Reconstruct signals
        oss_spike_indices = [ (spike[0] - start_samp) * self.sorter.T + spike[1] for spike in oss_spikes_filtered ]
        oss_amplitudes = [ spike[3] for spike in oss_spikes_filtered ]
        oss_reconstructed = self.sorter.reconstruct_signal(oss_spike_indices, oss_amplitudes)
        ks_spike_indices = [ (spike[0] - start_samp) * self.sorter.T + spike[1] for spike in kilo_spikes ]
        ks_amplitudes = [ spike[2] for spike in kilo_spikes ]
        ks_reconstructed = self.sorter.reconstruct_signal(ks_spike_indices, ks_amplitudes)

        # Fetch batch for comparison purposes
        config = {
            "W": end_samp - start_samp,
            "start_sample": start_samp
        }

        self.sorter.update_config(config)
        self.sorter.fetch_batch_from_file(start_sample=start_samp)
        self.sorter.preprocess_batch()

        # Plot the reconstructions alongside the batch
        plt.subplot(221)
        plt.title("LSS")
        for i, chan_index in enumerate(range(self.sorter.C)):
            oss_channel = oss_reconstructed[ chan_index : len(oss_reconstructed) : self.sorter.C ]
            plt.plot(oss_channel * 10 + 10 * i)

        plt.subplot(222)
        plt.title("KS")
        for i, chan_index in enumerate(range(self.sorter.C)):
            ks_channel = ks_reconstructed[ chan_index : len(ks_reconstructed) : self.sorter.C ]
            plt.plot(ks_channel * 3 + 25 * i)

        plt.subplot(223)
        plt.title("Fetched")
        for i, chan_index in enumerate(range(self.sorter.C)):
            batch_channel = self.sorter.get_batch()[ chan_index : : self.sorter.C ]
            plt.plot(batch_channel ** 3 + 50 * i)
        plt.show()


    def generate_psth_l1_differences(self, file):
        """
        Generates an L1 distance matrix between OSS and Kilosort PSTHs for each pair of templates.
        If both PSTHs are all zeros (norms are zero), sets L1 distance to infinity.
        
        Parameters:
        - file (str or Path): Path to the spike data file.
        
        Returns:
        - L1 (pd.DataFrame): A DataFrame where L1[S][T] represents the L1 distance between
                             the normalized PSTH of OSS template S and Kilosort template T.
        """
        # Load spike data from the file
        with open(self.directory / file, 'r') as f:
            spikes = [
                (int(line.split(' ')[0]), int(line.split(' ')[1]), float(line.split(' ')[2]))
                for line in f
            ]
        events = self.sorter.get_events()

        # Load Kilosort spike data
        KS_OUTPUT_DIR = Path("C:/", "SGL_DATA", "05_31", "imec_raw", "kilosort4")
        spike_templates = np.load(KS_OUTPUT_DIR / "spike_templates.npy")
        spike_times = np.load(KS_OUTPUT_DIR / "spike_times.npy")
        kilo_spikes = list(zip(spike_times, spike_templates))
        k = 50
        first_event = events[-k]
        last_event = events[-1]
        min_time = first_event - 1000 * 30
        max_time = last_event + 1000 * 30
        kilo_spikes = [(s, t) for s, t in kilo_spikes if min_time < s < max_time]
        kilo_spikes = [spike for spike in kilo_spikes if spike[1] in self.sorter.template_map]
        spikes = [(spike[0], spike[1]) for spike in spikes if spike[2] >= 0]
        psth_len = None  # Not used in this function, can be removed or utilized as needed

        # Initialize list of templates
        list_templates = sorted(self.sorter.template_map)
        num_templates = len(list_templates)

        # Initialize dictionaries to store normalized PSTHs
        oss_psths = {}
        ks_psths = {}

        # Initialize standard_x to ensure consistent x-axis across all PSTHs
        standard_x = None

        # Precompute OSS and KS PSTHs for all templates
        for template in list_templates:
            self.logger.info(f"Processing LSS PSTH for template {template}")
            # Process OSS spikes for template S
            spikes_cropped = [(s, t) for s, t in spikes if t == template]
            oss_x, oss_y = None, None
            for event in events[-k:]:
                psth_x, psth_y = self.psth(event, spikes_cropped)
                if psth_y is None:
                    continue
                if oss_y is not None:
                    oss_y = np.add(oss_y, psth_y)
                else:
                    oss_x, oss_y = psth_x, psth_y

            # If PSTH is None, replace y with zeros
            if oss_y is None:
                if standard_x is not None:
                    oss_x = standard_x.copy()
                    oss_y = np.zeros_like(standard_x)
                    self.logger.warning(f"LSS PSTH for template {template} is None. Replaced with zeros.")
                else:
                    self.logger.warning(f"LSS PSTH for template {template} is None and standard_x is not set. Skipping.")
                    continue

            # Determine standard_x from the first valid PSTH
            if standard_x is None:
                standard_x = oss_x.copy()

            # Ensure oss_x matches standard_x
            if not np.array_equal(oss_x, standard_x):
                self.logger.warning(f"LSS PSTH x-axis for template {template} does not match standard_x. Skipping.")
                continue

            # Normalize oss_y to have max y-value of 1
            max_y = np.max(oss_y)
            if max_y > 0:
                oss_y_normalized = oss_y / max_y
            else:
                oss_y_normalized = oss_y  # Remains zero

            oss_psths[template] = oss_y_normalized

            self.logger.info(f"Processing KS PSTH for template {template}")
            # Process KS spikes for template T
            kilo_spikes_cropped = [(s, t) for s, t in kilo_spikes if t == template]
            ks_x, ks_y = None, None
            for event in events[-k:]:
                psth_x, psth_y = self.psth(event, kilo_spikes_cropped)
                if psth_y is None:
                    continue
                if ks_y is not None:
                    ks_y = np.add(ks_y, psth_y)
                else:
                    ks_x, ks_y = psth_x, psth_y

            # If PSTH is None, replace y with zeros
            if ks_y is None:
                if standard_x is not None:
                    ks_x = standard_x.copy()
                    ks_y = np.zeros_like(standard_x)
                    self.logger.warning(f"KS PSTH for template {template} is None. Replaced with zeros.")
                else:
                    self.logger.warning(f"KS PSTH for template {template} is None and standard_x is not set. Skipping.")
                    continue

            # Ensure ks_x matches standard_x
            if not np.array_equal(ks_x, standard_x):
                self.logger.warning(f"KS PSTH x-axis for template {template} does not match standard_x. Skipping.")
                continue

            # Normalize ks_y to have max y-value of 1
            max_y = np.max(ks_y)
            if max_y > 0:
                ks_y_normalized = ks_y / max_y
            else:
                ks_y_normalized = ks_y  # Remains zero

            ks_psths[template] = ks_y_normalized

        # Verify that we have PSTHs for all templates
        valid_templates = list(set(oss_psths.keys()).intersection(set(ks_psths.keys())))
        if not valid_templates:
            self.logger.error("No valid PSTHs were generated. Returning empty L1 matrix.")
            return pd.DataFrame()

        # Initialize the L1 distance matrix
        L1 = pd.DataFrame(
            data=np.zeros((len(valid_templates), len(valid_templates))),
            index=valid_templates,
            columns=valid_templates
        )

        self.logger.info("Computing L1 distances between LSS and KS PSTHs.")

        # Compute L1 distances
        for S in valid_templates:
            oss_psth_S = oss_psths[S]
            for T in valid_templates:
                ks_psth_T = ks_psths[T]
                # Check if both PSTHs are all zeros
                if np.all(oss_psth_S == 0) and np.all(ks_psth_T == 0):
                    l1_distance = np.inf  # Set to infinity
                else:
                    # Compute L1 distance: sum of absolute differences
                    l1_distance = np.sum(np.abs(oss_psth_S - ks_psth_T))
                L1.at[S, T] = l1_distance

        self.logger.info("Completed L1 distance computation.")

        return L1

def visualize_l1_matrix_no_annotations(L1, title="L1 Distance Matrix", cmap="viridis", figsize=(12, 10), 
                                       linewidths=0.5, cbar=True, square=False, clustering=False):
    """
    Visualizes the L1 distance matrix using a heatmap without numerical annotations.
    Infinitely large distances are masked and shown in a distinct color.
    
    Parameters:
    - L1 (pd.DataFrame): The L1 distance matrix with OSS templates as rows and KS templates as columns.
    - title (str): The title of the heatmap.
    - cmap (str): The color map to use for the heatmap.
    - figsize (tuple): Figure size in inches.
    - linewidths (float): Width of the lines that will divide each cell.
    - cbar (bool): Whether to draw a color bar.
    - square (bool): If True, set the Axes aspect to "equal" so each cell is square.
    - clustering (bool): If True, perform hierarchical clustering on both axes.
    
    Returns:
    - None
    """
    # Replace inf values with NaN for masking
    L1_masked = L1.replace(np.inf, np.nan)

    # Create a mask for the inf values
    mask = L1_masked.isnull()

    plt.figure(figsize=figsize)
    
    if clustering:
        # Generate clustermap instead of a simple heatmap
        sns.clustermap(
            L1_masked,
            cmap=cmap,
            figsize=figsize,
            linewidths=linewidths,
            cbar=cbar,
            standard_scale=None,  # No standard scaling
            metric="euclidean",    # Distance metric for clustering
            method="average",      # Linkage method
            dendrogram_ratio=(.1, .2),
            cbar_pos=(0.02, 0.8, 0.05, 0.18),
            mask=mask  # Mask inf values
        )
        plt.title(title, y=1.05)  # Adjust title position
    else:
        # Create a heatmap without clustering
        sns.heatmap(
            L1_masked,
            cmap=cmap,
            annot=False,         # Remove annotations
            linewidths=linewidths,
            cbar=cbar,
            square=square,
            xticklabels=True,
            yticklabels=True,
            mask=mask,           # Mask inf values
            vmax=np.nanmax(L1_masked),  # Set vmax to exclude masked values
            vmin=np.nanmin(L1_masked)   # Set vmin to exclude masked values
        )
        
        # Overlay a layer to color the masked (infinite) cells
        # Choose a color that stands out, e.g., red
        ax = plt.gca()
        for i in range(L1.shape[0]):
            for j in range(L1.shape[1]):
                if np.isinf(L1.iloc[i, j]):
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='red', alpha=0.3))
        
        # Create a custom legend entry for infinity
        from matplotlib.patches import Patch
        handles = [Patch(facecolor='red', edgecolor='red', alpha=0.3, label='Infinite Distance')]
        ax.legend(handles=handles, loc='upper right')
        
        plt.title(title)
        plt.xlabel("Kilosort Templates (T)")
        plt.ylabel("OSS Templates (S)")
    
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.tight_layout()
    plt.show()

def generate_cuda_test_data(sorter, dir):
    batch_filename = "cg_batch.npy"
    spikes_filename = "cg_spikeIndices.npy"
    X_filename = "X.npy"

    # Save batch
    sorter.fetch_batch_from_file()
    sorter.preprocess_batch()
    np.save(dir / batch_filename, sorter.d_batch.get())
    print(f"Saved batch of shape {sorter.d_batch.get().shape}")

    # Save spikes
    spikes = []
    num_spikes = 8
    batch_reshaped = cp.reshape(sorter.d_batch, (sorter.W, sorter.C))
    projections = cp.array([convolve(batch_reshaped, d_template) for d_template in sorter.d_templates])
    for _ in range(num_spikes):
        best_template, best_sample = find_max_projection(projections)
        spikes.append((best_template + best_sample * sorter.T).item())
        
        for temp_diff in range(-10, 10):
            for sample_diff in range(-100, 100):
                template = best_template + temp_diff
                template = max(template, 0)
                template = min(template, sorter.T - 1)
                sample = best_sample + sample_diff
                sample = max(sample, 0)
                sample = min(sample, sorter.W)
                projections[template][sample] = 0

    np.save(dir / spikes_filename, spikes)
    print(f"Saved spikes = {spikes}")

    # Save X
    A = sorter.extract_submatrix(spikes)
    x = conjugate_gradients_gpu(A, sorter.d_batch)
    np.save(dir / X_filename, x)
    print(f"Saved X = {x}")

def isi_histograms(spike_times, spike_templates, template_map):
    prev_spike_time = { template : None for template in template_map }
    isis = { template : [] for template in template_map }
    spike_counts = { template : 0 for template in template_map }

    for spike_time, spike_template in list(zip(spike_times, spike_templates)):
        if spike_template not in template_map:
            continue

        if prev_spike_time[spike_template]:
            isi = (spike_time - prev_spike_time[spike_template]) / 30
            isis[spike_template].append(isi)
            spike_counts[spike_template] += 1
        
        prev_spike_time[spike_template] = spike_time

    sorted_spike_counts = sorted(spike_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Get the top n entries
    most_active_templates = [ template for template, _ in sorted_spike_counts[:10] ]

    for i, spike_template in enumerate(most_active_templates):
        isi = isis[spike_template]
        plt.subplot(5, 2, (i % 10) + 1)
        plt.hist(isis[spike_template], bins=list(range(0, 20, 1)))

        if i % 10 == 9:
            plt.show()

def merge_similar_templates(template_map, templates, threshold=0.95):
    """
    Merge similar templates based on cosine similarity by mapping indices of similar templates
    to the minimal index among them.

    Parameters:
    - template_map: 1D list or numpy array of template indices.
    - templates: 3D numpy array of shape (T, M, C)
    - threshold: float, cosine similarity threshold for merging templates.

    Returns:
    - A list of template indices, where similar neurons are merged to the minimal index among them.
    """
    # Ensure template_map is a numpy array for easier indexing
    template_map = np.array(template_map)
    
    # Flatten the templates to a 2D array of shape (T, M*C)
    T, M, C = templates.shape
    templates_flat = templates.reshape(T, M * C)

    # Normalize the templates to unit vectors to compute cosine similarity
    norms = np.linalg.norm(templates_flat, axis=1, keepdims=True)
    templates_normed = templates_flat / norms

    # Compute the cosine similarity matrix
    cosine_similarity_matrix = np.dot(templates_normed, templates_normed.T)

    # Convert cosine similarity to cosine distance
    cosine_distance_matrix = 1 - cosine_similarity_matrix
    np.fill_diagonal(cosine_distance_matrix, 0)  # Set diagonal to zero to ignore self-similarity

    # Use Agglomerative Clustering with precomputed distances
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity='precomputed',
        linkage='complete',
        distance_threshold=1 - threshold
    )
    labels = clustering.fit_predict(cosine_distance_matrix)

    # Map each cluster label to the minimal template index in that cluster
    cluster_to_min_index = {}
    for label in np.unique(labels):
        # Get indices of templates in the current cluster
        cluster_indices = np.where(labels == label)[0]
        # Map to the corresponding template_map indices
        mapped_indices = template_map[cluster_indices]
        # Find the minimal index among them
        min_index = np.min(mapped_indices)
        # Assign the minimal index to the cluster label
        cluster_to_min_index[label] = min_index

    # Create the new template_map by mapping each template's index to its cluster's minimal index
    new_template_map = []
    for idx in range(len(template_map)):
        label = labels[idx]
        min_index = cluster_to_min_index[label]
        new_template_map.append(min_index)

    return new_template_map


def main() -> int:
    random.seed()

    DRIVE_PATH = Path("C:/")
    BASE_PATH = Path(DRIVE_PATH, "SGL_DATA", "joplin_20240208")
    #BASE_PATH = Path(DRIVE_PATH, "SGL_DATA", "05_31")
    BIN_DIR = Path(BASE_PATH, "imec_raw")
   # KS_OUTPUT_DIR = Path(DRIVE_PATH, "Users", "Spike Sorter", "source", "repos", "OnlineSpikes_v2", "src", "kilosort4")
    KS_OUTPUT_DIR = Path(BASE_PATH, "kilosort4")
    DECODER_INPUT_DIR = Path(BASE_PATH, "decoder_input")
    CROPPED_OUTPUT_DIR = Path(BASE_PATH, "oss_input")
    BIN_FILE = "240208_g0_t0.imec0.ap.bin"
    BIN_META_FILE = "240208_g0_t0.imec0.ap.meta"
    CHANNEL_MAP_FILE = "NHP_chanmap.mat"

    config = {
        "tau": 0.11,
        "thresh": 1, # currently infinity
        "start_sample": 55935781 , # only matters for run()
        "W": 1500,
        "bin_file": BIN_DIR / BIN_FILE,
        "bin_meta_file": BIN_DIR / BIN_META_FILE,
        "oss_training_path": Path(BASE_PATH, "oss_input"),
        "ks_output_dir" : KS_OUTPUT_DIR,
        "dtype": np.float32,
        "debug_plots": False,
        "eventfile": DECODER_INPUT_DIR / "eventfile_15.txt",
        "output_dir": BASE_PATH / "cuda_output"
    }


    sorter = OSS(config)
    spikes_filename = "spikeOutput.txt"

    '''
    sorter.plot_templates(
        [   
            235,
            239
        ]
    )
    '''
    #template_map = merge_similar_templates(sorter.template_map, sorter.templates, 0.5)
    #sorter.logger.debug(f"Updated template_map = {template_map}")
    #sorter.update_config({"template_map": template_map})
    verifier = CUDAVerifier(sorter, Path(DRIVE_PATH, "SGL_DATA", "joplin_20240208") / "cuda_output")
    #verifier.verify()
   # verifier.compare_cosine_sims("cosSimilarities.txt")
    #verifier.plot_cosine_sim_statistics("cosSimilarities.txt")
    #verifier.verify_conv()
    #verifier.find_residual_threshold(spikes_filename, thresholds=[0])
    verifier.verify_single_neuron_psths(spikes_filename, 0)

    #while True:
    #    start = random.uniform(0, 4.5)
    #    verifier.compare_ks_oss_spikes_in_range(spikes_filename, 55935781 + 30000 * 60 * start, 55935781 + 30000 * 60 * start + 1500)
    #sorter.bar_graph_ks_spike_templates()
    #sorter.run()

    

    
    

    # Python ISI
    '''
    spikes = sorter.sort_spikes(4132252, 4132252 + 1500000)
    spike_times = [ spike_time for spike_time, _ in spikes ]
    spike_templates = [ spike_template for _, spike_template in spikes ]
    sorted_list = sorted(list(zip(spike_times, spike_templates)), key=lambda x: x[0])
    isi_histograms(spike_times, spike_templates, sorter.template_map)

    '''
    '''
    # CUDA ISI
    with open(BASE_PATH / "cuda_output" / "spikeOutput.txt", 'r') as f:
        spikes = [
            (int(line.split(' ')[0]), int(line.split(' ')[1]), float(line.split(' ')[2]))
            for line in f
        ]
    '''
    '''
    spike_times = [ spike_time for spike_time, _, _ in spikes ]
    spike_templates = [ spike_template for _, spike_template, _ in spikes ]
    sorted_list = sorted(list(zip(spike_times, spike_templates)), key=lambda x: x[0])
    isi_histograms(spike_times, spike_templates, sorter.template_map)
    '''

    '''
    similarity_matrix = np.zeros((len(sorter.templates), len(sorter.templates)))

    for i, template1 in enumerate(sorter.templates):
        for j, template2 in enumerate(sorter.templates):
            similarity_matrix[i, j] = cosine_similarity_flattened(template1.flatten(), template2.flatten())
    plt.hist(similarity_matrix.flatten(), bins=1000)
    plt.show()
    '''
   # heatmap(similarity_matrix)

   # sorter.run()
    #print(events[len(events) - 50])
    #print(events[len(events) - 1])
    
    #visualize_l1_matrix_no_annotations(verifier.generate_psth_l1_differences("spikeOutput.txt"))

   # generate_cuda_test_data(sorter, Path(BASE_PATH, "cuda_test_input"))

    #verifier.verify()
    #verifier.plot_A()
   # verifier.plot_ax()
    #verifier.plot_conv_result()
    #verifier.plot_residual_statistics("residualDecreases.txt")
    #verifier.verify_avg_psth("spikeOutputFixedThreshTau012.txt")
    #verifier.verify_psth("spikeOutputFixedThreshTau012.txt")
    #verifier.plot_spike_statistics("spikeStatistics.txt")
   # sorter.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
