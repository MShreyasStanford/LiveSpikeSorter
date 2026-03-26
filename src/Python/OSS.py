import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
import subprocess
#import cupy as cp
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
#from cupyx.scipy.signal import convolve2d as gpu_convolve2d
import time
from matplotlib.widgets import TextBox
from collections import defaultdict
from pprint import pprint
from torch.nn.functional import max_pool2d, avg_pool2d, conv1d, max_pool1d
from platform import python_version
from kilosort.io import BinaryFiltered, BinaryRWFile, BinaryFiltered
from kilosort.preprocessing import get_drift_matrix, fft_highpass
from kilosort.postprocessing import compute_spike_positions
from kilosort.run_kilosort import cluster_spikes, save_sorting
from torch.fft import fft, ifft, fftshift
from kilosort.template_matching import prepare_extract
import copy
from qtpy import QtCore

if torch.cuda.is_available():
    print("Using CUDA!")
else:
    print("Using CPU.")

# Code from Kilosort4
def bfile_from_ops(ops=None, ops_path=None, filename=None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cuda')

    if ops is None:
        if ops_path is not None:
            ops = load_ops(ops_path, device=device)
        else:
            raise ValueError('Must specify either `ops` or `ops_path`.')
    
    if filename is None:
        # Separate so that the same settings can be applied to the same binary
        # file at a new path, like when running pytests.
        filename = ops['filename']
    
    bfile = BinaryFiltered(
        filename, ops['n_chan_bin'], fs=ops['fs'], NT=ops['batch_size'],
        nt=ops['nt'], nt0min=ops['nt0min'], chan_map=ops['probe']['chanMap'],
        hp_filter=ops['preprocessing']['hp_filter'], whiten_mat=ops['Wrot'], dshift=ops['dshift'],
        device=device, do_CAR=ops['do_CAR'], artifact_threshold=ops['artifact_threshold'],
        invert_sign=ops['invert_sign'], dtype=ops['data_dtype'], tmin=ops['tmin'],
        tmax=ops['tmax'], shift=ops['shift'], scale=ops['scale']
        )

    return bfile

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
  #  matrix **= 3
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

def nearest_chans(ys, yc, xs, xc, nC, device=torch.device('cuda')):
    ds = (ys - yc[:,np.newaxis])**2 + (xs - xc[:,np.newaxis])**2
    iC = np.argsort(ds, 0)[:nC]
    iC = torch.from_numpy(iC).to(device)
    ds = np.sort(ds, 0)[:nC]

    return iC, ds

def prepare_matching(ops, U):
    nt = ops['nt']
    W = torch.tensor(ops['wPCA']).contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])

    #mu = (U**2).sum(-1).sum(-1)**.5
    #U2 = U / mu.unsqueeze(-1).unsqueeze(-1)

    UtU = torch.einsum('ikl, jml -> ijkm',  U, U)
    print(f"UtU shape = {UtU.shape}")
    print(f"WtW shape = {WtW.shape}")
    ctc = torch.einsum('ijkm, kml -> ijl', UtU, WtW)

    return ctc

def align_U(U, ops, device=torch.device('cuda')):
    print(f"align_U shape of U = {U.shape}")
    print(f"align_U shape of wPCA = {ops['wPCA'].shape}")
    print(f"align_U shape of wTemp = {ops['wTEMP'].shape}")
    Uex = torch.einsum('xyz, yt -> xtz', U.to(device), torch.tensor(ops['wPCA']))
    X = Uex.reshape(-1, ops['Nchan']).T
    X = conv1d(X.unsqueeze(1), torch.tensor(ops['wTEMP']).unsqueeze(1), padding=ops['nt']//2)
    Xmax = X.abs().max(0)[0].max(0)[0].reshape(-1, ops['nt'])
    imax = torch.argmax(Xmax, 1)

    Unew = Uex.clone() 
    for j in range(ops['nt']):
        ix = imax==j
        Unew[ix] = torch.roll(Unew[ix], ops['nt']//2 - j, -2)
    Unew = torch.einsum('xty, zt -> xzy', Unew, torch.tensor(ops['wPCA']))#.transpose(1,2).cpu()
    return Unew, imax

def dict_numpy_to_torch(ops):
    if isinstance(ops, np.ndarray):
        return torch.tensor(ops, device='cuda')
    elif isinstance(ops, dict):
        return {k: dict_numpy_to_torch(v) for k, v in ops.items()}
    elif isinstance(ops, list):
        return [dict_numpy_to_torch(v) for v in ops]
    elif isinstance(ops, tuple):
        return tuple(dict_numpy_to_torch(v) for v in ops)
    else:
        return ops

class OSS:
    def __init__(self, config):
        # Setup logger
        self.logger = logging.getLogger('OSS Logger')
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(ColorFormatter())

        self.logger.addHandler(ch)

        # Class members that are part of the config
        self.tau: float
        self.thresh: float
        self.start_sample: int
        self.ibatch: int 
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
        self.last_spike = max(self.spike_times)
        self.unclu_T = max(self.spike_detection_templates) + 1
        self.amplitudes = np.load(self.ks_output_dir / "amplitudes.npy")
        self.Wall3 = torch.from_numpy(np.load(self.ks_output_dir / 'Wall3.npy'))
        self.Wall3 = torch.tensor(self.Wall3, device='cuda')
        self.ctc = torch.from_numpy(np.load(self.ks_output_dir / 'ctc.npy'))
        self.ctc = torch.tensor(self.ctc, device='cuda')
        self.cluster_centroids = np.load(self.ks_output_dir / 'cluster_centroids.npy', allow_pickle=True).item()
        self.ops = np.load(self.ks_output_dir / "ops.npy", allow_pickle=True).item()
        self.ops['iU'] = torch.tensor(self.ops['iU'], device='cuda')
        self.ops['iC'] = torch.tensor(self.ops['iC'], device='cuda')
        self.ops['iCC'] = torch.tensor(self.ops['iCC'], device='cuda')
        self.ops['iKxx'] = torch.tensor(self.ops['iKxx'], device='cuda')
        self.ops['Wrot'] = torch.tensor(self.ops['Wrot'], device='cuda')
        #self.ops['wPCA'] = torch.tensor(self.ops['wPCA'], device='cuda')
        self.ops['wPCA'] = torch.tensor(np.load(Path('C:/', 'Users', 'Spike Sorter', 'source', 'repos', 'OnlineSpikes_v2', 'src', 'kilosort4') / 'wPCA.npy'), device='cuda')
        for key in self.ops['preprocessing']:
            self.logger.debug(f"Converting ops['{key}'] to Torch.tensor with device = 'cuda'")
            self.ops['preprocessing'][key] = torch.tensor(self.ops['preprocessing'][key], device='cuda')
        self.wPCA = self.ops['wPCA']
        self.n_pcs = self.ops['n_pcs']
        self.Th_learned = self.ops['Th_learned']
        self.T = self.templates.shape[0]
        self.M = self.templates.shape[1]
        self.C = self.templates.shape[2]
        self.W = self.ops['batch_size']
        self.imap = self.get_imap()
        self.logger.info(f"Loaded data from {self.oss_training_path} and {self.ks_output_dir} with parameters T = {self.templates.shape[0]} M = {self.templates.shape[1]} C = {self.templates.shape[2]}")
        self.logger.info(f"Sorting will be performed on channels: {min(self.channel_mask)} to {max(self.channel_mask)}")
        self.logger.info(f"Sorting will be performed on templates: {min(self.template_map)} to {max(self.template_map)}")
        self.logger.info(f"Kilosort sorted spikes from sample {min(self.spike_times)} to {max(self.spike_times)}")
        self.logger.debug(f"Loaded ops with keys {self.ops.keys()}")
        self.logger.debug(f"Loaded PCA with shape {self.wPCA.shape}.")
        self.logger.debug(f"Loaded Wall3 with shape {self.Wall3.shape}.")
        self.logger.debug(f"Loaded ctc with shape {self.ctc.shape}")
        self.logger.debug(f"Loaded imap with shape {self.imap.shape}")
        self.logger.debug(f"Detected batch size of {self.W}")
        self.logger.debug(f"spike_detection_templates = {self.spike_detection_templates}")
        self.logger.debug(f"spike_detection_templates.shape = {self.spike_detection_templates.shape}")
        self.logger.debug(f"spike_detection_templates max = {max(self.spike_detection_templates)}")
        self.logger.debug(f"spike_templates max = {max(self.spike_templates)}")
        self.logger.debug(f"Loaded cluster centroids = {self.cluster_centroids}")
        self.logger.debug(f"imap = {self.imap}")

        # GPU copies of Kilosort objects
        #self.d_whitening = cp.array(self.whitening)
        #self.d_templates = cp.array(self.templates)
        #self.d_wPCA = cp.array(self.wPCA)
        #self.logger.debug(f"d_templates.shape = {self.d_templates.shape}")

        # Form the matrix A
        #self.process_templates()
        #self.process_PCA()

        # Precompute exponential decay weights for cross-correlations
        self.process_exp_decay_weights()

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

        self.bfile = bfile_from_ops(ops=self.ops, device='cuda')
        self.logger.info(f"Successfully loaded bfile from ops = {self.bfile}.")

    def get_imap(self, thresh=0.8):
        map_counts = [ [0] * self.unclu_T for template in range(self.unclu_T) ]
        imap = [ None for template in range(self.unclu_T) ]
        for i in range(len(self.spike_templates)):
            map_counts[self.spike_detection_templates[i]][self.spike_templates[i]] += 1

        for template in range(self.unclu_T):
            freq = max(map_counts[template]) / sum(map_counts[template])
           #     self.logger.info(f"Template {template} in pre-image most often gets mapped to {np.argmax(map_counts[template])} with frequency {freq}.")
            if freq > thresh:
                imap[template] = np.argmax(map_counts[template])
        return np.array(imap)

    def get_events(self):
        with open(self.eventfile, 'r') as file:
            return [int(line) for line in file]

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

    def compute_closest_clusters(self, xs, ys):
        """
        Assign each spike to the closest cluster centroid based on Euclidean distance.

        Parameters:
        - xs (np.ndarray): 1D array of x positions of spikes.
        - ys (np.ndarray): 1D array of y positions of spikes.

        Returns:
        - closest_clusters (np.ndarray): 1D array of cluster indices assigned to each spike.
        """
        # Stack xs and ys to get spike_positions
        spike_positions = np.vstack((xs, ys)).T  # Shape: [num_spikes, 2]

        # Extract cluster indices and their centroid positions
        cluster_indices = np.array(list(self.cluster_centroids.keys()))
        cluster_positions = np.array(list(self.cluster_centroids.values()))  # Shape: [num_clusters, 2]

        # Compute the Euclidean distance from each spike to each cluster centroid
        # This results in a distance matrix of shape (num_spikes, num_clusters)
        diff = spike_positions[:, np.newaxis, :] - cluster_positions[np.newaxis, :, :]  # Shape: [num_spikes, num_clusters, 2]
        distances = np.linalg.norm(diff, axis=2)  # Euclidean distance along the last axis

        # Find the index of the closest cluster centroid for each spike
        min_indices = np.argmin(distances, axis=1)  # Shape: [num_spikes]

        # Map the indices back to cluster indices
        closest_clusters = cluster_indices[min_indices]  # Shape: [num_spikes]

        return closest_clusters

    def run(self):
        progress_bar = QtCore.Signal(int)
        self.output_f = open(self.output_dir / "spikeOutput.txt", 'w')
        all_spikes = []
        tiwave = torch.arange(-(self.M//2), self.M//2+1, device='cuda') 
        nC = self.ops['settings']['nearest_chans']
        iCC, iU, Ucc = prepare_extract(self.ops, self.Wall3, nC, device='cuda')
        k = 0
        st = np.zeros((10**6, 3), 'float64')
        tF  = torch.zeros((10**6, nC, self.ops['settings']['n_pcs']))
        tic0 = time.time()

        while True:
            #self.fetch_batch_from_file()
            self.kilosort_fetch_batch_from_file()
           # self.log_batch_statistics("Batch statistics prior to preprocessing")

            if self.debug_plots:
                self.plot_batch("Before preprocessing")

           # self.kilosort_preprocess_batch(int(ibatch))
           # self.log_batch_statistics("Batch statistics post preprocessing")

            if self.debug_plots:
                self.plot_batch("After preprocessing")

            stt, amps, Xres = self.kilosort_matching_pursuit()

            # Compute spike positions
            xfeat = Xres[iCC[:, iU[stt[:,1:2]]],stt[:,:1] + tiwave] @ self.ops['wPCA'].T
            xfeat += amps * Ucc[:,stt[:,1]]
            tF =  xfeat.transpose(0,1).cpu()
            xs, ys = compute_spike_positions(stt, tF, self.ops)
            closest_clusters = self.compute_closest_clusters(xs, ys)
         #   self.logger.info(f"xs head = {xs[0:10]}")
         #   self.logger.info(f"ys head = {ys[0:10]}")

            # Write spike to file
            offset = (self.ibatch - 1) * (self.ops['batch_size']) - self.M//2 + self.ops['nt0min'] - self.M

            for i, (spike_time, spike_template) in enumerate(stt.tolist()):
                if self.imap[spike_template]:
                    self.output_f.write(f"{offset + spike_time} {self.imap[spike_template]} {amps[i][0]}\n")
                else:
                    self.output_f.write(f"{offset + spike_time} {closest_clusters[i]} {amps[i][0]}\n")

            self.logger.info(f"Processed {self.W * self.ibatch} samples.")

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
            self.batch = self.batch.astype(self.dtype)
            self.ks_batch = copy.deepcopy(torch.from_numpy(self.batch))
            self.batch = self.batch.flatten(order='F')
            self.d_batch = cp.array(self.batch)
            samples_read = self.batch.size // self.C
            return samples_read

    def kilosort_fetch_batch_from_file(self):
        self.logger.debug(f"Fetching batch with ibatch = {self.ibatch}")
        self.ks_batch = self.bfile.padded_batch_to_torch(self.ibatch, self.ops)
        self.ibatch += 1

    def process_templates(self):
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

    def process_PCA(self):
        # Project to PCA space for KS implementation
        templates_reshaped = self.templates.transpose(0, 2, 1)  # Shape: (T, C, nt)

        # Flatten templates for batch processing
        templates_flat = templates_reshaped.reshape(-1, self.M).T  # Shape: (nt, T * C)

        # Compute PCA coefficients for all templates and channels
        w_flat = self.wPCA @ templates_flat  # Shape: (K, T * C)

        # Reshape w_flat to get Wall
        w_flat_T = w_flat.T  # Shape: (T * C, K)
        self.templates_PCA = torch.tensor(w_flat_T.reshape(self.T, self.n_pcs, self.C))
        self.templates_PCA, _ = align_U(self.templates_PCA, self.ops, device='cuda')
        self.logger.debug(f"Projected templates to PCA space with shape {self.templates_PCA.shape}.")

        # Check if wPCA components are orthonormal
        orthogonality_check = np.allclose(self.wPCA @ self.wPCA.T, np.eye(self.wPCA.shape[0]), atol=1e-6)
        print("PCA components are orthonormal:", orthogonality_check)

        # Compute variance in original templates
        original_variance = np.var(self.templates)
        print("Original variance:", original_variance)

        # Compute variance in PCA coefficients
        projected_variance = np.var(np.array(self.templates_PCA.flatten()))
        print("Projected variance:", projected_variance)

        # The projected variance should be less than or equal to the original variance
        variance_ratio = projected_variance / original_variance
        print("Variance ratio (projected/original):", variance_ratio)

    def process_exp_decay_weights(self):
        self.xc = self.ops['xc']
        self.yc = self.ops['yc']
        [self.ys, self.xs] = np.meshgrid(self.ops['yup'], self.ops['xup'])
        self.ys, self.xs = self.ys.flatten(), self.xs.flatten()

        self.nC = self.ops['nearest_chans']
        self.sig = self.ops['min_template_size']
        self.nsizes = self.ops['template_sizes']
        iC, ds = nearest_chans(self.ys, self.ys, self.xs, self.xs, self.nC, device='cuda')
        igood = ds[0,:] <= self.ops['max_channel_distance']**2
        iC = iC[:,igood]
        ds = ds[:,igood]
        self.ys = self.ys[igood]
        self.xs = self.xs[igood]
        ds_torch = torch.from_numpy(ds).to('cuda').float()
        template_sizes = self.sig * (1+torch.arange(self.nsizes, device='cuda'))
        weigh = torch.exp(-ds_torch.unsqueeze(-1) / template_sizes**2)
        weigh = torch.permute(weigh, (2, 0, 1)).contiguous()
        self.weigh = weigh / (weigh**2).sum(1).unsqueeze(1)**.5
        self.logger.debug(f"weigh.shape = {weigh.shape}")

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

    def kilosort_matching_pursuit(self):
        #ctc = prepare_matching(self.ops, torch.tensor(self.Wall3))
        batch = self.ks_batch
        wPCA = self.wPCA.contiguous()
       # self.logger.warning(f"ctc_max = {self.ctc.max()}")
       # self.logger.warning(f"Wall3_max = {self.Wall3.max()}")

        nm = (self.Wall3**2).sum(-1).sum(-1)
       # self.logger.warning(f"nm_max = {max(nm)}")
        B = conv1d(batch.unsqueeze(1), wPCA.unsqueeze(1), padding=self.M//2)
        B = torch.einsum('TkC, CkW -> TW', self.Wall3, B)
        trange = torch.arange(-self.M, self.M+1, device='cuda') 
        tiwave = torch.arange(-(self.M//2), self.M//2+1, device='cuda') 
        st = torch.zeros((100000,2), dtype = torch.int64, device = 'cuda')
        amps = torch.zeros((100000,1), dtype = torch.float, device = 'cuda')
        Xres = batch.clone()
        Xreconstructed = torch.zeros(batch.shape)
       # self.logger.debug(f"Xrex shape = {Xres.shape}")
       # self.logger.debug(f"Xrecon shape = {Xreconstructed.shape}")
        k = 0

        for _ in range(100):
            Cf = torch.relu(B)**2 / nm.unsqueeze(-1)
            Cf[:, :self.M] = 0
            Cf[:, -self.M:] = 0
            Cfmax, imax = torch.max(Cf, 0)
            Cmax  = max_pool1d(Cfmax.unsqueeze(0).unsqueeze(0), (2*self.M+1), stride = 1, padding = (self.M))
            cnd1 = Cmax[0,0] > self.Th_learned**2
            cnd2 = torch.abs(Cmax[0,0] - Cfmax) < 1e-9
            xs = torch.nonzero(cnd1 * cnd2)

            if len(xs)==0:
                break

            iX = xs[:,:1]
            iY = imax[iX]
            #isort = torch.sort(iX)

            nsp = len(iX)
            st[k:k+nsp, 0] = iX[:,0]
            st[k:k+nsp, 1] = iY[:,0]
            amps[k:k+nsp] = B[iY,iX] / nm[iY]
            amp = amps[k:k+nsp]

            k+= nsp

            #amp = B[iY,iX] 

            n = 2
            for j in range(n):
               # Xreconstructed[:, iX[j::n] + tiwave] += amp[j::n] * torch.einsum('ijk, jl -> kil', torch.from_numpy(self.Wall3[iY[j::n,0]]), wPCA)
                Xres[:, iX[j::n] + tiwave]  -= amp[j::n] * torch.einsum('ijk, jl -> kil', self.Wall3[iY[j::n,0]], wPCA)
                B[   :, iX[j::n] + trange]  -= amp[j::n] * self.ctc[:,iY[j::n,0],:]

            #for i, (spike_time, spike_template) in enumerate(st[k - nsp:k].tolist()):
            #    self.logger.info(f"Template {spike_template} spiked at sample {spike_time} with amplitude {amps[k - nsp + i]} and Cf value {Cf[iY, iX][i]}")

        #plt.show()
        #plt.subplot(211)
        #plt.plot(Xreconstructed.T)
        #plt.subplot(212)
        st = st[:k]
        amps = amps[:k]
        self.logger.info(f"Detected {len(st)} spikes.")
        #plt.plot(batch.T)
        #plt.show()
        #return None, spike_indices
        #return None, [ spike_time * self.T + spike_template for i, (spike_time, spike_template) in enumerate(st.tolist()) ]
        return st, amps, Xres

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
    def kilosort_preprocess_batch(self, ibatch):
        self.invert_sign = False
        self.do_CAR = True
        self.do_HP = True
        self.artifact_threshold = np.inf
        self.do_whitening = True
        self.dshift = self.ops['dshift']

        self.logger.debug(f"ibatch = {ibatch}")
        self.logger.debug(f"ks_batch.shape = {self.ks_batch.shape}")

        if self.invert_sign:
            self.ks_batch = self.ks_batch * -1

        self.ks_batch = self.ks_batch - self.ks_batch.mean(1).unsqueeze(1)

        if self.do_CAR:
            # remove the mean of each channel, and the median across channels
            self.ks_batch = self.ks_batch - torch.median(self.ks_batch, 0)[0]

        if torch.from_numpy(self.ops['preprocessing']['hp_filter']) is not None:
            fwav = fft_highpass(torch.from_numpy(self.ops['preprocessing']['hp_filter']), NT=self.W)
            self.ks_batch = torch.real(ifft(fft(self.ks_batch) * torch.conj(fwav)))
            self.ks_batch = fftshift(self.ks_batch, dim = -1)

        if self.artifact_threshold < np.inf:
            if torch.any(torch.abs(torch.tensor(self.ks_batch)) >= self.artifact_threshold):
                # Assume the batch contains a recording artifact.
                # Skip subsequent preprocessing, zero-out the batch.
                self.ks_batch = np.array(torch.zeros_like(torch.tensor(self.ks_batch)))

        # whitening, with optional drift correction
        self.logger.debug(f"Whitening shape = {self.whitening.shape}")
        self.logger.debug(f"Batch shape = {self.ks_batch.shape}")
        if self.ops['preprocessing']['whiten_mat'] is not None:
           # if self.dshift is not None and self.ops is not None and ibatch is not None:
           #     M = get_drift_matrix(self.ops, self.dshift[ibatch], device='cuda')
           #     #logger.info(M.dtype, X.dtype, self.whiten_mat.dtype)
           #     self.ks_batch = (M @ self.ops['preprocessing']['whiten_mat']) @ self.ks_batch
           # else:
                self.ks_batch = torch.from_numpy(self.ops['preprocessing']['whiten_mat']) @ self.ks_batch

    def preprocess_batch(self):
        self.mean_subtract_batch()
        self.median_subtract_batch()
        self.filter_batch()
        self.whiten_batch()
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
        self.logger.debug(f"Performing highpass filter on frequencies above {self.ops['highpass_cutoff']} hz.")
        for chan_index in range(self.C):
            chan = self.d_batch[chan_index : len(self.d_batch) : self.C]
            self.d_batch[chan_index : len(self.d_batch) : self.C] = cp.array(highpass(chan.get(), self.ops['highpass_cutoff'], self.sampling_rate_hz))

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


random.seed()

DRIVE_PATH = Path("C:/")
BASE_PATH = Path(DRIVE_PATH, "SGL_DATA", "05_31")
BIN_DIR = Path(BASE_PATH, "imec_raw")
KS_OUTPUT_DIR = Path('C:/', 'Users', 'Spike Sorter', 'source', 'repos', 'OnlineSpikes_v2', 'src', 'kilosort4')
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

config = {
    "tau": 0.11,
    "thresh": 1, # currently infinity
    "start_sample": 55935781 , # only matters for run()
   # "W": 1500,
    "ibatch": 0, # number between 0 and 200 for Kilosort
    "bin_file": BIN_DIR / BIN_FILE,
    "bin_meta_file": BIN_DIR / BIN_META_FILE,
    "oss_training_path": Path(BASE_PATH, "oss_training_full"),
    "ks_output_dir" : KS_OUTPUT_DIR,
    "dtype": np.float32,
    "debug_plots": False,
    "eventfile": DECODER_INPUT_DIR / "eventfile.txt",
    "output_dir": BASE_PATH / "python_oss_output"
}


sorter = OSS(config)
sorter.run()