import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
import subprocess
import torch
import torch.nn as nn
import math
import scipy
from enum import Enum
import statistics
from tqdm import tqdm
import os
import time
from functools import wraps
import logging
from colorama import Fore, Style, init
import time
from matplotlib.widgets import TextBox
from collections import defaultdict
from pprint import pprint
from torch.nn.functional import max_pool1d, avg_pool1d, conv1d, max_pool1d
from platform import python_version
from kilosort.io import BinaryRWFile
from kilosort.preprocessing import get_drift_matrix, fft_highpass
from kilosort.postprocessing import compute_spike_positions
from kilosort.run_kilosort import cluster_spikes, save_sorting
from torch.fft import fft, ifft, fftshift
from kilosort.template_matching import prepare_extract
import copy
from qtpy import QtCore
import warnings  # Ensure warnings is imported

# You might need to import these if not already available:
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression

if torch.cuda.is_available():
    print("Using CUDA on device cuda:1!")
else:
    print("Using CPU.")

def LOO_accuracy(spike_times, spike_templates, spike_amplitudes, 
                 event_times, event_labels, training_start, training_end, lookahead_start):
    """
    Computes the Leave-One-Out (LOO) accuracy of a logistic regression decoder trained on binned spike counts.
    
    This function assumes that spike data and event information are already loaded.
    It uses a fixed window of 100ms (100*30 samples) starting at the specified lookahead_start (in ms).
    CuPy is used for GPU-accelerated operations where possible.
    
    Parameters:
        spike_times (np.ndarray): Array of spike times.
        spike_templates (np.ndarray): Array of spike template identifiers.
        spike_amplitudes (np.ndarray): Array of spike amplitudes.
        event_times (np.ndarray): Array of event times.
        event_labels (np.ndarray): Array of event labels.
        training_start (int): Start sample for training data.
        training_end (int): End sample for training data.
        lookahead_start (int): Lookahead start (in ms) relative to the event at which a 100ms window begins.
    
    Returns:
        float: LOO accuracy in percentage.
    """
    # Convert inputs to NumPy arrays.
    spike_times = np.asarray(spike_times)
    spike_templates = np.asarray(spike_templates)
    spike_amplitudes = np.asarray(spike_amplitudes)
    event_times = np.asarray(event_times)
    event_labels = np.asarray(event_labels)
    
    # Crop spike data to the training period.
    left_index = int(np.where(spike_times > training_start)[0][0])
    right_index = int(np.where(spike_times < training_end)[0][-1])
    spike_times = spike_times[left_index: right_index]
    spike_templates = spike_templates[left_index: right_index]
    spike_amplitudes = spike_amplitudes[left_index: right_index]
    
    # Define the time range based on the training spikes.
    spike_start = spike_times[0]
    spike_end = spike_times[-1]
    
    # Crop events to fall within the spike training period.
    left_evt = int(np.where(event_times > spike_start)[0][0])
    right_evt = int(np.where(event_times < spike_end)[0][-1])
    event_times = event_times[left_evt: right_evt]
    event_labels = event_labels[left_evt: right_evt]
    
    # Binning parameters: 100ms window and 10ms step (converted to samples at 30kHz).
    bin_size = 100 * 30  # 100ms window (samples)
    bin_inc = 10 * 30    # 10ms step (samples)
    
    # Determine overall time range.
    base_time = np.minimum(event_times.min(), spike_times.min())
    max_time = np.maximum(event_times.max(), spike_times.max())
    num_bins = int(np.floor((max_time - base_time - bin_size) / bin_inc)) + 1
    
    # Get the set of unique templates.
    templates = np.unique(spike_templates)
    num_templates = int(templates.size)
    
    # Pre-compute bin boundaries.
    bin_left_edges = base_time + np.arange(num_bins) * bin_inc
    bin_right_edges = bin_left_edges + bin_size
    
    # Create a 2D array for binned spike counts.
    binned_counts = np.zeros((num_bins, num_templates), dtype=np.int32)
    for i, tmpl in enumerate(templates):
        tmpl_spikes = spike_times[spike_templates == tmpl]
        start_indices = np.searchsorted(tmpl_spikes, bin_left_edges, side='left')
        end_indices = np.searchsorted(tmpl_spikes, bin_right_edges, side='left')
        binned_counts[:, i] = end_indices - start_indices

    # Compute event bin indices.
    # The lookahead window starts at event_time + lookahead_start*30 samples.
    event_bin_indices = ((event_times + lookahead_start * 30 - base_time) // bin_inc).astype(np.int32)
    valid = event_bin_indices < num_bins
    if not bool(np.all(valid)):
        print("Warning: Some event times fall outside the binned range and will be ignored.")
        event_bin_indices = event_bin_indices[valid]
        event_labels = event_labels[valid]
    
    # Select features for the corresponding event bins.
    X = binned_counts[event_bin_indices]
    y = event_labels
    
    # Standardize features.
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    # Compute Leave-One-Out Cross-Validation Accuracy.
    loo = LeaveOneOut()
    loo_scores = cross_val_score(LogisticRegression(), X_norm, y, cv=loo)
    loo_accuracy = np.mean(loo_scores) * 100
    
    return loo_accuracy

def write_tensor(filename, A):
    with open(filename, 'w') as file:
        for x in A.flatten():
            file.write(f"{x} ")

# Modified function: added optional drift_amount parameter
def bfile_from_ops(ops=None, ops_path=None, filename=None, device=None, drift_amount=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:1')
        else:
            device = torch.device('cpu')

    if ops is None:
        if ops_path is not None:
            ops = load_ops(ops_path, device=device)
        else:
            raise ValueError('Must specify either `ops` or `ops_path`.')
    
    if filename is None:
        filename = ops['filename']
    
    # Pass drift_amount to BinaryFiltered
    bfile = BinaryFiltered(
        filename, ops['n_chan_bin'], fs=ops['fs'], NT=ops['batch_size'],
        nt=ops['nt'], nt0min=ops['nt0min'], chan_map=ops['probe']['chanMap'],
        hp_filter=ops['preprocessing']['hp_filter'], whiten_mat=ops['Wrot'], dshift=ops['dshift'],
        drift_amount=drift_amount,  # New parameter
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

# Modified class: added drift_amount as an optional parameter.
class BinaryFiltered(BinaryRWFile):
    def __init__(self, filename: str, n_chan_bin: int, fs: int = 30000, 
                 NT: int = 60000, nt: int = 61, nt0min: int = 20,
                 chan_map: np.ndarray = None, hp_filter: torch.Tensor = None,
                 whiten_mat: torch.Tensor = None, dshift: torch.Tensor = None,
                 drift_amount=None,  # New parameter to override drift
                 device: torch.device = None, do_CAR: bool = True,
                 artifact_threshold: float = np.inf, invert_sign: bool = False,
                 dtype=None, tmin: float = 0.0, tmax: float = np.inf,
                 shift=None, scale=None, file_object=None):

        super().__init__(filename, n_chan_bin, fs, NT, nt, nt0min, device,
                         dtype=dtype, tmin=tmin, tmax=tmax, shift=shift,
                         scale=scale, file_object=file_object) 
        self.chan_map = chan_map
        self.whiten_mat = whiten_mat
        self.hp_filter = hp_filter
        self.dshift = dshift
        self.drift_amount = drift_amount  # Save the override value if provided
        self.do_CAR = do_CAR
        self.invert_sign = invert_sign
        self.artifact_threshold = artifact_threshold

    def filter(self, X, ops=None, ibatch=None):
        # Pick only the channels specified in the chanMap
        if self.chan_map is not None:
            X = X[self.chan_map]

        if self.invert_sign:
            X = X * -1

        X = X - X.mean(1).unsqueeze(1)
        if self.do_CAR:
            X = X - torch.median(X, 0)[0]
    
        # High-pass filtering in the Fourier domain
        if self.hp_filter is not None:
            fwav = fft_highpass(self.hp_filter, NT=X.shape[1])
            X = torch.real(ifft(fft(X) * torch.conj(fwav)))
            X = fftshift(X, dim=-1)

        if self.artifact_threshold < np.inf:
            if torch.any(torch.abs(X) >= self.artifact_threshold):
                # Skip preprocessing if recording artifact is detected.
                return torch.zeros_like(X)

        # Whitening with optional drift correction
        if self.whiten_mat is not None:
            # If drift_amount is provided, use it instead of self.dshift[-1]
            if self.drift_amount is not None and ops is not None and ibatch is not None:
                M = get_drift_matrix(ops, self.dshift[-1] + self.drift_amount, device=self.device)
                X = (M @ self.whiten_mat) @ X
            elif self.dshift is not None and ops is not None and ibatch is not None:
                M = get_drift_matrix(ops, self.dshift[-1], device=self.device)
                X = (M @ self.whiten_mat) @ X
            else:
                X = self.whiten_mat @ X
        return X

    def __getitem__(self, *items):
        samples = super().__getitem__(*items)
        with warnings.catch_warnings():
            # Ignore warnings from torch (if any)
            warnings.filterwarnings("ignore", message="_torch_warning")
            X = torch.from_numpy(samples.T).to(self.device).float()
        return self.filter(X)
        
    def padded_batch_to_torch(self, ibatch, ops=None, return_inds=False):
        if return_inds:
            X, inds = super().padded_batch_to_torch(ibatch, return_inds=return_inds)
            return self.filter(X, ops, ibatch), inds
        else:
            X = super().padded_batch_to_torch(ibatch)
            return self.filter(X, ops, ibatch)

class TimerWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
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
    metadata = {}
    with open(filename, 'r') as bin_meta_input:
        for line in bin_meta_input:
            delimited = line.split('=') 
            if len(delimited) != 2:
                continue
            key = delimited[0]
            value = delimited[1].rstrip('\n')
            metadata[key] = value
    return metadata

def dict_numpy_to_torch(ops):
    if isinstance(ops, np.ndarray):
        return torch.tensor(ops, device='cuda:1')
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
        self.logger = logging.getLogger('LSS Logger')
        self.logger.setLevel(logging.DEBUG)
        # Only add a handler if one doesn't already exist.
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(ColorFormatter())
            self.logger.addHandler(ch)

        # Set class attributes from config
        self.ibatch: int = config["ibatch"]
        self.W : int = None  # Will be set later
        self.bin_file: str = config["bin_file"]
        self.ks_output_dir: Path = config["ks_output_dir"]
        self.bin_meta_file: str = config["bin_meta_file"]
        self.dtype: np.dtype = config["dtype"]
        self.output_dir: Path = config["output_dir"]
        self.sample_offset: int = config["sample_offset"]
        self.cuda_debug_dir: Path = config["cuda_debug_dir"]
        self.drift_amount = config.get("drift_amount", None)  # New drift_amount parameter

        # Load data from Kilosort
        self.spike_templates = np.load(self.ks_output_dir / "spike_templates.npy")
        self.spike_detection_templates = np.load(self.ks_output_dir / "spike_detection_templates.npy")
        self.unclu_T = max(self.spike_detection_templates) + 1
        self.Wall3 = torch.tensor(np.load(self.ks_output_dir / 'Wall3.npy'), device='cuda:1')
        self.ctc = torch.tensor(np.load(self.ks_output_dir / 'ctc.npy'), device='cuda:1')
        self.cluster_centroids = np.load(self.ks_output_dir / 'cluster_centroids.npy', allow_pickle=True).item()
        self.ops = np.load(self.ks_output_dir / "ops.npy", allow_pickle=True).item()
        self.ops['iU'] = torch.tensor(self.ops['iU'], device='cuda:1')
        self.ops['iC'] = torch.tensor(self.ops['iC'], device='cuda:1')
        self.ops['iCC'] = torch.tensor(self.ops['iCC'], device='cuda:1')
        self.ops['Wrot'] = torch.tensor(self.ops['Wrot'], device='cuda:1')
        self.ops['iKxx'] = torch.tensor(self.ops['iKxx'], device='cuda:1')
        self.ops['wPCA'] = torch.tensor(np.load(self.ks_output_dir / 'wPCA.npy'), device='cuda:1')

        for key in self.ops['preprocessing']:
            self.ops['preprocessing'][key] = torch.tensor(self.ops['preprocessing'][key], device='cuda:1')

        self.wPCA = self.ops['wPCA']
        self.n_pcs = self.ops['n_pcs']
        self.Th_learned = self.ops['Th_learned']
        self.M = self.ops['nt']
        self.W = self.ops['batch_size']
      #  self.imap = self.get_imap()

        # Parse metadata file for pre-cropped parameters
        metadata = parse_bin_meta_file(self.bin_meta_file)
        self.raw_C = int(metadata["nSavedChans"])   
        self.sampling_rate_hz = int(float(metadata["imSampRate"]))
        x_values = [ -51.5, 35.5, -35.5, 51.5 ]
        self.channel_positions = [ ( x_values[i % 4], (i // 2) * 20 ) for i in range(self.raw_C) ]
        self.logger.info(f"Reading data from {self.ks_output_dir}")
        self.logger.info(f"Detected {self.raw_C} channels at a sampling rate of {self.sampling_rate_hz} hz from the binary file")
        self.logger.info(f"Detected channel positions from {self.channel_positions[0:5]} to {self.channel_positions[-5:]}")
        self.logger.info(f"Detected {self.Wall3.shape[0]} number of templates.")
        self.logger.info(f"Detected {self.n_pcs} principal components.")
        self.logger.info(f"Using a threshold of {self.Th_learned}, batch size of {self.W}, and template size of {self.M} for matching pursuit.")

        # Additional class members that will be populated later
        self.batch: list = None
        self.d_batch = None

        # Create bfile and pass the drift_amount (if provided) along
        self.bfile = bfile_from_ops(ops=self.ops, device=torch.device('cuda:1'), filename=self.bin_file, drift_amount=self.drift_amount)

    def get_imap(self, thresh=0.8):
        map_counts = [ [0] * self.unclu_T for template in range(self.unclu_T) ]
        imap = [ None for template in range(self.unclu_T) ]
        for i in range(len(self.spike_templates)):
            map_counts[self.spike_detection_templates[i]][self.spike_templates[i]] += 1

        for template in range(self.unclu_T):
            freq = max(map_counts[template]) / sum(map_counts[template])
            if freq > thresh:
                imap[template] = np.argmax(map_counts[template])
        return np.array(imap)

    def update_config(self, config):
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.logger.warning(f"{key} is not a valid attribute of {self.__class__.__name__}")

    def compute_closest_clusters(self, xs, ys):
        """
        Assign each spike to the closest cluster centroid based on Euclidean distance.
        """
        spike_positions = np.vstack((xs, ys)).T
        cluster_indices = np.array(list(self.cluster_centroids.keys()))
        cluster_positions = np.array(list(self.cluster_centroids.values()))
        diff = spike_positions[:, np.newaxis, :] - cluster_positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        min_indices = np.argmin(distances, axis=1)
        closest_clusters = cluster_indices[min_indices]
        return closest_clusters

    def run(self):
        self.output_f = open(self.output_dir / "spikeOutput.txt", 'w')
        all_spikes = []
        tiwave = torch.arange(-(self.M//2), self.M//2+1, device=torch.device('cuda:1')) 
        nC = self.ops['settings']['nearest_chans']
        iCC, iU, Ucc = prepare_extract(self.ops, self.Wall3, nC, device=torch.device('cuda:1'))
        k = 0
        st = np.zeros((10**6, 3), 'float64')
        tF  = torch.zeros((10**6, nC, self.ops['settings']['n_pcs']))
        tic0 = time.time()
        t = 0

        # Main loop wrapped in try/except to exit gracefully when no more data is available
        while True:
            try:
                self.kilosort_fetch_batch_from_file()
            except Exception as e:
                self.logger.info("No more data to fetch or encountered an error while fetching batch.")
                break

            try:
                stt, amps, Xres = self.kilosort_matching_pursuit()
            except Exception as e:
                self.logger.info("No more data to process or encountered an error during matching pursuit.")
                break

            # Compute spike positions
            xfeat = Xres[iCC[:, iU[stt[:,1:2]]], stt[:,:1] + tiwave] @ self.ops['wPCA'].T
            xfeat += amps * Ucc[:, stt[:,1]]
            tF = xfeat.transpose(0,1).cpu()
            xs, ys = compute_spike_positions(stt, tF, self.ops)
            closest_clusters = self.compute_closest_clusters(xs, ys)

            # Write spike to file
            offset = (self.sample_offset + (self.ibatch - 1) * (self.ops['batch_size']) -
                      self.M//2 + self.ops['nt0min'] - self.M)

            for i, (spike_time, spike_template) in enumerate(stt.tolist()):
                self.output_f.write(f"{offset + spike_time},{closest_clusters[i]},{amps[i][0]}\n")
            t += 1
            if t % 10 == 0:
                self.logger.info(f"Processed {self.W * self.ibatch} samples.")

        self.output_f.close()
        self.logger.info("Run complete. Exiting gracefully.")

    def kilosort_fetch_batch_from_file(self):
        self.ks_batch = self.bfile.padded_batch_to_torch(self.ibatch, self.ops)
        self.ibatch += 1

    def kilosort_matching_pursuit(self):
        start = time.time()
        batch = self.ks_batch
        wPCA = self.wPCA.contiguous()
        nm = (self.Wall3**2).sum(-1).sum(-1)
        B = conv1d(batch.unsqueeze(1), wPCA.unsqueeze(1), padding=self.M//2)
        B = torch.einsum('TkC, CkW -> TW', self.Wall3, B)
        trange = torch.arange(-self.M, self.M+1, device=torch.device('cuda:1'))
        tiwave = torch.arange(-(self.M//2), self.M//2+1, device=torch.device('cuda:1')) 
        st = torch.zeros((100000,2), dtype=torch.int64, device=torch.device('cuda:1'))
        amps = torch.zeros((100000,1), dtype=torch.float, device=torch.device('cuda:1'))
        Xres = batch.clone()
        Xreconstructed = torch.zeros(batch.shape)
        k = 0

        for _ in range(100):
            Cf = torch.relu(B)**2 / nm.unsqueeze(-1)
            Cf[:, :self.M] = 0
            Cf[:, -self.M:] = 0
            Cfmax, imax = torch.max(Cf, 0)
            Cmax = max_pool1d(Cfmax.unsqueeze(0).unsqueeze(0), (2*self.M+1), stride=1, padding=(self.M))
            cnd1 = Cmax[0,0] > self.Th_learned**2
            cnd2 = torch.abs(Cmax[0,0] - Cfmax) < 1e-9
            xs = torch.nonzero(cnd1 * cnd2)
            if len(xs) == 0:
                break
            iX = xs[:, :1]
            iY = imax[iX]
            nsp = len(iX)
            st[k:k+nsp, 0] = iX[:, 0]
            st[k:k+nsp, 1] = iY[:, 0]
            amps[k:k+nsp] = B[iY, iX] / nm[iY]
            amp = amps[k:k+nsp]
            k += nsp
            n = 2
            for j in range(n):
                Xres[:, iX[j::n] + tiwave] -= amp[j::n] * torch.einsum('ijk, jl -> kil', self.Wall3[iY[j::n, 0]], wPCA)
                B[:, iX[j::n] + trange] -= amp[j::n] * self.ctc[:, iY[j::n, 0], :]

        st = st[:k]
        amps = amps[:k]
        end = time.time()

        return st, amps, Xres

# New function to run a drift test.
def drift_test(drift_amount):
    # Assume config is defined below; inject the drift_amount into the config
    config["drift_amount"] = drift_amount
    sorter = OSS(config)
    sorter.run()

def run_LOO(
    base_dir, 
    decoder_subdir="decoder_input", 
    ks_subdir="kilosort4", 
    oss_subdir="cuda_output",
    eventfile_names=["eventfile_15.txt", "eventfile_26.txt", "eventfile_37.txt", "eventfile_48.txt"],
    training_start=26984448, 
    training_end=83984384,
    lookahead_start=0
):
    """
    Computes LOO accuracy for KS and OSS spike data for a single lookahead start value,
    averaged over multiple eventfiles.
    
    Parameters:
      base_dir (Path or str): The base directory containing all subdirectories.
      decoder_subdir (str): Subdirectory for decoder input (event files).
      ks_subdir (str): Subdirectory for Kilosort spike data.
      oss_subdir (str): Subdirectory for OSS spike data.
      eventfile_names (list of str): List of event file names.
      training_start (int): Starting sample index for training.
      training_end (int): Ending sample index for training.
      lookahead_start (int): The lookahead start time in ms.
    
    Returns:
      avg_ks (float): Averaged LOO accuracy for KS spikes.
      avg_oss (float): Averaged LOO accuracy for OSS spikes.
    """
    # Set up directories.
    base_dir = Path(base_dir)
    decoder_dir = base_dir / decoder_subdir
    ks_dir = base_dir / ks_subdir
    oss_dir = base_dir / oss_subdir

    print(f"Computing LOO accuracy using:")
    print(f"  KS spikes: {decoder_dir / 'ksSpikeOutput.txt'}")
    print(f"  LSS spikes: {oss_dir / 'spikeOutput.txt'}")
    print(f"  Eventfiles: {eventfile_names}")
    print(f"  Training samples: {training_start} to {training_end}")
    print(f"  Lookahead start: {lookahead_start} ms")
    print("Loading spike data...")

    # Load Kilosort spikes.
    ks_spike_times = []
    ks_spike_templates = []
    ks_spike_amplitudes = []
    with open(decoder_dir / 'ksSpikeOutput.txt', 'r') as f:
        for line in f:
            tokens = line.split(',')
            ks_spike_times.append(int(tokens[0]))
            ks_spike_templates.append(int(tokens[1]))
            ks_spike_amplitudes.append(float(tokens[2]))
    ks_spike_times = np.array(ks_spike_times)
    ks_spike_templates = np.array(ks_spike_templates)
    ks_spike_amplitudes = np.array(ks_spike_amplitudes)

    # Sort the Kilosort spikes by time.
    sort_idx = np.argsort(ks_spike_times)
    ks_spike_times = ks_spike_times[sort_idx]
    ks_spike_templates = ks_spike_templates[sort_idx]
    ks_spike_amplitudes = ks_spike_amplitudes[sort_idx]

    # Load OSS spikes.
    oss_spike_times = []
    oss_spike_templates = []
    oss_spike_amplitudes = []
    with open(oss_dir / 'spikeOutput.txt', 'r') as f:
        for line in f:
            tokens = line.split(',')
            oss_spike_times.append(int(tokens[0]))
            oss_spike_templates.append(int(tokens[1]))
            oss_spike_amplitudes.append(float(tokens[2]))
    oss_spike_times = np.array(oss_spike_times)
    oss_spike_templates = np.array(oss_spike_templates)
    oss_spike_amplitudes = np.array(oss_spike_amplitudes)

    # Sort the OSS spikes by time.
    sort_idx = np.argsort(oss_spike_times)
    oss_spike_times = oss_spike_times[sort_idx]
    oss_spike_templates = oss_spike_templates[sort_idx]
    oss_spike_amplitudes = oss_spike_amplitudes[sort_idx]

    print("Spike data loaded successfully.")

    # Local helper function to load event data.
    def load_event_data(event_file_path):
        event_times = []
        event_labels = []
        with open(event_file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                event_times.append(float(tokens[0]))
                event_labels.append(int(tokens[1]))
        return np.array(event_times), np.array(event_labels)

    # Create list of event file paths.
    eventfile_paths = [decoder_dir / name for name in eventfile_names]
    
    ks_acc_list = []
    oss_acc_list = []
    
    # Loop over each event file.
    for event_file in eventfile_paths:
        event_times, event_labels = load_event_data(event_file)
        
        # LOO_accuracy is assumed to be defined elsewhere.
        acc_ks = LOO_accuracy(
            ks_spike_times, ks_spike_templates, ks_spike_amplitudes,
            event_times, event_labels, training_start, training_end, lookahead_start
        )
        acc_oss = LOO_accuracy(
            oss_spike_times, oss_spike_templates, oss_spike_amplitudes,
            event_times, event_labels, training_start, training_end, lookahead_start
        )
        ks_acc_list.append(acc_ks)
        oss_acc_list.append(acc_oss)
    
    avg_ks = np.mean(ks_acc_list)
    avg_oss = np.mean(oss_acc_list)
    
    return avg_ks, avg_oss

# Define paths and config (update paths as needed)
DRIVE_PATH = Path("C:/")
BASE_PATH = Path(DRIVE_PATH, "SGL_DATA", "joplin_20240208")
BIN_DIR = Path(BASE_PATH, "imec_raw")
KS_OUTPUT_DIR = Path(BASE_PATH, "kilosort4_train")
BIN_FILE = "240208_g0_t0_test.imec0.ap.bin"
BIN_META_FILE = "240208_g0_t0_test.imec0.ap.meta"
CHANNEL_MAP_FILE = "neuropixels_NHP_channel_map_dev_staggered_v1"

config = {
    "ibatch": 0, 
    "bin_file": BIN_DIR / BIN_FILE,
    "bin_meta_file": BIN_DIR / BIN_META_FILE,
    "ks_output_dir": KS_OUTPUT_DIR,
    "dtype": np.float32,
    "output_dir": BASE_PATH / "python_oss_output",
    "sample_offset": 26984448 - 256 * 64,
    "cuda_debug_dir": BASE_PATH / "cuda_test_input"
    # "drift_amount" can be added here or passed via drift_test()
}


if len(sys.argv < 2):
    print("Error: Expected 1 argument drift_test.")
    exit()

drift_amount = int(sys.argv[1])

result_file_path = BASE_PATH / f"drift_test_{drift_amount}.txt"
results = []

with open(result_file_path, 'w') as result_file:
    print(f"\nRunning drift_test with drift_amount = {drift_amount}")
    drift_test(drift_amount)  # This call updates the OSS spike file using the provided drift_amount.
    
    # Now compute LOO for the given lookahead start (e.g., 0 ms).
    oss_loo = run_LOO(
        base_dir=BASE_PATH,
        decoder_subdir="decoder_input",
        ks_subdir="kilosort4",
        oss_subdir="python_oss_output",
        eventfile_names=["eventfile_15.txt", "eventfile_26.txt", "eventfile_37.txt", "eventfile_48.txt"],
        training_start=26984448,
        training_end=83984384,
        lookahead_start=100  # Change this value if needed.
    )
    
    print(f"Drift: {drift_amount}, LSS LOO: {oss_loo:.4f}")
    result_file.write(f"{drift_amount}, {oss_loo:.4f}\n")
