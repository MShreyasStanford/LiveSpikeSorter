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
from torch.nn.functional import max_pool2d, avg_pool2d, conv1d, max_pool1d
from platform import python_version
from kilosort.io import BinaryRWFile
from kilosort.preprocessing import get_drift_matrix, fft_highpass
from kilosort.postprocessing import compute_spike_positions
from kilosort.run_kilosort import cluster_spikes, save_sorting
from torch.fft import fft, ifft, fftshift
from kilosort.template_matching import prepare_extract
from kilosort.clustering_qr import assign_iclust0
import copy
from qtpy import QtCore

if torch.cuda.is_available():
    print("Using CUDA!")
else:
    print("Using CPU.")

def write_tensor(filename, A):
    with open(filename, 'w') as file:
        for x in A.flatten():
            file.write(f"{x} ")

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

class BinaryFiltered(BinaryRWFile):
    def __init__(self, filename: str, n_chan_bin: int, fs: int = 30000, 
                 NT: int = 60000, nt: int = 61, nt0min: int = 20,
                 chan_map: np.ndarray = None, hp_filter: torch.Tensor = None,
                 whiten_mat: torch.Tensor = None, dshift: torch.Tensor = None,
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
        self.do_CAR = do_CAR
        self.invert_sign=invert_sign
        self.artifact_threshold = artifact_threshold

    def filter(self, X, ops=None, ibatch=None):
        # pick only the channels specified in the chanMap
        if self.chan_map is not None:
            X = X[self.chan_map]

        if self.invert_sign:
            X = X * -1

        X = X - X.mean(1).unsqueeze(1)
        if self.do_CAR:
            # remove the mean of each channel, and the median across channels
            X = X - torch.median(X, 0)[0]
    
        # high-pass filtering in the Fourier domain (much faster than filtfilt etc)
        if self.hp_filter is not None:
            fwav = fft_highpass(self.hp_filter, NT=X.shape[1])
            X = torch.real(ifft(fft(X) * torch.conj(fwav)))
            X = fftshift(X, dim = -1)

        if self.artifact_threshold < np.inf:
            if torch.any(torch.abs(X) >= self.artifact_threshold):
                # Assume the batch contains a recording artifact.
                # Skip subsequent preprocessing, zero-out the batch.
                return torch.zeros_like(X)

        # whitening, with optional drift correction
        if self.whiten_mat is not None:
            if self.dshift is not None and ops is not None and ibatch is not None:
                M = get_drift_matrix(ops, self.dshift[-1], device=self.device)
                X = (M @ self.whiten_mat) @ X
            else:
                X = self.whiten_mat @ X
        return X

    def __getitem__(self, *items):
        samples = super().__getitem__(*items)
        with warnings.catch_warnings():
            # Don't need this, we know about the warning and it doesn't cause
            # any problems. Doing this the "correct" way is much slower.
            warnings.filterwarnings("ignore", message=_torch_warning)
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
        for line in bin_meta_input:
            delimited = line.split('=')

            if len(delimited) != 2:
                continue

            key = delimited[0]
            value = delimited[1]
            value = value.rstrip('\n')
            metadata[key] = value

    return metadata

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

def cluster_centroids_pca_compute(templates, Wall, pc_feature_ind):
    T, nt, C = templates.shape
    T2, C2, P = Wall.shape
    _, nC = pc_feature_ind.shape
    assert T == T2 and C == C2, "templates and Wall must agree on T and C"
    
    # Allocate output
    centroids = np.zeros((T, nC * P), dtype=Wall.dtype)
    
    # Loop over templates
    for t in range(T):
        idx = pc_feature_ind[t]    # shape (K,)
        
        # select the K×P block of PC‐features
        W_sub = Wall[t, idx, :]    # shape (K, P)
        
        # simple average across the K channels
        centroids[t, :] = W_sub.flatten()

    return centroids


def assign_to_clusters(
    tF,
    centroids_pca,
    spike_xs,
    spike_ys,
    cluster_xy,
    R,
    device='cuda'
):
    # 1) collapse channel-dimension
    tF = np.asarray(tF, dtype=np.float32)            # (N, nC, D)
   # X = tF.mean(axis=1)                               # (N, D)
    X = tF.reshape(tF.shape[0], -1) 

    # 2) move features and centroids to torch
    dev = torch.device(device)
    Xg = torch.from_numpy(X).to(dev)                  # (N, D)
    mu = torch.from_numpy(
        np.asarray(centroids_pca, dtype=np.float32)
    ).to(dev)                                         # (K, D)

    # 3) compute the “inner-product” scores
    vv = Xg @ mu.T                                    # (N, K)
    nm = (mu**2).sum(1)                               # (K,)
    scores = 2*vv - nm                                # (N, K)

    # 4) build cx, cy tensors from cluster_xy dict
    K = mu.shape[0]
    cx = torch.full((K,), float('nan'), device=dev)
    cy = torch.full((K,), float('nan'), device=dev)
    for idx, (x, y) in cluster_xy.items():
        cx[idx] = float(x)   # ensure a Python float is assigned
        cy[idx] = float(y)

    # 5) compute mask of clusters within R of each spike
    spike_xs = torch.tensor(spike_xs, dtype=torch.float32, device=dev)  # (N,)
    spike_ys = torch.tensor(spike_ys, dtype=torch.float32, device=dev)  # (N,)
    dx2 = (spike_xs.unsqueeze(1) - cx.unsqueeze(0))**2  # (N, K)
    dy2 = (spike_ys.unsqueeze(1) - cy.unsqueeze(0))**2  # (N, K)
    mask = (dx2 + dy2) <= (R**2)                        # (N, K)

    # 6) disallow out-of-range clusters
    scores = scores.masked_fill(~mask, float('-inf'))

    # 7) pick the best-scoring cluster for each spike
    best = torch.argmax(scores, dim=1)                  # (N,)

    # 8) any spike with no True in mask[i] gets -1
    no_candidate = ~mask.any(dim=1)                     # (N,)
    best[no_candidate] = -1

    return best.cpu().numpy().astype(int)

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
        self.ibatch: int 
        self.W : int
        self.bin_file: str
        self.ks_output_dir: Path
        self.bin_meta_file: str
        self.dtype: np.dtype
        self.output_dir: str
        self.sample_offset: int
        self.cuda_debug_dir: str

        for key, value in config.items():
            setattr(self, key, value)

        # Load data from Kilosort
        self.spike_templates = np.load(self.ks_output_dir / "spike_templates.npy")
        self.spike_detection_templates = np.load(self.ks_output_dir / "spike_detection_templates.npy")
        self.unclu_T = max(self.spike_detection_templates) + 1
        self.Wall3 = torch.tensor(torch.from_numpy(np.load(self.ks_output_dir / 'Wall3.npy')), device='cuda')
        self.ctc = torch.tensor(torch.from_numpy(np.load(self.ks_output_dir / 'ctc.npy')), device='cuda')
        self.cluster_centroids = np.load(self.ks_output_dir / 'cluster_centroids.npy', allow_pickle=True).item()
        self.ops = np.load(self.ks_output_dir / "ops.npy", allow_pickle=True).item()
        self.ops['iU'] = torch.tensor(self.ops['iU'], device='cuda')
        self.ops['iC'] = torch.tensor(self.ops['iC'], device='cuda')
        self.ops['iCC'] = torch.tensor(self.ops['iCC'], device='cuda')
        self.ops['Wrot'] = torch.tensor(self.ops['Wrot'], device='cuda')
        self.ops['iKxx'] = torch.tensor(self.ops['iKxx'], device='cuda')
        self.ops['wPCA'] = torch.tensor(np.load(self.ks_output_dir / 'wPCA.npy'), device='cuda')
        pc_feature_ind = np.load(self.ks_output_dir / 'pc_feature_ind.npy')
        Wall = np.load(self.ks_output_dir / 'Wall.npy')
        templates = np.load(self.ks_output_dir / 'templates.npy')
        self.cluster_centroids_pca = cluster_centroids_pca_compute(templates, Wall, pc_feature_ind)

        for key in self.ops['preprocessing']:
            self.ops['preprocessing'][key] = torch.tensor(self.ops['preprocessing'][key], device='cuda')

        self.wPCA = self.ops['wPCA']
        self.n_pcs = self.ops['n_pcs']
        self.Th_learned = self.ops['Th_learned']
        self.M = self.ops['nt']
        self.ops['batch_size'] = 60000
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
        self.logger.info(f"Using a threshold of {self.Th_learned}, batch size of {self.W}, and templaet size of {self.M} for matching pursuit.")

        # To chunk batches of size 60000 to 1500
        self._fetch_buffer = None
        self._fetch_buffer_pos = 0
        self._subbatch_size = 1500

        # Additional class members that will be populated later
        self.batch: list
        self.d_batch: cp.array

        self.bfile = bfile_from_ops(ops=self.ops, device='cuda', filename=self.bin_file)
        self.lookback = self.M * 2

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
        self.output_f = open(self.output_dir / "spikeOutput.txt", 'w')
        all_spikes = []
        tiwave = torch.arange(-(self.M//2), self.M//2+1, device='cuda') 
        nC = self.ops['settings']['nearest_chans']
        iCC, iU, Ucc = prepare_extract(self.ops, self.Wall3, nC, device='cuda')
        k = 0
        st = np.zeros((10**6, 3), 'float64')
        tF  = torch.zeros((10**6, nC, self.ops['settings']['n_pcs']))
        tic0 = time.time()

        t = 0
        while True:
            self.kilosort_fetch_batch_from_file()
            stt, amps, Xres = self.kilosort_matching_pursuit()

            # Compute spike positions
            xfeat = Xres[iCC[:, iU[stt[:,1:2]]],stt[:,:1] + tiwave] @ self.ops['wPCA'].T
            xfeat += amps * Ucc[:,stt[:,1]]
            tF =  xfeat.transpose(0,1).cpu()
            xs, ys = compute_spike_positions(stt, tF, self.ops)
           # closest_clusters = self.compute_closest_clusters(xs, ys)
            
            
            closest_clusters = assign_to_clusters(
                tF,
                self.cluster_centroids_pca,
                xs, ys,
                self.cluster_centroids,
                80, # distance threshold in microns
                device='cuda'
            )
            

            # Write spike to file
            #offset = self.sample_offset + (self.ibatch - 1) * (self.ops['batch_size']) - self.M//2 + self.ops['nt0min'] - self.M
            offset = self.sample_offset + self._fetch_buffer_pos - self._subbatch_size   + (self.ibatch - 1) * (self.ops['batch_size']) - self.M//2 + self.ops['nt0min'] - self.M

            for i, (spike_time, spike_template) in enumerate(stt.tolist()):
                #if self.imap[spike_template]:
                #    self.output_f.write(f"{offset + spike_time} {self.imap[spike_template]} {amps[i][0]}\n")
                #else:
              #  print(f"Spike template {spike_template} has been assigned to {closest_clusters[i]}.")
                self.output_f.write(f"{offset + spike_time},{closest_clusters[i]},{amps[i][0]}\n")

            t += 1
            if t % 10 == 0:
                self.logger.info(f"Processed {self.W * self.ibatch + self._fetch_buffer_pos} samples.")

    def kilosort_fetch_batch_from_file_chunked(self):
        # if buffer is empty or too small for another subbatch, refill it
        if (self._fetch_buffer is None
            or self._fetch_buffer_pos + self._subbatch_size
                > self._fetch_buffer.shape[-1]):
            # grab one big batch of 60 000 samples
            self._fetch_buffer = self.bfile.padded_batch_to_torch(
                self.ibatch, self.ops
            )
            self.ibatch += 1
            self._fetch_buffer_pos = 0

        # slice out the next 1500-sample chunk
        start = self._fetch_buffer_pos
        end   = start + self._subbatch_size
        # assume padded_batch_to_torch returns [n_channels, n_samples]
        self.ks_batch = self._fetch_buffer[:, start:end]
        self._fetch_buffer_pos = end

    def kilosort_fetch_batch_from_file_chunked_lookback(self):
        """
        Produce a subbatch of shape [n_channels, lookback + subbatch_size].
        Each subbatch overlaps the previous one by exactly `lookback` samples.
        Assumes padded_batch_to_torch returns [n_channels, n_samples].
        """
        import torch

        lb = int(getattr(self, "lookback", 0))
        need_core = int(self._subbatch_size)

        # Refill big buffer if we don't have enough room for the next *core* subbatch.
        # (Left overlap is handled via prev tail / within-buffer slice.)
        if (self._fetch_buffer is None
            or self._fetch_buffer_pos + need_core > self._fetch_buffer.shape[-1]):

            # Stash tail from previous buffer for cross-buffer left overlap
            if getattr(self, "_fetch_buffer", None) is not None and lb > 0:
                tail_len = min(lb, self._fetch_buffer.shape[-1])
                # clone so this survives after we overwrite _fetch_buffer
                self._prev_tail = self._fetch_buffer[:, -tail_len:].clone()
            else:
                self._prev_tail = None

            # Grab one big batch (e.g., 60_000 samples)
            self._fetch_buffer = self.bfile.padded_batch_to_torch(self.ibatch, self.ops)
            self.ibatch += 1
            self._fetch_buffer_pos = 0

        # Core slice indices inside the current big buffer
        start = self._fetch_buffer_pos
        end   = start + need_core

        # Safety check: we guaranteed this by the refill logic above
        # but assert anyway to surface logic bugs early.
        assert end <= self._fetch_buffer.shape[-1], "Internal: core slice overruns big buffer"

        if lb <= 0:
            # No overlap requested
            self.ks_batch = self._fetch_buffer[:, start:end]
        else:
            # Build the left-overlap part of exact length `lb`
            left_start_inbuf = max(0, start - lb)
            left_inbuf = self._fetch_buffer[:, left_start_inbuf:start]  # may be shorter than lb
            have = left_inbuf.shape[-1]
            need = lb - have

            if need > 0:
                if getattr(self, "_prev_tail", None) is not None and self._prev_tail.numel() > 0:
                    pad = self._prev_tail[:, -need:] if self._prev_tail.shape[-1] >= need else \
                          torch.cat(
                              [self._prev_tail,
                               torch.zeros(self._fetch_buffer.shape[0], need - self._prev_tail.shape[-1],
                                           device=self._fetch_buffer.device, dtype=self._fetch_buffer.dtype)],
                              dim=-1
                          )
                else:
                    # First ever subbatch: left-pad with zeros
                    pad = torch.zeros(self._fetch_buffer.shape[0], need,
                                      device=self._fetch_buffer.device, dtype=self._fetch_buffer.dtype)
                left_part = torch.cat([pad, left_inbuf], dim=-1)  # exactly lb
            else:
                # We have at least lb samples inside this buffer
                left_part = left_inbuf[:, -lb:]  # trim to exactly lb

            core = self._fetch_buffer[:, start:end]
            self.ks_batch = torch.cat([left_part, core], dim=-1)

        # Advance cursor by the *core* size (overlap is only to the left)
        self._fetch_buffer_pos = end

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
        trange = torch.arange(-self.M, self.M+1, device='cuda')
        tiwave = torch.arange(-(self.M//2), self.M//2+1, device='cuda') 
        st = torch.zeros((100000,2), dtype = torch.int64, device = 'cuda')
        amps = torch.zeros((100000,1), dtype = torch.float, device = 'cuda')
        Xres = batch.clone()
        Xreconstructed = torch.zeros(batch.shape)
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

            nsp = len(iX)
            st[k:k+nsp, 0] = iX[:,0]
            st[k:k+nsp, 1] = iY[:,0]
            amps[k:k+nsp] = B[iY,iX] / nm[iY]
            amp = amps[k:k+nsp]

            k += nsp
            n = 2
            for j in range(n):
                Xres[:, iX[j::n] + tiwave]  -= amp[j::n] * torch.einsum('ijk, jl -> kil', self.Wall3[iY[j::n,0]], wPCA)
                B[   :, iX[j::n] + trange]  -= amp[j::n] * self.ctc[:,iY[j::n,0],:]
            print(".", end='',flush=True)
        st = st[:k]
        amps = amps[:k]
        end = time.time()
       # self.logger.info(f"Detected {len(st)} spikes in batch of shape {batch.shape}")

        return st, amps, Xres

DRIVE_PATH = Path("C:/")
BASE_PATH = Path(DRIVE_PATH, "SGL_DATA", "joplin_20240222")
BIN_DIR = Path(BASE_PATH, "imec_raw")
KS_OUTPUT_DIR = Path(BASE_PATH, "kilosort4_train")
BIN_FILE = "240222_g0_t0.imec0.ap.bin"
BIN_META_FILE = "240222_g0_t0.imec0.ap.meta"
CHANNEL_MAP_FILE = "neuropixels_NHP_channel_map_dev_staggered_v1"

config = {
    "ibatch": 0, 
    "bin_file": BIN_DIR / BIN_FILE,
    "bin_meta_file": BIN_DIR / BIN_META_FILE,
    "ks_output_dir" : KS_OUTPUT_DIR,
    "dtype": np.float32,
    "output_dir": BASE_PATH / "python_oss_output",
    "sample_offset": 0,
    "cuda_debug_dir": BASE_PATH / "cuda_test_input"
}

sorter = OSS(config)
sorter.run()