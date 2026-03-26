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
import copy
from qtpy import QtCore

torch.set_printoptions(threshold=10_000)
if torch.cuda.is_available():
    print("Using CUDA!")
else:
    print("Using CPU.")

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

    def __getitem__(self, *items):
        samples = super().__getitem__(*items)
        with warnings.catch_warnings():
            # Don't need this, we know about the warning and it doesn't cause
            # any problems. Doing this the "correct" way is much slower.
            warnings.filterwarnings("ignore", message=_torch_warning)
            X = torch.from_numpy(samples.T).to(self.device).float()
        return self.filter(X)
        
    def padded_batch_to_torch(self, ibatch, ops=None, return_inds=False):
    	return super().padded_batch_to_torch(ibatch)

def write_tensor(filename, A):
    """
    Write tensor A to a file with the given filename.
    Each file is written with no extension and contains the flattened float values.
    """
    with open(filename, 'w') as file:
        for x in A.flatten():
            file.write(f"{float(x)} ")

def compute_closest_clusters(centroids, xs, ys):
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
        cluster_indices = np.array(list(centroids.keys()))
        cluster_positions = np.array(list(centroids.values()))  # Shape: [num_clusters, 2]

        # Compute the Euclidean distance from each spike to each cluster centroid
        # This results in a distance matrix of shape (num_spikes, num_clusters)
        diff = spike_positions[:, np.newaxis, :] - cluster_positions[np.newaxis, :, :]  # Shape: [num_spikes, num_clusters, 2]
        distances = np.linalg.norm(diff, axis=2)  # Euclidean distance along the last axis

        # Find the index of the closest cluster centroid for each spike
        min_indices = np.argmin(distances, axis=1)  # Shape: [num_spikes]

        # Map the indices back to cluster indices
        closest_clusters = cluster_indices[min_indices]  # Shape: [num_spikes]

        return closest_clusters

###########################################################################################################
###########################################################################################################
###########################################################################################################
# Paths
DRIVE_PATH = Path("C:/")
BASE_PATH = Path(DRIVE_PATH, "SGL_DATA", "Quicky_20190305_R02_Ori")
BIN_DIR = Path(BASE_PATH)
BIN_FILE = "Quicksilver_20190305_02_r1_g0_t0_train.imec.ap.bin"
ks_output_dir = BASE_PATH / "kilosort4_train"
test_dir = Path("C:/", "SGL_DATA", "Quicky_20190305_R02_Ori", "cuda_test_input")
bin_file = BIN_DIR / BIN_FILE
ibatch = 2 * 30000 * 60 // 60000

# Batch params
ops = np.load(ks_output_dir / "ops.npy", allow_pickle=True).item()
ops['iU'] = torch.tensor(ops['iU'], device='cuda')
ops['iC'] = torch.tensor(ops['iC'], device='cuda')
ops['iCC'] = torch.tensor(ops['iCC'], device='cuda')
ops['Wrot'] = torch.tensor(ops['Wrot'], device='cuda')
ops['iKxx'] = torch.tensor(ops['iKxx'], device='cuda')
ops['wPCA'] = torch.tensor(np.load(ks_output_dir / 'wPCA.npy'), device='cuda')
for key in ops['preprocessing']:
    ops['preprocessing'][key] = torch.tensor(ops['preprocessing'][key], device='cuda')
bfile = bfile_from_ops(ops=ops, device='cuda', filename=bin_file)

batch_size = 3000

# Tensor op params
wPCA = torch.tensor(ops['wPCA']).contiguous()
Wall3 = torch.tensor(torch.from_numpy(np.load(ks_output_dir / 'Wall3.npy')), device='cuda')
ctc = torch.tensor(torch.from_numpy(np.load(ks_output_dir / 'ctc.npy')), device='cuda')
nm = (Wall3**2).sum(-1).sum(-1)
M = int(ops['nt'])
W = batch_size
Th_learned = ops['Th_learned']
iCC, iU, Ucc = prepare_extract(ops, torch.tensor(Wall3), ops['settings']['nearest_chans'], device='cpu')
cluster_centroids = np.load(ks_output_dir / 'cluster_centroids.npy', allow_pickle=True).item()

# ------ Start writing test data ------
# 1. Initial batch
X = bfile.padded_batch_to_torch(ibatch)
X = X[:-1, :batch_size]
print(f"Writing batch to {test_dir / 'batch'} with shape {X.shape}.")

if True:
    write_tensor(test_dir / "batch", X)

# 2. Mean removal
X = X - X.mean(1).unsqueeze(1)
print(f"Writing mean-removed batch to {test_dir / 'batch_mean_removed'} with shape {X.shape}.")
if True:
    write_tensor(test_dir / "batch_mean_removed", X)

# 3. Median removal
X = X - torch.median(X, 0)[0]
print(f"Writing median-removed batch to {test_dir / 'batch_median_removed'} with shape {X.shape}.")
if True:
    write_tensor(test_dir / "batch_median_removed", X)

# 4. Highpass filter
fwav = torch.tensor(fft_highpass(torch.tensor(bfile.hp_filter), NT=X.shape[1]), device=X.device)
X = torch.real(ifft(fft(X) * torch.conj(fwav)))
X = fftshift(X, dim = -1)
print(f"Writing highpassed batch to {test_dir / 'batch_highpassed'} with shape {X.shape}.")
if True:
    write_tensor(test_dir / "batch_highpassed", X)

# 5. Whiten
X = torch.tensor(bfile.whiten_mat, device=X.device) @ X
print(f"Writing whitened batch to {test_dir / 'batch_whitened'} with shape {X.shape}.")
if True:
    write_tensor(test_dir / "batch_whitened", X)

# 6. Drift correect
driftmat = get_drift_matrix(ops, bfile.dshift[-1], device=bfile.device)
X = driftmat @ X
print(f"Writing drift corrected batch to {test_dir / 'drift_corrected'} with shape {X.shape}.")
if True:
    write_tensor(test_dir / "batch_drift_corrected", X)

# 7. Project batch to PCA space
B = conv1d(X.unsqueeze(1), wPCA.unsqueeze(1), padding=M//2)
print(f"Writing batchPCA to {test_dir / 'batchPCA'} with shape {B.shape}.")
if True:
    write_tensor(test_dir / "batchPCA", B)

# 8. Compute cross-correlation with templates
B = torch.einsum('TkC, CkW -> TW', Wall3, B)
print(f"Writing convResult to {test_dir / 'convResult'} with shape {B.shape}.")
if True:
    write_tensor(test_dir / "convResult", B)

# 9. Normalize the result
trange = torch.arange(-M, M+1, device='cuda')
tiwave = torch.arange(-(M//2), M//2+1, device='cuda') 
st = torch.zeros((100000,2), dtype = torch.int64, device = 'cuda')
amps = torch.zeros((100000,1), dtype = torch.float, device = 'cuda')
Xres = X.clone()
Xreconstructed = torch.zeros(X.shape)
k = 0

Cf = torch.relu(B)**2 / nm.unsqueeze(-1)
Cf[:, :M] = 0
Cf[:, -M:] = 0
print(f"Writing Cf to {test_dir / 'Cf'} with shape {Cf.shape}.")
if True:
    write_tensor(test_dir / "Cf", Cf)

# 10. Collapse along timepoint with max
Cfmax, imax = torch.max(Cf, 0)
print(f"Writing Cfmax to {test_dir / 'Cfmax'} with shape {Cfmax.shape}.")
if True:
    write_tensor(test_dir / "Cfmax", Cfmax)

# 11. Max pool
Cmax  = max_pool1d(Cfmax.unsqueeze(0).unsqueeze(0), (2*M+1), stride = 1, padding = (M))
print(f"Writing Cfmaxpool to {test_dir / 'Cfmaxpool'} with shape {Cmax.shape}.")
if True:
    write_tensor(test_dir / 'Cfmaxpool', Cmax)

# 12. Matching indices
cnd1 = Cmax[0,0] > Th_learned**2
cnd2 = torch.abs(Cmax[0,0] - Cfmax) < 1e-9
xs = torch.nonzero(cnd1 * cnd2) 
iX = xs[:,:1]
iY = imax[iX]

print(f"Writing matched spike times to {test_dir / 'spikeTimes'} with shape {iX.shape}")
if True:
    write_tensor(test_dir / 'iX', iX)

print(f"Writing matched spike templates to {test_dir / 'spikeTemplates'} with shape {iY.shape}")
if True:
    write_tensor(test_dir / 'iY', iY)

# 13. Amplitudes
nsp = len(iX)
st[k:k+nsp, 0] = iX[:,0]
st[k:k+nsp, 1] = iY[:,0]
amps[k:k+nsp] = B[iY,iX] / nm[iY]
amp = amps[k:k+nsp]

print(f"Writing spike amplitudes to {test_dir / 'spikeAmplitudes'} with shape {amp.shape}.")
if True:
    write_tensor(test_dir / 'spikeAmplitudes', amp)

# 14. Update residuals
k += nsp
n = 1
for j in range(n):
    Xres[:, iX[j::n] + tiwave]  -= amp[j::n] * torch.einsum('ijk, jl -> kil', Wall3[iY[j::n,0]], wPCA)
    conv_contribution = amp[j::n] * ctc[:,iY[j::n,0], :]
    B[   :, iX[j::n] + trange]  -= conv_contribution
st = st[:k]
amps = amps[:k]

print(f"Writing updated convResult to {test_dir / 'convResultUpdated'} with shape {B.shape}.")
if True:
    write_tensor(test_dir / 'convResultUpdated', B)
print(f"Writing convContri to {test_dir / 'convContri'} with shape {conv_contribution.shape}.")
if True:
    write_tensor(test_dir / 'convContri', conv_contribution)
print(f"Writing Xres to {test_dir / 'Xres'} with shape {Xres.shape}.")
if True:
    write_tensor(test_dir / 'Xres', Xres)

# 15. Compute xs, ys
xfeat = Xres[iCC[:, iU[st[:,1:2]]],st[:,:1] + tiwave] @ ops['wPCA'].T
xfeat += amps * Ucc[:,st[:,1]]
tF =  xfeat.transpose(0,1).cpu()
xs, ys = compute_spike_positions(st, tF, ops)
print(f"Writing xs to {test_dir / 'xs'} with shape {xs.shape}.")
if True:
    write_tensor(test_dir / 'xs', xs)
print(f"Writing ys to {test_dir / 'ys'} with shape {ys.shape}.")
if True:
    write_tensor(test_dir / 'ys', ys)

# Compute closest clusters
closest_clusters = compute_closest_clusters(cluster_centroids, xs, ys)
print(f"Writing closest clusters to {test_dir / 'closest_clusters'} with shape {closest_clusters.shape}")
if True:
    write_tensor(test_dir / 'closest_clusters', closest_clusters)