import sys
from pathlib import Path
from crop_methods import *
import kilosort
import numpy as np
import torch
import subprocess
import time
import os
import ctypes
import shlex
from kilosort.preprocessing import get_drift_matrix
from kilosort.template_matching import prepare_extract
import copy
import random

def random_index(shape):
    """
    Generates a random index for a tensor with the given shape.
    """
    if not isinstance(shape, tuple):
        raise TypeError("Shape must be a tuple.")
    
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError("All dimensions in shape must be positive integers.")
    
    return tuple(random.randrange(dim) for dim in shape)

DRIVE_PATH = Path("C:/")
BASE_PATH = Path(DRIVE_PATH, "SGL_DATA", "05_31")
KS_OUTPUT_DIR = Path("C:/Users/Spike Sorter/source/repos/OnlineSpikes_v2/src/kilosort4")

spike_times = np.load(KS_OUTPUT_DIR / "spike_times.npy")
spike_templates = np.load(KS_OUTPUT_DIR / "spike_templates.npy")
spike_detection_templates = np.load(KS_OUTPUT_DIR / "spike_detection_templates.npy")
templates = np.load(KS_OUTPUT_DIR / "templates.npy")
whitening_mat = np.load(KS_OUTPUT_DIR / "whitening_mat.npy")
channel_map = np.load(KS_OUTPUT_DIR / "channel_map.npy")
#cluster_kslabels = load_cluster_ks_labels(KS_OUTPUT_DIR / "cluster_KSLabel.tsv")
spike_positions = np.load(KS_OUTPUT_DIR / "spike_positions.npy")
ops = np.load(KS_OUTPUT_DIR / "ops.npy", allow_pickle=True).item()
Wall3 = np.ascontiguousarray(np.load(KS_OUTPUT_DIR / "Wall3.npy"))
Wall3_permuted = np.ascontiguousarray(torch.tensor(Wall3.copy()).permute(0, 2, 1).contiguous().numpy())
ctc = np.ascontiguousarray(np.load(KS_OUTPUT_DIR / "ctc.npy"))
ctc_permuted = np.ascontiguousarray(torch.tensor(ctc.copy()).permute(1, 0, 2).contiguous().numpy())
wPCA = np.ascontiguousarray(np.load(KS_OUTPUT_DIR / "wPCA.npy"))
wPCA_permuted = copy.deepcopy(wPCA)
wPCA_permuted = torch.from_numpy(wPCA_permuted)
wPCA_permuted = wPCA_permuted.permute(1, 0).contiguous().numpy()
cluster_centroids = np.load(KS_OUTPUT_DIR / "cluster_centroids.npy", allow_pickle=True).item()
cluster_centroids = [value for value in cluster_centroids.values()]
drift_matrix = np.array(get_drift_matrix(ops, ops['dshift'][-1], device='cpu'))
iCC, iU, Ucc = prepare_extract(ops, torch.tensor(Wall3), ops['settings']['nearest_chans'], device='cpu')
preclustered_template_waveforms = np.ascontiguousarray(torch.einsum('ijk, jl -> kil', torch.tensor(Wall3), torch.tensor(wPCA)).permute(1, 2, 0).contiguous().numpy())

output_dir = Path(BASE_PATH, "oss_training_full")
output_dir.mkdir(parents=True, exist_ok=True)

tensors = [
   # ("spike_times", spike_times),
   # ("spike_templates", spike_templates),
   # ("spike_detection_templates", spike_detection_templates),
   # ("templates", templates),
   # ("whitening_mat", whitening_mat),
   # ("channel_map", channel_map),
   # ("spike_positions", spike_positions),
    ("Wall3", Wall3),
    ("Wall3_permuted", Wall3_permuted),
    ("ctc", ctc),
    ("ctc_permuted", ctc_permuted),
    ("wPCA", wPCA),
    ("wPCA_permuted", wPCA_permuted),
    ("drift_matrix", drift_matrix),
    # ("iCC", iCC),
    #("iU", iU),
    ("Ucc", Ucc),
    ("preclustered_template_waveforms", preclustered_template_waveforms)
]

def generate_random_indices_for_tensor(tensor, tensor_name, output_dir, num_samples=10):
    """
    Generates random indices for a given tensor and writes them to a .txt file.
    """
    print(f"Generating test data for {tensor_name} with shape {tensor.shape} with type {tensor.dtype}")
    txt_file_path = output_dir / f"{tensor_name}.txt"
    with open(txt_file_path, 'w') as f:
        for _ in range(num_samples):
            # Handle numpy arrays, torch tensors, and lists
            if isinstance(tensor, np.ndarray) or torch.is_tensor(tensor) or isinstance(tensor, list):
                if isinstance(tensor, list):
                    shape = (len(tensor),)
                else:
                    shape = tensor.shape
                if shape:  # Non-scalar tensor
                    index = random_index(shape)
                    if isinstance(tensor, list):
                        value = tensor[index[0]]
                    elif torch.is_tensor(tensor):
                        value = tensor[index].item()
                    else:
                        value = tensor[index]
                    num_dims = len(shape)
                    indices_str = ' '.join(map(str, index))
                    line = f"{num_dims} {indices_str} {value}\n"
                else:  # Scalar tensor
                    value = tensor.item() if isinstance(tensor, np.ndarray) or torch.is_tensor(tensor) else tensor
                    num_dims = 0
                    line = f"{num_dims} {value}\n"
                f.write(line)
            else:
                # Unsupported type
                continue

for tensor_name, tensor in tensors:
    generate_random_indices_for_tensor(tensor, tensor_name, output_dir, num_samples=10)

batch = 
