import sys
from pathlib import Path
from crop_methods import k_most_active_channels, k_most_active_channels_parallelized, crop_kilosort_output
import kilosort
import numpy as np
import torch
import subprocess
import time
import os
import ctypes
import shlex

print(kilosort.__file__)

recordings = [
    ("C://SGL_DATA//joplin_20240208//imec_raw//neuropixels_NHP_channel_map_dev_staggered_v1.mat", 
        "C://SGL_DATA//joplin_20240215//kilosort4", 
        "C://SGL_DATA//joplin_20240215//imec_raw//240222_g0_t0_train.imec0.ap.bin"
    ),
]

ks_settings = {'n_chan_bin': 385}

for recording in recordings:
    print(f"Running Kilosort on recording {recording[2]}, storing results in {recording[1]}.")
    kilosort.run_kilosort(
        settings=ks_settings,
        probe_name=recording[0],
        results_dir=recording[1],
        filename=recording[2]
    )