import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
import subprocess
import torch
import torch.nn as nn
import random
import math
import scipy
from enum import Enum
import statistics
from tqdm import tqdm

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
SPIKES_OUTPUT_FILE = Path(BASE_PATH, "sorter_output", "spikeOutputCudaSmallNumChannels.txt")
OSS_TRAINING_PATH = Path("C:/SGL_DATA/05_31/oss_training_std/")

WINDOW_SIZE = 10 # multiple of # samples in template
SAMPLING_RATE_MS = 30
SIMULATED_NUM_SPIKES = 20
SIMULATED_AMPLITUDE = 1
IS_SIMULATED_NOISY = True
IS_SIMULATED = False

def psth(spikes, event_time, start_samp_offset=-600 * 30, end_samp_offset=500 * 30, inc=10 * 30, window_size=100 * 30):
    # The list of counts to plot for PSTH
    psth_x = []
    psth_y = []

    start = start_samp_offset
    end = start_samp_offset + window_size

    while end <= end_samp_offset:
        window_spike_count = 0
        for (sample, template) in spikes:
            if sample >= start + event_time and sample <= end + event_time:
                window_spike_count += 1

        psth_x.append(end / 30)
        psth_y.append(window_spike_count)
        start += inc
        end += inc

    return psth_x, psth_y
    
event_times = [1606338, 1739837, 1876583, 2009328, 2146080, 2278072, 2409818 ]
    
with open(SPIKES_OUTPUT_FILE, 'r') as file:
    spikes = [ (int(line.strip().split(',')[0].strip()), int(line.strip().split(',')[1].strip())) for line in file if len(line.split(',')) == 3 ]

for i, event_time in enumerate(event_times):
    psth_x, psth_y = psth(spikes, event_time)
    if i == 0:
        x = psth_x
        y = psth_y
    else:
        y = [y[j] + psth_y[j] for j in range(len(psth_y))]

plt.plot(x, y)
plt.show()


                    

