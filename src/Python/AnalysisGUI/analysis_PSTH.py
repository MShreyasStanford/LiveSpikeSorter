from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import random
import os
import pickle
import numpy as np
import cupy as cp
import tqdm


__analysis_metadata__ = {
    "name": "Compute PSTHs",
    "description": "Compute PSTH given spike and event data.",
    "parameters": [
        ("Spike File (.txt)", "FilePath", r"^(\d+),(\d+),(\d+\.\d+)(?:,[^,]+)*$"),
        ("Event File", "FilePath", r"^(\d+)\s+(\d+)$"),
        ("Number of Trials to Subsample (-1 for all)", "int"),
        ("Type of PSTH", "Enum", [ "One PSTH per Neuron", "One PSTH per Neuron per Condition", "One PSTH with #-Template Lines", "One PSTH with 1 Line"])
    ]
}

def pearson_corr_numpy(x, y):
    x_arr = np.array(x)
    y_arr = np.array(y)
    return np.corrcoef(x_arr, y_arr)[0, 1]

def stonum(s):
    if '.' in s or 'e' in s or 'E' in s:
        return float(s)
    else:
        return int(s)

def deterministic_hash(s: str) -> int:
    hash_val = 5381
    for c in s:
        # Equivalent to: hash_val = hash_val * 33 + ord(c)
        hash_val = ((hash_val << 5) + hash_val) + ord(c)
    # Optionally, restrict to 32 bits
    return hash_val & 0xFFFFFFFF

def psth(event_time, spikes, templates=None, inc=10*30, window_size=100*30):
    if len(spikes) == 0:
        return None, None

    if min([spike[0] for spike in spikes]) >= event_time + 500 * 300:
        return None, None

    if max([spike[0] for spike in spikes]) <= event_time - 500 * 30:
        return None, None

    # The list of counts to plot for PSTH
    psth_x = []
    psth_y = []

    start = -500 * 30
    end = start + window_size
    
    while end <= 500 * 30:
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

def get_events(eventfile):
    print(f"Loading events from {eventfile}.")
    with open(eventfile, 'r') as file:
        return  [stonum(line) if len(line.split(' ')) <= 1 else stonum(line.split(' ')[0]) for line in file ]
    return None

def cropped_events(events, min_spike, max_spike, k):
    cropped_events = [ event for event in events if min_spike <= event <= max_spike ]
    if k == -1:
        return cropped_events
    cropped_events = sorted(random.sample(cropped_events, min(k, len(cropped_events))))
    return cropped_events

def get_spikes(file):
    print(f"Loading spikes from {file}")
    with open(file, 'r') as f:
        return [
            (stonum(line.split(',')[0]), stonum(line.split(',')[1]))
            for line in f
        ]
    return None

def compute_psth_data_gpu(spikes, events):
    psth_data = {}
    # Convert events list to a CuPy array (in sample units)
    events_cp = cp.asarray(events)  # shape: (num_events,)

    # Define parameters (in samples)
    inc = 10 * 30         # 300
    window_size = 100 * 30  # 3000
    start_initial = -500 * 30  # -6000
    max_end = 500 * 30         # 9000

    # Determine number of windows. We require: start_initial + j*inc + window_size <= max_end.
    num_windows = int((max_end - (start_initial + window_size)) // inc + 1)  # should be 41
    # Window offsets (in samples)
    window_starts = cp.arange(num_windows) * inc + start_initial  # shape: (num_windows,)
    # Standard x-axis computed as window_end/30
    standard_x = cp.asnumpy(window_starts + window_size) / 30  # convert back to time units

    # Determine maximum template number from spikes1 and spikes2
    T = max(s[1] for s in spikes) if spikes else 0

    for template in tqdm.tqdm(range(1, T + 1)):
        # Filter spikes for the current template
        template_spikes = [s for s in spikes if s[1] == template]
        
        # If there are no spikes in both lists, skip this template
        if not template_spikes:
            continue

        # Convert spike times to CuPy arrays (they are assumed to be in sample units)
        if template_spikes:
            S = cp.asarray([s[0] for s in template_spikes])
            S = cp.sort(S)
        else:
            S = cp.array([], dtype=cp.float32)

        lower_bounds = events_cp[:, None] + window_starts[None, :]
        upper_bounds = lower_bounds + window_size

        # Use searchsorted to count spikes falling in each window for S1:
        if S.size > 0:
            left_indices = cp.searchsorted(S, lower_bounds, side='left')
            right_indices = cp.searchsorted(S, upper_bounds, side='right')
            counts = right_indices - left_indices  # shape: (num_events, num_windows)
            agg_counts = cp.sum(counts, axis=0)       # shape: (num_windows,)
            aggregated_counts = cp.asnumpy(agg_counts)
        else:
            aggregated_counts = np.zeros(num_windows, dtype=int)

        psth_data[template] = {
            'standard_x': standard_x.tolist(), 
            'oss_y': aggregated_counts.tolist()
        }

    return psth_data

def hashed_pickle_file_name(file, eventfile, spikes, events):
    return deterministic_hash(f'{file}_{eventfile}_{len(spikes)}_{len(events)}')

def run(directory, file, eventfile, k, mode):
    # Load spike data from file
    spikes = get_spikes(file)

    if not spikes:
        print(f"{file} was not loaded correctly.")
        return

    # Parameters
    T = max([ spike[1] for spike in spikes ])

    # Load eventfile
    events = get_events(eventfile)

    if not events:
        print("Could not load events.")
        return

    # Crop events to cut on processing time
    min_spike = min([ spike[0] for spike in spikes ])
    max_spike = max([ spike[0] for spike in spikes ])
    events = cropped_events(events, min_spike, max_spike, k)

    # Check if pickled file exists
    psth_pickle_file = directory / f"{hashed_pickle_file_name(file, eventfile, spikes, events)}.pkl"

    if os.path.exists(psth_pickle_file):
        print(f"Loading PSTH data from pickle: {psth_pickle_file}")
        with open(psth_pickle_file, 'rb') as pf:
            psth_data = pickle.load(pf)
    else:
        psth_data = compute_psth_data_gpu(spikes, events)

        # After processing all templates, pickle the PSTH data.
        with open(psth_pickle_file, 'wb') as pf:
            pickle.dump(psth_data, pf)

    if mode == "One PSTH with #-Template Lines":
        # Initialize the figure and subplots
        fig, ax = plt.subplots(1, 1, figsize=(18, 10), sharex=True)
        ax.set_title(f"{file}")

        standard_x = None
        # Plot the lines for the single-neuron PSTHs
        for template in range(1, T + 1):
            oss_y = psth_data[template]['oss_y']
            curr_x = psth_data[template]['standard_x']

            if standard_x is None:
                standard_x = curr_x.copy()

            # Plot OSS data
            if not oss_y:
                oss_y = [0] * len(standard_x)

            ax.plot(standard_x, oss_y, label=f"Template {template}", alpha=0.7)

        # Adjust the main plot area to make room for TextBox on the right
        plt.subplots_adjust(right=0.8)
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust layout to make room for TextBox
        plt.savefig(f"{directory / 'single_neuron_psths.png'}")
        plt.clf()
    elif mode == "One PSTH with 1 Line":
        all_oss = []
        standard_x = None
        for template in range(1, T + 1):
            oss_y = psth_data[template]['oss_y']
            curr_x = psth_data[template]['standard_x']
            if standard_x is None:
                standard_x = curr_x.copy()
            if not oss_y:
                oss_y = [0] * len(standard_x)

            all_oss.append(oss_y)

        avg_oss = np.mean(all_oss, axis=0)

        if np.max(avg_oss) != 0:
            avg_oss = avg_oss / np.max(avg_oss)

        plt.plot(standard_x, avg_oss, alpha=0.9)
        plt.savefig(f"{directory / 'avg_psth.png'}")
        plt.clf()
    elif mode == "One PSTH per Neuron":
        all_oss = []
        for template in range(1, T + 1):
            oss_y = psth_data[template]['oss_y']
            if not oss_y:
                oss_y = [0] * len(standard_x)
            all_oss.append(oss_y)
