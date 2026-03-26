#!/usr/bin/env python3
"""
NoiseCorrelation.py

Compute pairwise noise correlations between neurons based on trial-to-trial spike count variability
around event onsets within a specified lookahead window.  For each event label k:
 1. Build the spike-count matrix C^{(k)} of shape [neurons x trials].
 2. Z-score each neuron's trial counts: (C_ik - μ_i)/σ_i, with a small std threshold.
 3. Compute the Pearson correlation matrix on the z-scored data.

Then average those correlation matrices across labels (skipping NaNs) to get a single
noise-correlation matrix.

The script saves:
 - 'noise_correlation_matrix.png': heatmap
 - 'noise_correlation_histogram.png': distribution of upper-triangular values
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

__analysis_metadata__ = {
    "name": "Noise Correlation",
    "description": "Compute the noise correlation between neurons based on trial-to-trial spike count variabilitiy around event onsets within a specified time window.",
    "parameters": [
        ("Event File (.txt)", "FilePath", r"^(\d+)\s+(\d+)$"),
        ("Spike File (.txt)", "FilePath", r"^(\d+),(\d+),(\d+\.\d+)(?:,[^,]+)*$"),
        ("Lookahead start (samples)", "int"),
        ("Lookahead end (samples)", "int")
    ]
}

def run(base_path, eventfile, spike_file, lookahead_start, lookahead_end, std_thresh=1e-12):
    print(f"Using spikes from {spike_file} and events from {eventfile} with lookahead window [{lookahead_start}, {lookahead_end}].")
    """
    Compute the noise correlation matrix for neurons, with per-condition z-scoring.

    Parameters
    ----------
    base_path : str
        Directory containing the event and spike files.
    eventfile : str
        Filename (relative to base_path) of the event file. Each line:
        <event_time> <label>
    spike_file : str
        Filename (relative to base_path) of the spike file. Each line:
        <spike_time>,<neuron_id>,<amplitude>
    lookahead_start : float
        Start of window (relative to event_time) to count spikes.
    lookahead_end : float
        End of window (relative to event_time) to count spikes.
    std_thresh : float, optional
        Minimum standard deviation for z-scoring; values below this are set to 1.

    Returns
    -------
    noise_corr : np.ndarray
        (n_neurons x n_neurons) averaged noise-correlation matrix.
    """
    # Load event times and labels
    ev_path = os.path.join(base_path, eventfile)
    events, labels = [], []
    with open(ev_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                events.append(float(parts[0]))
                labels.append(int(parts[1]))
            except ValueError:
                continue

    unique_labels = sorted(set(labels))
    if not unique_labels:
        raise RuntimeError("No valid event labels found in eventfile.")

    # Load spike data
    spk_path = os.path.join(base_path, spike_file)
    try:
        data = np.loadtxt(spk_path, delimiter=',')
    except Exception as e:
        raise RuntimeError(f"Error loading spike file: {e}")
    if data.size == 0:
        raise ValueError("Spike file is empty or malformed.")

    spike_times = data[:, 0]
    spike_neurons = data[:, 1].astype(int)

    unique_neurons = np.unique(spike_neurons)
    n_neurons = len(unique_neurons)
    neuron_to_idx = {n: i for i, n in enumerate(unique_neurons)}

    # Prepare accumulators for raw Pearson sums and counts
    corr_sum = np.zeros((n_neurons, n_neurons), dtype=float)
    count_mat = np.zeros((n_neurons, n_neurons), dtype=int)

    # Process each label separately
    for lbl in unique_labels:
        # get event times for this label
        ev_times = [t for t, l in zip(events, labels) if l == lbl]
        n_trials = len(ev_times)
        if n_trials < 2:
            # skip labels without enough trials for correlation
            continue

        # Build spike-count matrix for this label: neurons x trials
        counts = np.zeros((n_neurons, n_trials), dtype=float)
        for ti, t in enumerate(ev_times):
            start = t + lookahead_start
            end = t + lookahead_end
            mask = (spike_times >= start) & (spike_times < end)
            idxs = [neuron_to_idx[n] for n in spike_neurons[mask]]
            if idxs:
                cnt = np.bincount(idxs, minlength=n_neurons)
            else:
                cnt = np.zeros(n_neurons, dtype=int)
            counts[:, ti] = cnt

        # Z-score each neuron's trial counts
        means = counts.mean(axis=1, keepdims=True)
        stds = counts.std(axis=1, keepdims=True)
        stds_fixed = stds.copy()
        stds_fixed[stds_fixed < std_thresh] = 1.0
        counts_z = (counts - means) / stds_fixed

        # Compute Pearson correlation on z-scored data
        corr_mat = np.corrcoef(counts_z)
        # Accumulate non-NaN entries
        valid = ~np.isnan(corr_mat)
        corr_sum[valid] += corr_mat[valid]
        count_mat[valid] += 1

    # Average raw Pearson correlations across labels, skipping missing entries
    mask_valid = count_mat > 0
    noise_corr = np.full((n_neurons, n_neurons), np.nan)
    noise_corr[mask_valid] = corr_sum[mask_valid] / count_mat[mask_valid]

    # --- Save heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(noise_corr, interpolation='nearest', aspect='auto')
    ax.set_title('Noise Correlation Matrix')
    ax.set_xlabel('Neuron index')
    ax.set_ylabel('Neuron index')
    fig.colorbar(cax, ax=ax, label='Pearson r')
    hm_path = os.path.join(base_path, 'noise_correlation_matrix.png')
    fig.savefig(hm_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Save histogram of upper-triangular values (i<j) ---
    triu_idx = np.triu_indices(n_neurons, k=1)
    vals = noise_corr[triu_idx]
    vals = vals[~np.isnan(vals)]
    print(f"mean = {vals.mean()}")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(vals, bins='auto')
    ax2.set_title('Histogram of Noise Correlations (upper triangular)')
    ax2.set_xlabel('Correlation coefficient')
    ax2.set_ylabel('Count')
    hist_path = os.path.join(base_path, 'noise_correlation_histogram.png')
    fig2.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
