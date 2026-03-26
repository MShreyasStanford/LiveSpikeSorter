#!/usr/bin/env python3
"""
SignalCorrelation.py

Compute pairwise signal correlations between neurons based on their firing rates
around event onsets within a specified lookahead window, save a heatmap, and
handle edge cases gracefully.
"""
import os
import numpy as np
import matplotlib
# Use non-interactive backend to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt

__analysis_metadata__ = {
    "name": "Signal Correlation",
    "description": "Compute pairwise signal correlations between neurons based on their firing rates around event onsets within a specified time window.",
    "parameters": [
        ("Event File", "FilePath", r"^(\d+)\s+(\d+)$"),
        ("Spike File", "FilePath"),
        ("Lookahead Start (samples)", "int"),
        ("Lookahead End (samples)", "int")
    ]
}


def run(base_path, eventfile, spike_file, lookahead_start, lookahead_end):
    print(f"Using spikes from {spike_file} and events from {eventfile} with lookahead window [{lookahead_start}, {lookahead_end}].")
    # Load events
    event_path = os.path.join(base_path, eventfile)
    events, labels = [], []
    with open(event_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                events.append(float(parts[0]))
                labels.append(int(parts[1]))
            except ValueError:
                continue

    # Unique labels
    unique_labels = sorted(set(labels))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    n_labels = len(unique_labels)

    # Load spikes
    spike_path = os.path.join(base_path, spike_file)
    try:
        data = np.loadtxt(spike_path, delimiter=',')
    except Exception as e:
        raise RuntimeError(f"Error loading spike file: {e}")
    if data.ndim == 1 and data.size == 0:
        raise ValueError("Spike file is empty or malformed.")

    spike_times = data[:, 0]
    spike_neurons = data[:, 1].astype(int)

    # Unique neurons
    unique_neurons = np.unique(spike_neurons)
    neuron_to_idx = {n: i for i, n in enumerate(unique_neurons)}
    n_neurons = len(unique_neurons)

    # Prepare accumulation
    window_length = lookahead_end - lookahead_start
    if window_length <= 0:
        raise ValueError("lookahead_end must be greater than lookahead_start.")

    rates_sum = np.zeros((n_neurons, n_labels))
    event_counts = np.zeros(n_labels)

    # Accumulate firing rates per label
    for t, lbl in zip(events, labels):
        idx_lbl = label_to_idx[lbl]
        start, end = t + lookahead_start, t + lookahead_end
        mask = (spike_times >= start) & (spike_times < end)
        idxs = [neuron_to_idx[n] for n in spike_neurons[mask]]
        if idxs:
            counts = np.bincount(idxs, minlength=n_neurons)
        else:
            counts = np.zeros(n_neurons, dtype=int)
        rates = counts / window_length
        rates_sum[:, idx_lbl] += rates
        event_counts[idx_lbl] += 1

    # Average rates across events of each label
    for i in range(n_labels):
        if event_counts[i] > 0:
            rates_sum[:, i] /= event_counts[i]

    # Compute Pearson correlation matrix (np.corrcoef handles normalization)
    corr_matrix = np.corrcoef(rates_sum)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr_matrix, interpolation='nearest', aspect='auto')
    ax.set_title('Signal Correlation Matrix')
    ax.set_xlabel('Template index')
    ax.set_ylabel('Template index')
    fig.colorbar(cax, ax=ax, label='Pearson r')

    out_fig = os.path.join(base_path, "signal_correlation_matrix.png")
    fig.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Extract values above the diagonal and skip NaNs
    triu_indices = np.triu_indices(n_neurons, k=1)
    triu_vals = corr_matrix[triu_indices]
    triu_vals = triu_vals[~np.isnan(triu_vals)]

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(triu_vals, bins='auto')
    print(f"mean = {np.sum(triu_vals.mean())}")
    ax2.set_title('Histogram of Signal Correlations (upper triangular)')
    ax2.set_xlabel('Correlation coefficient')
    ax2.set_ylabel('Count')

    hist_path = os.path.join(base_path, "signal_correlation_histogram.png")
    fig2.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
