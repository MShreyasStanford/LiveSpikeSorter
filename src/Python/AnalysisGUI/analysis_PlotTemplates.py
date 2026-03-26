"""
plot_templates.py

This script loads data from the kilosort output directory (base_dir / "kilosort4") and
plots template waveforms for clusters with firing rate >= 1 Hz. For each eligible template:
  - The best channel (largest L1 norm) is extracted.
  - Its 1D waveform is mapped into a fixed drawing area (150×80 pixels) so that the waveform
    is centered inside this box.
  - The fixed drawing area is then placed via an AnnotationBbox so that its center coincides
    exactly with the cluster centroid.
  
No visible centroid dots are plotted. However, an invisible scatter is added to ensure the
axes autoscale to include all cluster centers. Finally, the axes limits are expanded by 10%
to ensure that the entire fixed waveform boxes are visible.

Usage:
    run(base_dir)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.lines import Line2D
import random
from matplotlib.font_manager import FontProperties
import pandas as pd

__analysis_metadata__ = {
    "name": "Plot Template Waveforms",
    "description": "Given a Kilosort directory, plot the template waveforms spatially.",
    "parameters": [
        ("Kilosort Directory", "DirPath")
    ]
}

def compute_template_positions(templates, channel_positions):
    num_temps = templates.shape[0]
    num_chans = templates.shape[2]
    template_positions = {}

    for i in range(num_temps):
        support = []    # List of channel indices that are not all zeros
        amplitudes = [] # List of amplitudes for each supported channel
        
        for j in range(num_chans):
            channel_waveform = templates[i, :, j]
            # Check whether this channel has any non-zero elements
            if np.any(channel_waveform != 0):
                # Compute amplitude as (max - min)
                amp = channel_waveform.max() - channel_waveform.min()
                support.append(j)
                amplitudes.append(amp)
                
        if support:
            amplitudes = np.array(amplitudes)
            # Calculate weighted sum of positions using amplitudes as weights
            weighted_sum = np.zeros(2)
            for idx, j in enumerate(support):
                weighted_sum += amplitudes[idx] * channel_positions[j]
            weighted_position = weighted_sum / amplitudes.sum()
        else:
            weighted_position = np.array([np.nan, np.nan])
        
        # Store the weighted (x, y) position as a tuple for the current template index
        template_positions[i] = tuple(weighted_position)

    return template_positions

def get_cluster_centroids(ks_dir):
    #return np.load(ks_dir / 'cluster_centroids.npy', allow_pickle=True).item()
    '''
    spike_positions = np.load(ks_dir / "spike_positions.npy")
    spike_templates = np.load(ks_dir / "spike_templates.npy")
    T = max(spike_templates) + 1
    average_pos = { template: (0, 0) for template in range(T) }
    counts = { template: 0 for template in range(T) }
    for (x, y), template in zip(spike_positions, spike_templates):
        average_pos[template] = (average_pos[template][0] + x, average_pos[template][1] + y)
        counts[template] += 1
    average_pos = { template: (average_pos[template][0] / counts[template], average_pos[template][1] / counts[template]) for template in range(T) }
    return average_pos
    '''
    templates = np.load(ks_dir / 'templates.npy')
    channel_positions = np.load(ks_dir / 'channel_positions.npy')
    return compute_template_positions(templates, channel_positions)
    

def run(base_dir, kilosort_dir):
    base_dir = Path(base_dir)
    ks_dir = kilosort_dir

    # Load font
    script_dir = Path(__file__).resolve().parent
    font_dir = Path(script_dir, "fonts", "gill-sans-2")
    font_prop = FontProperties(fname=font_dir / "Gill Sans.otf", size=14)

    # Load template types
    channel_positions = np.load(kilosort_dir / "channel_positions.npy")

    # Load cluster types
    cluster_ks_labels = pd.read_csv(kilosort_dir / "cluster_KSLabel.tsv", sep='\t', header=0)
    cluster_labels = dict(zip(cluster_ks_labels['cluster_id'], cluster_ks_labels['KSLabel']))

    # Load cluster centroids.
    cluster_centroids = get_cluster_centroids(ks_dir)

    # Load templates tensor (shape: (T, n, C)).
    try:
        templates = np.load(ks_dir / 'templates.npy')
    except Exception as e:
        print(f"Error loading templates.npy: {e}")
        return
    T, n, C = templates.shape
    print(f"Found {T} templates, each with {n} samples across {C} channels.")

    # Load spike data for firing rate computation.
    try:
        spike_times = np.load(ks_dir / 'spike_times.npy')
        spike_templates = np.load(ks_dir / 'spike_templates.npy')
    except Exception as e:
        print(f"Error loading spike data: {e}")
        return

    duration_sec = (spike_times.max() - spike_times.min()) / 30000.0
    print(f"Recording duration: {duration_sec:.2f} s")

    # Compute firing rates: template index i -> rate in Hz.
    firing_rates = {}
    for i in range(T):
        count = np.sum(spike_templates == i)
        firing_rates[i] = count / duration_sec

    # For debugging, print firing rates for clusters with centroids.
    for cluster_id in sorted(cluster_centroids.keys()):
        rate = firing_rates.get(cluster_id, 0)

    # Set up the figure.
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.axis('off')

    # Fixed drawing area dimensions.
    box_width, box_height = 150 / 2 * 1.2, 80 / 2

    plotted = 0
    # Loop over each cluster in the centroids dictionary.
    for cluster_id in sorted(cluster_centroids.keys()):
        rate  = firing_rates.get(cluster_id, 0)
        label = cluster_labels.get(cluster_id, '').lower()
        if rate < 1 or label != 'good':
            continue

        # Get the cluster centroid.
        cx, cy = cluster_centroids[cluster_id]

        # Extract the corresponding template (assuming cluster_id is the index).
        temp = templates[cluster_id, :, :]  # shape (n, C)
        l1_norms = np.sum(np.abs(temp), axis=0)
        best_channel = np.argmax(l1_norms)
        waveform = temp[:, best_channel]     # shape (n,)

        # Horizontal mapping: linearly map time indices from 5% to 95% of box width.
        x_pixels = np.linspace(0.05 * box_width, 0.95 * box_width, n)

        # Vertical mapping: center the waveform by subtracting its mean.
        mean_val = np.mean(waveform)
        dev = waveform - mean_val
        max_dev = np.max(np.abs(dev))
        if max_dev == 0:
            norm_waveform = np.full_like(waveform, 0.5)
        else:
            norm_waveform = 0.5 + 0.4 * (dev / max_dev)
        y_pixels = norm_waveform * box_height

        # Create the fixed drawing area.
        da = DrawingArea(box_width, box_height, 0, 0)
        color = (random.random(), random.random(), random.random())
        line = Line2D(x_pixels, y_pixels, color=color, lw=1.8, alpha=0.7)
        da.add_artist(line)

        # Place the drawing area so that its center aligns with the cluster centroid.
        ab = AnnotationBbox(da, (cx, cy), xycoords='data',
                            frameon=False, pad=0, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
        plotted += 1

    print(f"Plotted {plotted} templates.")
    
    # Add an invisible scatter of eligible cluster centers to force autoscaling.
    eligible_x = [cluster_centroids[c][0] for c in cluster_centroids
                  if firing_rates.get(c, 0) >= 1 and cluster_labels.get(c, '').lower() == 'good']
    eligible_y = [cluster_centroids[c][1] for c in cluster_centroids
                  if firing_rates.get(c, 0) >= 1 and cluster_labels.get(c, '').lower() == 'good']
    if eligible_x and eligible_y:
        ax.scatter(eligible_x, eligible_y, alpha=0)

    # Plot channel positions as low-opacity gray squares
    xs_ch = channel_positions[:, 0]
    ys_ch = channel_positions[:, 1]
    ax.scatter(xs_ch, ys_ch, marker='s', s=10, color='gray', alpha=0.3)

    # Get current x and y limits and then expand them by 10%
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 0.1*(x1-x0), x1 + 0.1*(x1-x0))
    ax.set_ylim(y0 - 0.1*(y1-y0), y1 + 0.1*(y1-y0))

    ax.grid(True, linestyle='--', alpha=0.5)
    # ensure no text elements remain
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{base_dir / 'template_plot.pdf'}")