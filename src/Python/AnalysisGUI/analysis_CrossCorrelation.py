import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

__analysis_metadata__ = {
    "name": "Cross-correlation Matrix",
    "description": "Compute zero-lag cross-correlation coefficient matrix for all neuron spike masks and save as image and CSV.",
    "parameters": [
        ("Spike File (.txt)", "FilePath", r"^(\d+),(\d+),(\d+\.\d+)(?:,[^,]+)*$")
    ]
}

def run(base_path, spike_output_file):
    """
    Loads spike times and templates, builds binary spike masks (30-sample bins) for each neuron,
    computes the Pearson correlation coefficient matrix across all neuron pairs,
    saves the matrix to CSV and as a heatmap image, then returns it.
    """
    # Load spike data
    spike_times = []
    spike_templates = []
    with open(spike_output_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                t = int(row[0]); templ = int(row[1])
            except (ValueError, IndexError):
                continue
            spike_times.append(t)
            spike_templates.append(templ)

    spike_times = np.array(spike_times)
    spike_templates = np.array(spike_templates)
    if spike_times.size == 0:
        print("No spike data found.")
        return None

    # Define binning parameters
    window_size = 30  # samples per bin
    max_time = spike_times.max()
    num_bins = int(max_time // window_size) + 1

    # Identify unique neurons
    neurons = np.unique(spike_templates)
    n_neurons = neurons.size

    # Build masks: shape (n_neurons, num_bins)
    masks = np.zeros((n_neurons, num_bins), dtype=float)
    for i, neuron in enumerate(neurons):
        bins = spike_times[spike_templates == neuron] // window_size
        bins = np.unique(bins)
        masks[i, bins] = 1

    # Compute correlation matrix (rows=neurons)
    corr_mat = np.corrcoef(masks)

    # Save to CSV with neuron IDs as header
    out_csv = Path(base_path) / 'crosscorr_matrix.csv'
    header = ','.join(map(str, neurons.tolist()))
    np.savetxt(out_csv, corr_mat, delimiter=',', header=header, comments='')
    print(f"Saved cross-correlation matrix to {out_csv}")

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_mat, aspect='equal', interpolation='none')
    plt.colorbar(im, label='Correlation Coefficient')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    plt.title('Cross-correlation Matrix')
    plt.tight_layout()

    # Save figure
    out_img = Path(base_path) / 'crosscorr_mat.png'
    plt.savefig(out_img)
    print(f"Saved cross-correlation heatmap to {out_img}")