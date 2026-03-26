from pathlib import Path
import numpy as np
import cupy as cp  # Using CuPy for GPU acceleration
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
import matplotlib.pyplot as plt
import tqdm

__analysis_metadata__ = {
    "name": "Decoder LOO Accuracy Plot",
    "description": "Compute PSTH given spike and event data.",
    "parameters": [
        ("Training start (samples)", "int"),
        ("Training end (samples)", "int"),
        ("Event file names (.txt's)", "list"),
    ]
}

def LOO_accuracy(spike_times, spike_templates, spike_amplitudes, 
                 event_times, event_labels, training_start, training_end, lookahead_start):
    """
    Computes the Leave-One-Out (LOO) accuracy of a logistic regression decoder
    trained on binned spike counts.

    Parameters:
        spike_times (np.ndarray): Array of spike times.
        spike_templates (np.ndarray): Array of spike template identifiers.
        spike_amplitudes (np.ndarray): Array of spike amplitudes.
        event_times (np.ndarray): Array of event times.
        event_labels (np.ndarray): Array of event labels.
        training_start (int): Start sample for training data.
        training_end (int): End sample for training data.
        lookahead_start (int): Lookahead start (in ms) relative to the event.

    Returns:
        float: LOO accuracy in percentage.
    """
    # Convert inputs to CuPy arrays.
    spike_times_cp = cp.asarray(spike_times)
    spike_templates_cp = cp.asarray(spike_templates)
    spike_amplitudes_cp = cp.asarray(spike_amplitudes)
    event_times_cp = cp.asarray(event_times)
    event_labels_cp = cp.asarray(event_labels)

    # Crop spike data to the training period.
    left_index = int(cp.asnumpy(cp.where(spike_times_cp > training_start)[0][0]))
    right_index = int(cp.asnumpy(cp.where(spike_times_cp < training_end)[0][-1]))
    spike_times_cp = spike_times_cp[left_index: right_index]
    spike_templates_cp = spike_templates_cp[left_index: right_index]
    spike_amplitudes_cp = spike_amplitudes_cp[left_index: right_index]

    # Define the time range based on the training spikes.
    spike_start = spike_times_cp[0]
    spike_end = spike_times_cp[-1]

    # Crop events to fall within the spike training period.
    left_evt = int(cp.asnumpy(cp.where(event_times_cp > spike_start)[0][0]))
    right_evt = int(cp.asnumpy(cp.where(event_times_cp < spike_end)[0][-1]))
    event_times_cp = event_times_cp[left_evt: right_evt]
    event_labels_cp = event_labels_cp[left_evt: right_evt]

    # Binning parameters: 100ms window and 10ms step (converted to samples at 30kHz).
    bin_size = 100 * 30  # 100ms window (samples)
    bin_inc = 10 * 30    # 10ms step (samples)

    # Determine overall time range.
    base_time = cp.minimum(event_times_cp.min(), spike_times_cp.min())
    max_time = cp.maximum(event_times_cp.max(), spike_times_cp.max())
    num_bins = int(cp.floor((max_time - base_time - bin_size) / bin_inc).get()) + 1

    # Get the set of unique templates.
    templates = cp.unique(spike_templates_cp)
    num_templates = int(templates.size)

    # Pre-compute bin boundaries.
    bin_left_edges = base_time + cp.arange(num_bins) * bin_inc
    bin_right_edges = bin_left_edges + bin_size

    # Create a 2D array for binned spike counts.
    binned_counts = cp.zeros((num_bins, num_templates), dtype=cp.int32)
    for i, tmpl in enumerate(cp.asnumpy(templates)):
        tmpl_spikes = spike_times_cp[spike_templates_cp == tmpl]
        start_indices = cp.searchsorted(tmpl_spikes, bin_left_edges, side='left')
        end_indices = cp.searchsorted(tmpl_spikes, bin_right_edges, side='left')
        binned_counts[:, i] = end_indices - start_indices

    # Compute event bin indices.
    # The lookahead window starts at event_time + lookahead_start*30 samples.
    event_bin_indices = ((event_times_cp + lookahead_start * 30 - base_time) // bin_inc).astype(cp.int32)
    valid = event_bin_indices < num_bins
    if not bool(cp.all(valid)):
        print("Warning: Some event times fall outside the binned range and will be ignored.")
        event_bin_indices = event_bin_indices[valid]
        event_labels_cp = event_labels_cp[valid]

    # Select features for the corresponding event bins.
    X = binned_counts[event_bin_indices]
    y = event_labels_cp

    # Convert features to NumPy arrays for scikit-learn.
    X = cp.asnumpy(X)
    y = cp.asnumpy(y)

    # Standardize features.
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Compute Leave-One-Out Cross-Validation Accuracy.
    loo = LeaveOneOut()
    loo_scores = cross_val_score(LogisticRegression(), X_norm, y, cv=loo)
    loo_accuracy = np.mean(loo_scores) * 100

    return loo_accuracy


def run(base_dir, training_start, training_end, eventfile_names):
    """
    Runs the Decoder LOO analysis pipeline using specified event files.
    
    Parameters:
        base_dir (str or Path): Base directory where data folders reside.
        training_start (int): Starting sample for training data.
        training_end (int): Ending sample for training data.
        eventfile_names (list of str): List of eventfile names (e.g. ['eventfile_15.txt', ...])
    """
    base_dir = Path(base_dir)
    decoder_dir = base_dir / "decoder_input"
    # Use the provided list of eventfile names
    eventfile_paths = [decoder_dir / name for name in eventfile_names]
    
    print(f"Using eventfiles: {[p.name for p in eventfile_paths]}")
    
    # Load Kilosort spikes.
    ks_dir = base_dir / "kilosort4"
    ks_spike_times = np.load(ks_dir / "spike_times.npy")
    ks_spike_templates = np.load(ks_dir / "spike_templates.npy")
    ks_spike_amplitudes = np.load(ks_dir / "amplitudes.npy")

    # Load OSS spikes.
    oss_dir = base_dir / "python_oss_output"
    oss_spike_times = []
    oss_spike_templates = []
    oss_spike_amplitudes = []
    with open(oss_dir / 'spikeOutput_assign_pca60.txt', 'r') as f:
        for line in f:
            tokens = line.split(',')
            oss_spike_times.append(int(tokens[0]))
            oss_spike_templates.append(int(tokens[1]))
            oss_spike_amplitudes.append(float(tokens[2]))
    oss_spike_times = np.array(oss_spike_times)
    oss_spike_templates = np.array(oss_spike_templates)
    oss_spike_amplitudes = np.array(oss_spike_amplitudes)

    # Helper function to load event data from a file.
    def load_event_data(event_file_path):
        event_times = []
        event_labels = []
        with open(event_file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                event_times.append(float(tokens[0]))
                event_labels.append(int(tokens[1]))
        return np.array(event_times), np.array(event_labels)

    # (Assuming LOO_accuracy is defined elsewhere in this module)
    loo_ks_avg = []
    loo_oss_avg = []
    lookahead_starts = np.arange(-600, 1401, 10)
    for la in tqdm.tqdm(np.array(lookahead_starts), ascii=True):
        ks_acc_list = []
        oss_acc_list = []
        for event_file in eventfile_paths:
            event_times, event_labels = load_event_data(event_file)
            acc_ks = LOO_accuracy(ks_spike_times, ks_spike_templates, ks_spike_amplitudes,
                                  event_times, event_labels, training_start, training_end, la)
            acc_oss = LOO_accuracy(oss_spike_times, oss_spike_templates, oss_spike_amplitudes,
                                   event_times, event_labels, training_start, training_end, la)
            ks_acc_list.append(acc_ks)
            oss_acc_list.append(acc_oss)
        loo_ks_avg.append(np.mean(ks_acc_list))
        loo_oss_avg.append(np.mean(oss_acc_list))

    plt.figure(figsize=(10, 6))
    plt.plot(lookahead_starts, loo_ks_avg, label='Kilosort', marker='o')
    plt.plot(lookahead_starts, loo_oss_avg, label='OSS', marker='s')
    plt.xlabel("Lookahead Start (ms)")
    plt.ylabel("LOO Accuracy (%)")
    plt.title("Averaged LOO Accuracy vs. Lookahead Start Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_dir / 'LOO_avg.png'}")
    plt.clf()