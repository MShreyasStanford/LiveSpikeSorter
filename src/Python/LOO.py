from pathlib import Path
import numpy as np
import cupy as cp  # Using CuPy for GPU acceleration
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
import matplotlib.pyplot as plt

def LOO_accuracy(spike_times, spike_templates, spike_amplitudes, 
                 event_times, event_labels, training_start, training_end, lookahead_start):
    """
    Computes the Leave-One-Out (LOO) accuracy of a logistic regression decoder trained on binned spike counts.
    
    This function assumes that spike data and event information are already loaded.
    It uses a fixed window of 100ms (100*30 samples) starting at the specified lookahead_start (in ms).
    CuPy is used for GPU-accelerated operations where possible.
    
    Parameters:
        spike_times (np.ndarray): Array of spike times.
        spike_templates (np.ndarray): Array of spike template identifiers.
        spike_amplitudes (np.ndarray): Array of spike amplitudes.
        event_times (np.ndarray): Array of event times.
        event_labels (np.ndarray): Array of event labels.
        training_start (int): Start sample for training data.
        training_end (int): End sample for training data.
        lookahead_start (int): Lookahead start (in ms) relative to the event at which a 100ms window begins.
    
    Returns:
        float: LOO accuracy in percentage.
    """
    # Convert inputs to CuPy arrays.
    spike_times = cp.asarray(spike_times)
    spike_templates = cp.asarray(spike_templates)
    spike_amplitudes = cp.asarray(spike_amplitudes)
    event_times = cp.asarray(event_times)
    event_labels = cp.asarray(event_labels)
    
    # Crop spike data to the training period.
    left_index = int(cp.asnumpy(cp.where(spike_times > training_start)[0][0]))
    right_index = int(cp.asnumpy(cp.where(spike_times < training_end)[0][-1]))
    spike_times = spike_times[left_index: right_index]
    spike_templates = spike_templates[left_index: right_index]
    spike_amplitudes = spike_amplitudes[left_index: right_index]
    
    # Define the time range based on the training spikes.
    spike_start = spike_times[0]
    spike_end = spike_times[-1]
    
    # Crop events to fall within the spike training period.
    left_evt = int(cp.asnumpy(cp.where(event_times > spike_start)[0][0]))
    right_evt = int(cp.asnumpy(cp.where(event_times < spike_end)[0][-1]))
    event_times = event_times[left_evt: right_evt]
    event_labels = event_labels[left_evt: right_evt]
    
    # Binning parameters: 100ms window and 10ms step (converted to samples at 30kHz).
    bin_size = 100 * 30  # 100ms window (samples)
    bin_inc = 10 * 30    # 10ms step (samples)
    
    # Determine overall time range.
    base_time = cp.minimum(event_times.min(), spike_times.min())
    max_time = cp.maximum(event_times.max(), spike_times.max())
    num_bins = int(cp.floor((max_time - base_time - bin_size) / bin_inc).get()) + 1
    
    # Get the set of unique templates.
    templates = cp.unique(spike_templates)
    num_templates = int(templates.size)
    
    # Pre-compute bin boundaries.
    bin_left_edges = base_time + cp.arange(num_bins) * bin_inc
    bin_right_edges = bin_left_edges + bin_size
    
    # Create a 2D array for binned spike counts.
    binned_counts = cp.zeros((num_bins, num_templates), dtype=cp.int32)
    for i, tmpl in enumerate(cp.asnumpy(templates)):
        tmpl_spikes = spike_times[spike_templates == tmpl]
        start_indices = cp.searchsorted(tmpl_spikes, bin_left_edges, side='left')
        end_indices = cp.searchsorted(tmpl_spikes, bin_right_edges, side='left')
        binned_counts[:, i] = end_indices - start_indices

    # Compute event bin indices.
    # The lookahead window starts at event_time + lookahead_start*30 samples.
    event_bin_indices = ((event_times + lookahead_start * 30 - base_time) // bin_inc).astype(cp.int32)
    valid = event_bin_indices < num_bins
    if not bool(cp.all(valid)):
        print("Warning: Some event times fall outside the binned range and will be ignored.")
        event_bin_indices = event_bin_indices[valid]
        event_labels = event_labels[valid]
    
    # Select features for the corresponding event bins.
    X = binned_counts[event_bin_indices]
    y = event_labels
    
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


# -------------------------------
# Main code to average over multiple eventfiles
# -------------------------------

# Define directories.
base_dir = Path('C:/', 'SGL_DATA', 'joplin_20240222')
decoder_dir = base_dir / "decoder_input"
ks_dir = base_dir / "kilosort4"
oss_dir = base_dir / "cuda_output"

# List of eventfiles.
eventfile_names = ["eventfile_15.txt", "eventfile_26.txt", "eventfile_37.txt", "eventfile_48.txt"]
eventfile_paths = [decoder_dir / name for name in eventfile_names]

# Training parameters.
training_start = 26984448
training_end = 83984384

print(f"Plotting LOO accuracy using spikes from {decoder_dir / 'spikeOutput.txt'} and {ks_dir} using eventfiles {eventfile_names}, training on samples {training_start} to {training_end}.")
print("Loading spike data...")

# Load Kilosort spikes.
ks_spike_times = np.load(ks_dir / "spike_times.npy")
ks_spike_templates = np.load(ks_dir / "spike_templates.npy")
ks_spike_amplitudes = np.load(ks_dir / "amplitudes.npy")

# Load OSS spikes.
oss_spike_times = []
oss_spike_templates = []
oss_spike_amplitudes = []
with open(oss_dir / 'spikeOutput_train_test.txt', 'r') as f:
    for line in f:
        tokens = line.split(',')
        oss_spike_times.append(int(tokens[0]))
        oss_spike_templates.append(int(tokens[1]))
        oss_spike_amplitudes.append(float(tokens[2]))
oss_spike_times = np.array(oss_spike_times)
oss_spike_templates = np.array(oss_spike_templates)
oss_spike_amplitudes = np.array(oss_spike_amplitudes)

print("Spike data loaded successfully.")

# Function to load event data from a file.
def load_event_data(event_file_path):
    event_times = []
    event_labels = []
    with open(event_file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            event_times.append(float(tokens[0]))
            event_labels.append(int(tokens[1]))
    return np.array(event_times), np.array(event_labels)

# Define lookahead start times in milliseconds (-600ms to 1400ms, 10ms increments).
lookahead_starts = np.arange(-200, 501, 10)
lookahead_ends = lookahead_starts + 100
loo_ks_avg = []   # Averaged LOO accuracy for Kilosort.
loo_oss_avg = []  # Averaged LOO accuracy for OSS.

# Loop over each lookahead start time.
for la in lookahead_starts:
    ks_acc_list = []
    oss_acc_list = []
    
    print(f"Computing LOO for lookahead = {la} ms over {len(eventfile_paths)} eventfiles.")
    
    # Loop over each event file.
    for event_file in eventfile_paths:
        event_times, event_labels = load_event_data(event_file)
        
        acc_ks = LOO_accuracy(ks_spike_times, ks_spike_templates, ks_spike_amplitudes,
                              event_times, event_labels, training_start, training_end, la)
        acc_oss = LOO_accuracy(oss_spike_times, oss_spike_templates, oss_spike_amplitudes,
                               event_times, event_labels, training_start, training_end, la)
        ks_acc_list.append(acc_ks)
        oss_acc_list.append(acc_oss)
    
    # Average over eventfiles.
    loo_ks_avg.append(np.mean(ks_acc_list))
    loo_oss_avg.append(np.mean(oss_acc_list))

# Plot the averaged LOO accuracy over lookahead start times for both methods.
np.save(f"{base_dir / 'loo_ks_avg.npy'}", loo_ks_avg)
np.save(f"{base_dir / 'loo_oss_avg.npy'}", loo_oss_avg)

plt.figure(figsize=(10, 6))
plt.plot(lookahead_ends, loo_ks_avg, label='Kilosort', marker='o')
plt.plot(lookahead_ends, loo_oss_avg, label='LSS', marker='s')
plt.xlabel("Lookahead Start (ms)")
plt.ylabel("LOO Accuracy (%)")
plt.title("Averaged LOO Accuracy vs. Lookahead Start Time (over 4 eventfiles)")
plt.legend()
plt.grid(True)
plt.show()
