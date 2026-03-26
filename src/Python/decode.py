from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score

def plot_psth(binned_counts, base_time, bin_inc, bin_size, event_times):
    psth_x = np.array(range(-200, 400 + bin_inc // 30, bin_inc // 30))
    psth_y = np.zeros(psth_x.shape)
    for event in event_times:
        left = int((event - 300 * 30 - base_time) / bin_inc)

        if left < 0:
            continue

        right = left + 10
        count = sum(sum(binned_counts[left: right]))
        psth_y[0] += count

        left += 1
        right += 1
        i = 1
        while right * bin_inc + base_time < event + 400 * 30:
            count += sum(binned_counts[right, :])
            count -= sum(binned_counts[left, :])
            psth_y[i] += count
            i += 1
            left += 1
            right += 1

    plt.plot(psth_x, psth_y)
    plt.show()


# -------------------------------
# Main decoder code
# -------------------------------
is_kilo_spikes = True

# For training: spike and event files are in the decoder directory
base_dir = Path('C:/', 'SGL_DATA', 'joplin_20240222')
decoder_dir = Path(base_dir, "decoder_input")
ks_dir = Path(base_dir, "kilosort4")
oss_dir = Path(base_dir, "cuda_output")
event_file = decoder_dir / 'eventfile_15.txt'

if is_kilo_spikes:
    print(f"Mapping {ks_dir} spikes, {event_file} events -> {decoder_dir / 'ks_predictions.txt'}.")
    spike_times = np.load(ks_dir / "spike_times.npy")
    spike_templates = np.load(ks_dir / "spike_templates.npy")
    spike_amplitudes = np.load(ks_dir / "amplitudes.npy")
else:
    print(f"Mapping {oss_dir / 'spikeOutputGood.txt'}, {event_file} events -> {decoder_dir / 'oss_predictions.txt'}.")
    with open(oss_dir / 'spikeOutput.txt', 'r') as f:
        spike_times = []
        spike_templates = []
        spike_amplitudes = []

        for line in f:
            tokens = line.split(',')
            spike_times.append(int(tokens[0]))
            spike_templates.append(int(tokens[1]))
            spike_amplitudes.append(float(tokens[2]))

        spike_times = np.array(spike_times)
        spike_amplitudes = np.array(spike_amplitudes)
        spike_templates = np.array(spike_templates)

# Parameters for what to train on
training_start = 26984448
training_end = 83984384

print(f"Training from sample {training_start} to {training_end}.")
left_index = np.where(spike_times > training_start)[0][0]
right_index = np.where(spike_times < training_end)[0][-1]
spike_times = spike_times[left_index : right_index]
spike_templates = spike_templates[left_index : right_index]
spike_amplitudes = spike_amplitudes[left_index : right_index]

spike_start = spike_times[0]
spike_end = spike_times[-1]
print(f"Loaded spikes from {spike_start} to {spike_end}.")
bin_size = 100 * 30  # window size (e.g., 1500 samples)
bin_inc = 10 * 30   # step size (e.g., 300 samples)

print("Loading event times.")
event_times = []
event_labels = []
with open(event_file, 'r') as f:
    for line in f:
        tokens = line.strip().split()
        event_times.append(float(tokens[0]))
        event_labels.append(int(tokens[1]))
event_times = np.array(event_times)
event_labels = np.array(event_labels)

# Crop events to training period
left_index = np.where(event_times > spike_start)[0][0]
right_index = np.where(event_times < spike_end)[0][-1]
event_times = event_times[left_index : right_index]
event_labels = event_labels[left_index : right_index]

print(f"Training on {len(event_times)} events from {event_times[0]} to {event_times[-1]}.")
base_time = min(event_times.min(), spike_times.min())
max_time = max(event_times.max(), spike_times.max())
num_bins = int(np.floor((max_time - base_time - bin_size) / bin_inc)) + 1

# Get the set of templates from the training spike data (assumes integer codes)
templates = np.unique(spike_templates)
num_templates = len(templates)

print(f"Binning training data from {min(event_times)} to {max(event_times)}.")

# Pre-compute bin boundaries for training data
bin_left_edges = base_time + np.arange(num_bins) * bin_inc
bin_right_edges = bin_left_edges + bin_size

# Create a 2D array to hold binned spike counts (rows: bins, columns: templates)
binned_counts = np.zeros((num_bins, num_templates), dtype=int)
for col, tmpl in enumerate(templates):
    tmpl_spikes = spike_times[spike_templates == tmpl]
    start_indices = np.searchsorted(tmpl_spikes, bin_left_edges, side='left')
    end_indices = np.searchsorted(tmpl_spikes, bin_right_edges, side='left')
    binned_counts[:, col] = end_indices - start_indices

event_bin_indices = ((event_times + 150*30 - base_time) // bin_inc).astype(int)
valid = event_bin_indices < num_bins
if not np.all(valid):
    print("Warning: Some event times fall outside the binned range and will be ignored.")
event_bin_indices = event_bin_indices[valid]
event_times = event_times[valid]
event_labels = event_labels[valid]

# -------------------------------
# Compute normalized training features
# -------------------------------
print("Training...")

X = binned_counts[event_bin_indices]
y = event_labels

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Compute Leave-One-Out Cross-Validation Accuracy on normalized data
loo = LeaveOneOut()
loo_scores = cross_val_score(LogisticRegression(), X_norm, y, cv=loo)
print("Leave-One-Out Accuracy (normalized): {:.2f}%".format(np.mean(loo_scores) * 100))

# Train final model on all training data for decoding
model = LogisticRegression()
model.fit(X_norm, y)
print("Final model trained on all training data.")
print("Model intercept:")
print(model.intercept_)

# -------------------------------
# Processing test spikes
# -------------------------------
print("Processing test spikes.")
dec_spike_times = np.load(ks_dir / 'spike_times.npy')
dec_spike_templates = np.load(ks_dir / 'spike_templates.npy')
dec_spike_amplitudes = np.load(ks_dir / 'amplitudes.npy')

left_index = np.where(dec_spike_times > training_end)[0][0]
dec_spike_times = dec_spike_times[left_index : ]
dec_spike_templates = dec_spike_templates[left_index : ]
dec_spike_amplitudes = dec_spike_amplitudes[left_index : ]

print(f"Testing with spikes from {min(dec_spike_times)} to {max(dec_spike_times)}.")
base_time_dec = dec_spike_times.min()
max_time_dec = dec_spike_times.max()
num_bins_dec = int(np.floor((max_time_dec - base_time_dec - bin_size) / bin_inc)) + 1

bin_left_edges_dec = base_time_dec + np.arange(num_bins_dec) * bin_inc
bin_right_edges_dec = bin_left_edges_dec + bin_size

binned_counts_dec = np.zeros((num_bins_dec, num_templates), dtype=int)
for col, tmpl in enumerate(templates):
    tmpl_spikes = dec_spike_times[dec_spike_templates == tmpl]
    tmpl_spikes.sort()
    start_indices_dec = np.searchsorted(tmpl_spikes, bin_left_edges_dec, side='left')
    end_indices_dec = np.searchsorted(tmpl_spikes, bin_right_edges_dec, side='left')
    binned_counts_dec[:, col] = end_indices_dec - start_indices_dec

X_dec = binned_counts_dec
X_dec_norm = scaler.transform(X_dec)
dec_predictions = model.predict(X_dec_norm)

if is_kilo_spikes:
    prediction_file = decoder_dir / "ks_predictions.txt"
else:
    prediction_file = decoder_dir / "oss_predictions.txt"

with open(prediction_file, 'w') as f:
    for i in range(num_bins_dec):
        # Use the RIGHT edge of the bin as the sample time.
        bin_time = bin_right_edges_dec[i]
        label = dec_predictions[i]
        f.write(f"{bin_time} {label}\n")

print(f"Decoder predictions written to {prediction_file}")
