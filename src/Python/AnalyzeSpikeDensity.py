from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

USE_KS_CENTROIDS = False
# --- Read the spike file as before ---
base_dir = Path("C:/", "SGL_DATA", "joplin_20240208", "cuda_output")
spike_file = base_dir / 'spikeOutput.txt'

# --- Read in Kilosort training file if we're analyzing OSS output
kilo_dir = Path("C:/", "SGL_DATA", "joplin_20240208", 'kilosort4')
cluster_centroids = np.load(kilo_dir / 'cluster_centroids.npy', allow_pickle=True).item()

print(f"Analyzing spikes from {spike_file}.")
with open(spike_file, 'r') as f:
    spikes = [ (int(line.split(',')[0]), int(line.split(',')[1]), float(line.split(',')[2]), float(line.split(',')[3])) 
               for line in f ]
    
# Extract columns and convert to numpy arrays
spike_times = np.array([spike[0] for spike in spikes])
spike_templates = np.array([spike[1] for spike in spikes])

if USE_KS_CENTROIDS:
	spike_ys = np.array([cluster_centroids[spike[1]][1] for spike in spikes])
else:
	spike_ys = np.array([spike[3] for spike in spikes])
# spike_amplitudes is available if needed
# spike_amplitudes = np.array([spike[2] for spike in spikes])

min_time = spike_times.min()
max_time = spike_times.max()
num_templates = spike_templates.max() + 1

total_time_s = (max_time - min_time) / 30000  # convert samples to seconds
print(f"Spikes per second of {spike_file}: {len(spikes) / total_time_s}")

# --- Define parameters for moving window ---
window_size = 30000 * 60  # 1 minute window (in samples)
step_size = 30000         # update every second (30000 samples)

# Create time bins (window starting positions) from the start until there's room for a full window
time_bins = np.arange(min_time, max_time - window_size, step_size)

# --- Compute moving firing rate per template ---
plt.figure(figsize=(12, 8))
unique_templates = np.unique(spike_templates)
unique_templates = unique_templates[-25:]
for template in unique_templates:
    # Get spike times for this template and sort them
    template_times = np.sort(spike_times[spike_templates == template])
    
    rates = []      # to store firing rate for each window
    t_centers = []  # to store center time (in samples) of each window
    
    # Slide the window over the recording
    for t in time_bins:
        # Count spikes in the window [t, t+window_size) using searchsorted (efficient for sorted arrays)
        left_index = np.searchsorted(template_times, t, side='left')
        right_index = np.searchsorted(template_times, t + window_size, side='left')
        count = right_index - left_index
        
        rate = count / 60.0  # Convert count in a minute to Hz (spikes per second)
        rates.append(rate)
        t_centers.append(t + window_size/2)  # use the window center as the time point

    # Convert sample indices to seconds for plotting
    t_centers_sec = np.array(t_centers) / 30000
    plt.plot(t_centers_sec, rates, label=f"Template {template}")

plt.xlabel("Time (s)")
plt.ylabel("Firing rate (Hz)")
plt.title("Moving Firing Rate (1-Minute Window) per Neuron")
plt.legend()
plt.show()

# --- Compute average y-position of spikes over all templates ---
# For efficiency, sort the overall spike times (and corresponding y-values)
sorted_indices = np.argsort(spike_times)
sorted_times = spike_times[sorted_indices]
sorted_ys = spike_ys[sorted_indices]

avg_y = []      # to store average y for each window
t_centers_all = []  # to store center time of each window

for t in time_bins:
    left_index = np.searchsorted(sorted_times, t, side='left')
    right_index = np.searchsorted(sorted_times, t + window_size, side='left')
    
    # Compute the average only if there is at least one spike in the window
    if right_index > left_index:
        avg = np.mean(sorted_ys[left_index:right_index])
    else:
        avg = np.nan  # or you could choose to skip this window
    avg_y.append(avg)
    t_centers_all.append(t + window_size/2)

# Convert sample indices to seconds for plotting
t_centers_all_sec = np.array(t_centers_all) / 30000

plt.figure(figsize=(10, 6))
plt.plot(t_centers_all_sec, avg_y, label="Average Y-position")
plt.xlabel("Time (s)")
plt.ylabel("Average Y-Position")
plt.title("Average Y-position of Spiking Activity Over Time")
plt.legend()
plt.show()

# --- Compute moving average y-position per neuron (template) ---
plt.figure(figsize=(12, 8))
unique_templates = np.unique(spike_templates)

for template in unique_templates:
    # Get spike times and corresponding y-values for this template
    template_mask = (spike_templates == template)
    template_times = spike_times[template_mask]
    template_ys = spike_ys[template_mask]
    
    # Sort by spike time (if not already sorted)
    sorted_indices = np.argsort(template_times)
    template_times_sorted = template_times[sorted_indices]
    template_ys_sorted = template_ys[sorted_indices]
    
    avg_y = []      # to store average y for each window
    t_centers = []  # to store the center time (in samples) of each window
    
    # Slide the window over the recording
    for t in time_bins:
        left_index = np.searchsorted(template_times_sorted, t, side='left')
        right_index = np.searchsorted(template_times_sorted, t + window_size, side='left')
        if right_index > left_index:
            avg_val = np.mean(template_ys_sorted[left_index:right_index])
        else:
            avg_val = np.nan  # No spikes in this window
        avg_y.append(avg_val)
        t_centers.append(t + window_size/2)
        
    # Convert time centers from samples to seconds for plotting
    t_centers_sec = np.array(t_centers) / 30000.0
    plt.plot(t_centers_sec, avg_y, label=f"Template {template}")

plt.xlabel("Time (s)")
plt.ylabel("Average Y Position")
plt.title("Moving Average Y Position per Neuron (1-Minute Window)")
plt.legend()
plt.show()
