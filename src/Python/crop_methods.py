import numpy as np
import os
import statistics
from pathlib import Path
import heapq
import csv

def support(template):
    return np.where(np.any(template != 0, axis=0))[0]

def subset_of(A, B): 
    for x in A:
        if x not in B:
            return False
    return True

def k_lowest_std_channels(templates, C, filename, k, contig_channels_left, contig_channels_right):
    T = templates.shape[0]
    N = templates.shape[2]
    W = templates.shape[1]

    stds = {}
    
    with open(filename, 'rb') as fidInput:
        entries_to_read = 50 * C * W
        y = np.fromfile(fidInput, dtype=np.int16, offset=0, count=entries_to_read)

        if y.size == 0 or y.size != entries_to_read:
            print(f"Error: read {y.size} entries from {filename}, expected {entries_to_read}")
            exit(1)


        float_vec = np.vectorize(lambda x : float(x))
        
        for chan_index in range(contig_channels_left, contig_channels_right):
            stds[chan_index] = statistics.stdev(float_vec(y[chan_index : len(y) : C]))

    supports = [support(template) for template in templates]
        
    min_index = contig_channels_left
    minimum = sum([stds[chan_index] for chan_index in range(contig_channels_left, contig_channels_left + k)]) / k
    for chan_index in range(contig_channels_left + 1, contig_channels_right - k):
        avg_std = sum([stds[i] for i in range(chan_index, chan_index + k)]) / k
        if avg_std < minimum:
            minimum = avg_std
            min_index = chan_index

    template_mask = []
    for i, supp in enumerate(supports):
        if subset_of(supp, range(min_index, min_index + k)):
            template_mask.append(i)
            
    return template_mask, range(min_index, min_index + k)


def k_largest_indices(lst, k):
    return [index for value, index in heapq.nlargest(k, [(value, index) for index, value in enumerate(lst)])]

def k_most_active_around_event(spike_times, spike_templates, templates, k, event_time, contig_channels_left, contig_channels_right):
    T = templates.shape[0]
    W = templates.shape[1]
    N = templates.shape[2]

    num_spikes_prior = [0 for _ in range(T)]
    num_spikes_post = [0 for _ in range(T)]

    for i, spike_time in enumerate(spike_times):
        spike_template = spike_templates[i]

        if -100 * 30 <= spike_time - event_time <= 0:
            num_spikes_prior[spike_template] += 1

        if 100 * 30 <= spike_time - event_time <= 200 * 30:
            num_spikes_post[spike_template] += 1

    print(num_spikes_prior)
    print(num_spikes_post)
    num_spikes_ratio = [num_spikes_post[i] / num_spikes_prior[i] if num_spikes_prior[i] != 0 else 0 for i in range(T)]
    print(num_spikes_ratio)
    print(k_largest_indices(num_spikes_ratio, 10))
            
      
# Selects the best size k window of channel indices such that the
# templates supported on these indices have the most activity
def k_most_active_channels(spike_times, spike_templates, templates, k, contig_channels_left, contig_channels_right):
    T = templates.shape[0] # number of templates
    N = templates.shape[2] # number of channels
    num_spikes = [0 for _ in range(T)]

    # Count number of spikes for each template
    for t in range(spike_times.size):
        num_spikes[spike_templates[t]] += 1

    # Compute the support of each template
    supports = []
    for template in templates:
        supports.append(support(template))

    # Sliding window across CHANNELS for maximum activity
    maxim_index = 0 
    maxim = sum([num_spikes[i] for i in range(T) if subset_of(supports[i], range(contig_channels_left, contig_channels_left + k))])
    for i in range(contig_channels_left + 1, contig_channels_right - k): 
        spike_total = sum([num_spikes[j] for j in range(T) if subset_of(supports[j], range(i, k + i))])
        if spike_total > maxim:
            maxim_index = i 
            maxim = spike_total

    template_mask = []
    for i, supp in enumerate(supports):
        if subset_of(supp, range(maxim_index, maxim_index + k)):
            template_mask.append(i)
    
    return template_mask, range(maxim_index, maxim_index + k)

def k_most_active_channels_parallelized(spike_times, spike_templates, templates, k):
    T = templates.shape[0]
    N = templates.shape[2]
    k = k // 2
    num_spikes = [0 for _ in range(T)]

    for t in range(spike_times.size):
        num_spikes[spike_templates[t]] += 1

    supports = []
    for template in templates:
        supports.append(support(template))

    maxim_index_1 = 0
    maxim_index_2 = k
    maxim = sum([num_spikes[i] for i in range(T) if subset_of(supports[i], range(0, k))]) + \
        sum([num_spikes[i] for i in range(T) if subset_of(supports[i], range(k, 2 * k))])

    spike_count_in_channel_segment = {}
    for left in range(0, N - k):
        spike_count_in_channel_segment[(left, left + k)] = sum([num_spikes[i] for i in range(T) if subset_of(supports[i], range(left, left + k))])
        
    for i in range(1, N - 2 * k):
        for j in range(i + k, N - k):
            spike_total = spike_count_in_channel_segment[(i, i + k)] + spike_count_in_channel_segment[(j, j + k)]
            
            if spike_total > maxim:
                maxim_index_1 = i
                maxim_index_2 = j
                maxim = spike_total

    template_mask_1 = []
    template_mask_2 = []
    for i, supp in enumerate(supports):
        if subset_of(supp, range(maxim_index_1, maxim_index_1 + k)):
            template_mask_1.append(i)
    for i, supp in enumerate(supports):
        if subset_of(supp, range(maxim_index_2, maxim_index_2 + k)):
            template_mask_2.append(i)

    return template_mask_1, template_mask_2, range(maxim_index_1, maxim_index_1 + k), range(maxim_index_2, maxim_index_2 + k)

def crop_kilosort_output(templates, whitening_mat, channel_map, channel_mask, template_mask, out_path):
    k = len(channel_mask)
    # Crop data to these indices
    deep_copy_templates = np.array(templates[:, :, channel_mask])
    deep_copy_templates = deep_copy_templates[template_mask, :, :]
    deep_copy_whitening_mat = np.array(whitening_mat[np.ix_(channel_mask, channel_mask)])
    deep_copy_channel_map = np.array(channel_map[channel_mask])

    
    if not os.path.exists(out_path):
        print(f"Creating directory {str(out_path)}")
        os.makedirs(str(out_path))
    
    np.save(out_path / f"templates.npy", deep_copy_templates)
    np.save(out_path / f"whiteningMat.npy", deep_copy_whitening_mat)
    np.save(out_path / f"channelMap.npy", deep_copy_channel_map)
    np.save(out_path / f"channelMask.npy", channel_mask)
    np.save(out_path / f"templateMap.npy", template_mask)

def parse_bin_meta_file(filename):
    metadata = { }
    with open(filename, 'r') as bin_meta_input:
        n_channels = -1
        for line in bin_meta_input:
            delimited = line.split('=')

            if len(delimited) != 2:
                continue

            key = delimited[0]
            value = delimited[1]

            if key == "nSavedChans":
                metadata["nSavedChans"] = int(value)

    if "nSavedChans" not in metadata:
        print("Error occurred while parsing binary metadata file for nSavedChans.")

    return metadata

def load_cluster_ks_labels(filename):
    with open(filename) as cluster_KSLabel_file:
        rd = csv.reader(cluster_KSLabel_file, delimiter='\t', quotechar='"')
        cluster_kslabels = { int(cluster_id): ks_label for cluster_id, ks_label in rd if cluster_id != "cluster_id" }
        return cluster_kslabels

def filter_no_mua(template_map, cluster_kslabels):
    return [ template for template in template_map if cluster_kslabels[template] == 'good' ]

def filter_high_activity(template_map, spike_templates, spike_times, low_hz):
    spike_count = { template: 0 for template in template_map }

    for spike_template in spike_templates:
        if spike_template not in template_map:
            continue

        spike_count[spike_template] += 1

    recording_length_s = max(spike_times) / 30000
    print(spike_count)
    print(recording_length_s)

    return [ template for template in template_map if spike_count[template] / recording_length_s >= low_hz ]