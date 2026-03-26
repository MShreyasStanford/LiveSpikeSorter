from pathlib import Path
import numpy as np
from collections import defaultdict

def best_map_error_fraction(spike_detection_templates, spike_templates):
    """
    Computes the fraction of entries misclassified by the best possible
    deterministic map from spike_detection_templates to spike_templates.
    
    Parameters:
    - spike_detection_templates: list of integers (values in {0, 1, ..., N})
    - spike_templates: list of integers (values in {0, 1, ..., M}, with M < N)
    
    Returns:
    - error_fraction: float, fraction of entries that the best map gets wrong.
                      It ranges from 0 (perfect mapping) to 1.
    """
    if len(spike_detection_templates) != len(spike_templates):
        raise ValueError("Both lists must have the same length.")
    
    # Count occurrences: for each detection template, count how often it maps to each spike_template
    joint_counts = defaultdict(lambda: defaultdict(int))
    for det, clust in zip(spike_detection_templates, spike_templates):
        joint_counts[det][clust] += 1
    
    total_correct = 0
    total = len(spike_detection_templates)
    
    # For each detection template, add the highest count (best possible prediction)
    for det, counts in joint_counts.items():
        total_correct += max(counts.values())
    
    accuracy = total_correct / total
    error_fraction = 1 - accuracy
    
    return error_fraction

def entropic_complexity(spike_detection_templates, spike_templates):
    """
    Computes the complexity of the clustering by calculating the average 
    conditional entropy H(spike_templates | spike_detection_templates).

    Parameters:
    - spike_detection_templates: list of integers (values in {0, 1, ..., N})
    - spike_templates: list of integers (values in {0, 1, ..., M}, with M < N)

    Returns:
    - complexity: float, the average conditional entropy in bits.
    """
    # Check that both lists have the same length
    if len(spike_detection_templates) != len(spike_templates):
        raise ValueError("Lists must have the same length.")
    
    # Count the joint occurrences: for each detection template, how often does it map to each spike_template?
    joint_counts = defaultdict(lambda: defaultdict(int))
    detection_counts = defaultdict(int)
    total = len(spike_detection_templates)
    
    for det, clust in zip(spike_detection_templates, spike_templates):
        joint_counts[det][clust] += 1
        detection_counts[det] += 1
    
    # Calculate weighted conditional entropy
    total_entropy = 0.0
    for det, clust_dict in joint_counts.items():
        # Frequency of this detection template
        p_det = detection_counts[det] / total
        # Create probability distribution for spike_templates given this detection template
        counts = np.array(list(clust_dict.values()), dtype=float)
        probs = counts / counts.sum()
        # Compute entropy for this detection template
        entropy = -np.sum(probs * np.log2(probs))
        total_entropy += p_det * entropy
    
    return total_entropy

kilosort_dir = Path('C:/', 'Users', 'Spike Sorter', 'source', 'repos', 'OnlineSpikes_v2', 'src', 'kilosort4')
training_dir = Path('C:/', 'Users', 'Spike Sorter', 'source', 'repos', 'OnlineSpikes_v2', 'src', 'kilosort4_stabletrain')
#kilosort_dir = Path('C:/', 'SGL_DATA', 'joplin_20240208', 'kilosort4')
#training_dir = Path('C:/', 'SGL_DATA', 'joplin_20240208', 'kilosort4_1000_1500')

spike_detection_templates = np.load(kilosort_dir / 'spike_detection_templates.npy')
spike_templates = np.load(kilosort_dir / 'spike_templates.npy')
training_spike_detection_templates = np.load(training_dir / 'spike_detection_templates.npy')
training_spike_templates = np.load(training_dir / 'spike_templates.npy')

assert len(spike_detection_templates) == len(spike_templates)
print(f"KS clustering went from {max(spike_detection_templates) + 1} templates to {max(spike_templates) + 1} templates.")
print(f"Training has {max(training_spike_detection_templates) + 1} unclustered templates, and {max(training_spike_templates) + 1} templates.")

complexity = best_map_error_fraction(spike_detection_templates, spike_templates) # a complexity of 0 means the map is deterministic

print(f"Clustering complexity of {kilosort_dir}: {complexity}.")