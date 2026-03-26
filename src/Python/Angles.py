from scipy.io import loadmat
import numpy as np
from pathlib import Path

base_dir = Path('C:/', 'SGL_DATA', 'joplin_20240222')
bhv_file = base_dir / 'bhv.mat'
eventfile = Path(base_dir, "decoder_input") / 'eventfile.txt'
output = Path(base_dir, 'decoder_input') / 'eventfile_labeled.txt'

print(f"Mapping {eventfile} x {bhv_file} -> {output}.")
with open(eventfile, 'r') as f:
	events = [int(event.split(' ')[0]) for event in f]

bhv_data = loadmat(bhv_file)
bhv = bhv_data['bhv']
trials = bhv['Trials'][0, 0]
angles = np.array([trial['angleID'][0, 0] for trial in trials.flatten()])
stop_conditions = np.array([trial['stopCondition'][0, 0] for trial in trials.flatten()])
# Get indices for successful trials (where stopCondition equals 1)
success_indices = np.where(stop_conditions == 1)[0]
angles = angles[success_indices]

assert len(angles) == len(events)

with open(output, 'w') as f:
	for i in range(len(events)):
		f.write(f"{events[i]} {angles[i]}\n")

print(f"Wrote complete eventfile to {output}.")

# For each pair (i, i+4), generate a submatrix eventfile with relabeling (i -> 0, i+4 -> 1)
# Here, we assume the unique labels are {1,2,...,8}
for i in range(1, 5):
    lower_label = i
    higher_label = i + 4
    # Create a boolean mask to select rows with the labels in the current pair
    mask = np.isin(angles, [lower_label, higher_label])
    sub_events = np.array(events)[mask]
    sub_labels = angles[mask]
    
    # Remap labels: lower_label becomes 0 and higher_label becomes 1
    remapped_labels = np.where(sub_labels == lower_label, 0, 1)
    
    # Define the output file for this pair
    sub_output = base_dir / 'decoder_input' / f"eventfile_{lower_label}{higher_label}.txt"
    with open(sub_output, 'w') as f:
        for ts, new_label in zip(sub_events, remapped_labels):
            f.write(f"{ts} {new_label}\n")
    
    print(f"Wrote submatrix eventfile for labels {lower_label} and {higher_label} to {sub_output}.")