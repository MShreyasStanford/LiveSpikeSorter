from pathlib import Path
from tqdm import tqdm

base_dir = Path("C:/", "SGL_DATA", "joplin_20240208", "cuda_output")
spike_file = base_dir / 'spikeOutput.txt'

with open(spike_file, 'r') as f:
	spikes = [ (int(line.split(',')[0]), int(line.split(',')[1]), float(line.split(',')[2])) for line in f ]
	T = max([ spike[1] for spike in spikes ])
	last_spike_time = [0] * (T + 1)
	num_duplicates = 0

	for i in tqdm(range(len(spikes))):
		spike = spikes[i]
		spike_time = spike[0]
		spike_template = spike[1]

		if spike_time - last_spike_time[spike_template] < 7:
			num_duplicates += 1
		else:
			last_spike_time[spike_template] = spike_time

	print(f"Number of duplicate spikes in {spike_file}: {num_duplicates}.")