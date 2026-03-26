from pathlib import Path

#spikes_dir = Path('C:/', 'SGL_DATA', '05_31', 'decoder_input')
spikes_dir = Path('C:/', 'SGL_DATA', 'joplin_20240222', 'cuda_output')
spikes_file = 'spikeOutput_train.txt'
offset_spike_file = 'spikeOutputOffset.txt'

#offset_amount = 29999104
#offset_amount = 44990464
#offset_amount = 59998208
#offset_amount = 44990464 -1 * 256 * 64\
#offset_amount = 14385152 - 256 * 64
offset_amount = 26984448 - 256 * 64
delim = ','

print(f"Offsetting spike times {spikes_dir / spikes_file} -> {offset_spike_file} by {offset_amount}.")
with open(spikes_dir / spikes_file, 'r') as f:
	spikes = [ (int(line.split(delim)[0]), int(line.split(delim)[1]), float(line.split(delim)[2]), float(line.split(delim)[3])) for line in f ]

with open(spikes_dir / offset_spike_file, 'w') as f:
	for spike in spikes:
		f.write(f"{spike[0] + offset_amount},{spike[1]},{spike[2]},{spike[3]}\n")
