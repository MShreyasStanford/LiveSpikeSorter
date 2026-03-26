from pathlib import Path
import numpy as np

KS_OUTPUT_DIR = Path('C://', 'SGL_DATA', 'joplin_20240222', 'kilosort4_train') 
#KS_OUTPUT_DIR = Path("C:/", "Users", "Spike Sorter", "source", "repos", "OnlineSpikes_v2", "src", "kilosort4_stabletrain")
DECODER_INPUT_DIR = Path('C://', 'SGL_DATA', 'joplin_20240222', 'decoder_input')
sample_offset = 0
start = 0
#end = 26984448
end = 1231233131231


spike_times = np.load(KS_OUTPUT_DIR / "spike_times.npy")
spike_templates = np.load(KS_OUTPUT_DIR / "spike_templates.npy")
amplitudes = np.load(KS_OUTPUT_DIR / "amplitudes.npy")
spike_positions = np.load(KS_OUTPUT_DIR / "spike_positions.npy")
spike_ys = spike_positions[:, 1]

spike_output_location = DECODER_INPUT_DIR / "ksSpikeOutput.txt"

with open(spike_output_location, 'w') as f:
	for i in range(len(spike_times)):
		if start <= spike_times[i] <= end:
			f.write(f"{spike_times[i]},{spike_templates[i]},{amplitudes[i]}\n")
