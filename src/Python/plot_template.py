from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random

base_dir = Path("C:/", "SGL_DATA", "joplin_20240222")
ks_dir = base_dir / "kilosort4"

templates = np.load(ks_dir / 'templates.npy')
while True:
	template_index = random.randint(0, templates.shape[0])
	num_chans = templates.shape[2]
	support = []    # List of channel indices that are not all zeros

	for j in range(num_chans):
	    channel_waveform = templates[template_index, :, j]

	    # Check whether this channel has any non-zero elements
	    if np.any(channel_waveform != 0):
	        support.append(j)

	template_on_support = templates[template_index , : , min(support) : max(support)]
	print(template_on_support.shape)
	for i in range(template_on_support.shape[1]):
		plt.plot(template_on_support[:, i] * 5+ 10 * i)
	plt.show()