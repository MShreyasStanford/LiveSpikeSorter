from pathlib import Path
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import tqdm

__analysis_metadata__ = {
    "name": "Plot Template",
    "description": "Given a Kilosort directory, plot a template waveform.",
    "parameters": [
        ("Kilosort Directory", "DirPath"),
        ("Template Index", "int")
    ]
}

def run(base_dir, kilosort_dir, template_ind):
    templates = np.load(kilosort_dir / 'templates.npy')
    if template_ind >= templates.shape[0]:
        print(f"Template {template_ind} out of range.", file=stderr)
        return

    template = templates[template_ind].T
    channel_masses = np.sum(template, axis=1)
    max_chan = np.argmax(channel_masses)
    plt.plot(template[max_chan], linewidth=8)
    plt.axis('off')
    plt.savefig(f'transparent_template_plot_{template_ind}.pdf', transparent=True)
    plt.clf()