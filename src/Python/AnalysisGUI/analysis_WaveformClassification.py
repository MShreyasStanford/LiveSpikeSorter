import numpy as np
import csv
from pathlib import Path

__analysis_metadata__ = {
    "name": "Waveform Classification",
    "description": "Classify templates into peak-trough or trough-peak (short/long) categories using fixed sampling rate and threshold.",
    "parameters": [
        ("Templates File (.npy)", "FilePath")
    ]
}

def run(base_path, templates_file):
    """
    Loads templates from templates_file (n_templates, n_samples, n_channels),
    classifies each into:
      - 'peak-trough' if the global max precedes the global min,
      - 'trough-peak-short' if min precedes max within 200 μs,
      - 'trough-peak-long' otherwise.
    Saves results as base_path / 'waveform_classification.csv' and returns the list of categories.
    """
    templates_path = Path(base_path) / templates_file
    templates = np.load(templates_path)
    if templates.ndim != 3:
        raise ValueError(f"Expected templates ndarray of shape (T, N, C), got {templates.shape}")

    num_templates, num_samples, num_channels = templates.shape
    fs = 30000  # Hz
    threshold_us = 200.0
    threshold_samples = int(np.round(threshold_us * fs / 1e6))

    categories = []
    for idx in range(num_templates):
        wave = templates[idx]  # shape (num_samples, num_channels)
        ptp_by_ch = wave.ptp(axis=0)
        primary_ch = int(np.argmax(ptp_by_ch))
        trace = wave[:, primary_ch]

        peak_idx = int(np.argmax(trace))
        trough_idx = int(np.argmin(trace))

        if peak_idx < trough_idx:
            cat = 'peak-trough'
        else:
            dist = abs(peak_idx - trough_idx)
            if dist <= threshold_samples:
                cat = 'trough-peak-short'
            else:
                cat = 'trough-peak-long'
        categories.append(cat)

    out_csv = Path(base_path) / 'waveform_classification.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['template_index', 'category'])
        for i, cat in enumerate(categories):
            writer.writerow([i, cat])

    print(f"Saved waveform classifications to {out_csv}")