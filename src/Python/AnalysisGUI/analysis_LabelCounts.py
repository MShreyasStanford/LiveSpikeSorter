from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict

__analysis_metadata__ = {
    "name": "Label Counts",
    "description": "Count the number of spikes for each event label.",
    "parameters": [
        ("Event File (.txt)", "FilePath", r"^(\d+)\s+(\d+)$"),
        ("Spike File (.txt)", "FilePath", r"^(\d+),(\d+),(\d+\.\d+)(?:,[^,]+)*$"),
        ("Lookahead start (sample)", "int")
    ]
}

def run(base_dir, eventfile_path, spike_file, lookahead):
    base_dir = Path(base_dir)

    # 1. Read the event file.
    events = []  # each element is a tuple (time, event_label)
    try:
        with open(eventfile_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    print(f"Error: Line '{line.strip()}' is not in the format 't label'.")
                    return
                try:
                    t = float(parts[0])
                    event_label = int(parts[1])
                    events.append((t, event_label))
                except Exception as e:
                    print(f"Error parsing line '{line.strip()}': {e}")
                    return
    except Exception as e:
        print(f"Error reading event file {eventfile_path}: {e}")
        return

    if not events:
        print("Error: No events found in the event file.")
        return

    print(f"Loaded {len(events)} events from {eventfile_path}.")

    # 2. Load spike times and templates from cuda_output/spikeOutput.txt.
    #    Each line is expected to be:
    #    spikeTime, spikeTemplate, spikeAmplitude, someOtherField
    spikes_by_template = defaultdict(list)
    try:
        with open(spike_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    continue
                try:
                    t = float(parts[0])
                    template = int(parts[1])
                    spikes_by_template[template].append(t)
                except Exception:
                    continue
        # Convert lists to numpy arrays.
        for template in spikes_by_template:
            spikes_by_template[template] = np.array(spikes_by_template[template])
    except Exception as e:
        print(f"Error loading spike data from {spike_file}: {e}")
        return

    total_spikes = sum(len(arr) for arr in spikes_by_template.values())
    print(f"Loaded spike data for {len(spikes_by_template)} templates with total {total_spikes} spikes.")

    # 3. Count spikes for each event window for each template.
    #    For each event (t, event_label), we count spikes in [t+lookahead, t+lookahead+100)
    counts_by_template = defaultdict(lambda: defaultdict(int))
    for t, event_label in events:
        window_start = t + lookahead
        window_end = window_start + 100
        for template, spike_times in spikes_by_template.items():
            count = np.sum((spike_times >= window_start) & (spike_times < window_end))
            counts_by_template[template][event_label] += count

    # 4. Define the sinusoidal function.
    def sinusoid(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    # 5. Prepare vertically stacked subplots.
    templates = sorted(counts_by_template.keys())
    num_templates = len(templates)
    # Use a smaller height per subplot if there are many templates.
    if num_templates > 10:
        height_per_subplot = 0.3
    else:
        height_per_subplot = 0.5
    fig, axes = plt.subplots(num_templates, 1, figsize=(10, height_per_subplot*num_templates), sharex=True)
    # If only one subplot exists, wrap it into a list.
    if num_templates == 1:
        axes = [axes]
    # Reduce vertical spacing.
    plt.subplots_adjust(hspace=0.1)

    # 6. Iterate over each template, plot data and attempt a sinusoid fit.
    for i, template in enumerate(templates):
        counts_dict = counts_by_template[template]
        event_labels = sorted(counts_dict.keys())
        counts = np.array([counts_dict[label] for label in event_labels])
        x_data = np.array(event_labels)
        ax = axes[i]
     #   ax.scatter(x_data, counts, color='skyblue', edgecolor='black', label="Data")

        # Only fit if there is enough variation and data points.
        variation = np.max(counts) - np.min(counts)
        if len(x_data) < 4 or variation < 1e-6:
            continue

        # Initial guesses for parameters.
        A0 = variation / 2
        B0 = 0.1        # initial frequency guess
        C0 = 0          # initial phase guess
        D0 = np.mean(counts)
        initial_guess = [A0, B0, C0, D0]

        # Set parameter bounds. We add a small epsilon to ensure strict inequality.
        eps = 1e-6
        lower_bounds = [0, 0, -np.pi, np.min(counts)]
        upper_bounds = [2 * A0 + eps, 1.0, np.pi, np.max(counts) + eps]

        try:
            popt, pcov = curve_fit(
                sinusoid,
                x_data,
                counts,
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000
            )
            A, B, C, D = popt
            x_fit = np.linspace(np.min(x_data), np.max(x_data), 500)
            y_fit = sinusoid(x_fit, A, B, C, D) * 10
            ax.plot(x_fit, y_fit, color='red')
        except Exception as e:
            print(f"Template {template}: Error fitting sinusoid: {e}")
        ax.legend()

    axes[-1].set_xlabel("Event Label")
    plt.tight_layout()
    output_path = base_dir / "label_counts.png"
    plt.savefig(output_path)
    plt.clf()
    print(f"Figure saved to {output_path}")