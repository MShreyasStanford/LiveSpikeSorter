import os
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

__analysis_metadata__ = {
    "name": "Tuning Significance",
    "description": "TODO",
    "parameters": [
        ("Spikes File (.txt)", "FilePath"),
        ("Event File (.txt)", "FilePath", r"^(\d+)\s+(\d+)$"),
        ("Lookahead Start (samples)", "int"),
        ("Lookahead End (samples)", "int")
    ]
}


def run(base_dir, spike_output, eventfile, lookahead_start, lookahead_end):
    """
    Perform one-way ANOVA per template (neuron) to test tuning across conditions.

    For each spike template j:
      - Build group_i: a list of spike counts in the lookahead window for each trial of condition i.
      - Perform one-way ANOVA across these condition-specific vectors.

    Outputs:
      - 'anova_results.csv': rows of [template, F_statistic, p_value]
      - 'anova_pvalue_histogram.png': histogram of p-values across templates
      - 'template_<j>_low_p.png': mean±SEM plot for up to 5 templates with p < 0.05
      - 'template_<j>_high_p.png': mean±SEM plot for up to 5 templates with p > 0.9

    Parameters
    ----------
    base_dir : str or Path
        Directory to write output files.
    spike_output : str or Path
        CSV file with columns: spike_time (int), spike_template (int).
    eventfile : str or Path
        Whitespace-delimited file with columns: event_time (int), label (int).
    lookahead_start : int
        Start offset (in samples) after event_time.
    lookahead_end : int
        End offset (in samples) after event_time.
    """
    # Ensure base directory
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Load spike data
    spike_path = Path(spike_output)
    if not spike_path.is_absolute():
        spike_path = base_path / spike_path
    spike_times, spike_templates = [], []
    with open(spike_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                t = int(row[0]); templ = int(row[1])
            except (ValueError, IndexError):
                continue
            spike_times.append(t); spike_templates.append(templ)
    spike_times = np.array(spike_times)
    spike_templates = np.array(spike_templates)

    # Load events
    event_path = Path(eventfile)
    if not event_path.is_absolute():
        event_path = base_path / eventfile
    event_times, event_labels = [], []
    with open(event_path, 'r') as f:
        for line in f:
            parts = line.split()
            try:
                et = int(parts[0]); lbl = int(parts[1])
            except (ValueError, IndexError):
                continue
            event_times.append(et); event_labels.append(lbl)
    event_times = np.array(event_times)
    event_labels = np.array(event_labels)

    # Unique identifiers
    unique_templates = np.unique(spike_templates)
    unique_labels = np.unique(event_labels)

    # Initialize per-template per-label counts
    counts = {templ: {lbl: [] for lbl in unique_labels} for templ in unique_templates}

    # Iterate over trials and count spikes in window
    for et, lbl in zip(event_times, event_labels):
        window_start = et + lookahead_start
        window_end = et + lookahead_end
        mask = (spike_times >= window_start) & (spike_times < window_end)
        trial_templates = spike_templates[mask]
        unique_t, counts_t = np.unique(trial_templates, return_counts=True)
        trial_count = dict(zip(unique_t, counts_t))
        for templ in unique_templates:
            counts[templ][lbl].append(trial_count.get(templ, 0))

    # Perform ANOVA per template
    results = []  # list of (templ, F_stat, p_val)
    for templ in unique_templates:
        groups = [counts[templ][lbl] for lbl in unique_labels]
        # Require each group to have at least 2 samples for f_oneway
        if any(len(g) < 2 for g in groups):
            continue
        # Skip if all values are constant (to avoid scipy warning)
        if all(len(set(g)) == 1 for g in groups):
            continue
        try:
            F_stat, p_val = f_oneway(*groups)
        except Exception:
            continue
        results.append((templ, F_stat, p_val))

    # Save results to CSV, sorted by p-value ascending
    out_csv = base_path / 'anova_results.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['template', 'F_statistic', 'p_value'])
        for templ, F_stat, p_val in sorted(results, key=lambda x: x[2]):
            writer.writerow([templ, F_stat, p_val])

    # Plot histogram of p-values
    p_vals = [p for _, _, p in results]
    plt.figure()
    plt.hist(p_vals, bins=50)
    plt.xlabel('p-value')
    plt.ylabel('Count')
    plt.title('Distribution of ANOVA p-values')
    plt.tight_layout()
    plt.savefig(base_path / 'anova_pvalue_histogram.png')
    plt.close()

    # Identify templates with low and high p-values
    low_thresh = 0.05
    high_thresh = 0.9
    sorted_by_p = sorted(results, key=lambda x: x[2])
    low_templates = [t for t, _, p in sorted_by_p if p < low_thresh][:5]
    high_templates = [t for t, _, p in sorted(results, key=lambda x: x[2], reverse=True) if p > high_thresh][:5]

    # Function to plot mean ± SEM per condition
    def plot_template_mean_sem(templ, p_val, category):
        means = []
        sems = []
        for lbl in unique_labels:
            vals = np.array(counts[templ][lbl])
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            means.append(mean)
            sems.append(sem)
        plt.figure()
        plt.errorbar(unique_labels, means, yerr=sems, fmt='-o', capsize=5)
        plt.xlabel('Condition')
        plt.ylabel('Mean spike count ± SEM')
        plt.title(f'Template {templ} ({category}), p={p_val:.3f}')
        plt.tight_layout()
        fname = base_path / f'template_{templ}_{category}.png'
        plt.savefig(fname)
        plt.close()

    # Plot for low p-value templates
    for templ, F_stat, p_val in [r for r in results if r[0] in low_templates]:
        plot_template_mean_sem(templ, p_val, 'low_p')

    # Plot for high p-value templates
    for templ, F_stat, p_val in [r for r in results if r[0] in high_templates]:
        plot_template_mean_sem(templ, p_val, 'high_p')

    print(f"ANOVA complete for {len(results)} templates. Results saved to {out_csv} and plots generated.")