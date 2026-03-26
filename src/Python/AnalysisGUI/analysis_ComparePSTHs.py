from pathlib import Path
import matplotlib.pyplot as plt
import random
import os
import pickle
import numpy as np
import cupy as cp
import tqdm
from matplotlib.font_manager import FontProperties
import bisect

__analysis_metadata__ = {
    "name": "Compare PSTHs Between Two Spike Files",
    "description": "For each template, compute and compare normalized PSTHs from two spike files against a common event file.",
    "parameters": [
        ("Spike File 1 (.txt)", "FilePath", r"^(\d+),(\d+),(\d+\.\d+)(?:,[^,]+)*$"),
        ("Spike File 2 (.txt)", "FilePath", r"^(\d+),(\d+),(\d+\.\d+)(?:,[^,]+)*$"),
        ("Event File (.txt)", "FilePath", r"^(\d+)\s+(\d+)$"),
        ("Number of Trials to Subsample (-1 for all)", "int")
    ]
}

def stonum(s: str):
    return float(s) if ('.' in s or 'e' in s or 'E' in s) else int(s)

def get_events(eventfile):
    print(f"Loading events from {eventfile}.")
    with open(eventfile, 'r') as file:
        return  [stonum(line) if len(line.split(' ')) <= 1 else stonum(line.split(' ')[0]) for line in file ]
    return None

def cropped_events(events, min_spike, max_spike, k):
    cropped_events = [ event for event in events if min_spike <= event <= max_spike ]
    if k == -1:
        return cropped_events
    cropped_events = sorted(random.sample(cropped_events, min(k, len(cropped_events))))
    return cropped_events

def get_spikes(file):
    print(f"Loading spikes from {file}")
    with open(file, 'r') as f:
        return [
            (stonum(line.split(',')[0]), stonum(line.split(',')[1]))
            for line in f
        ]
    return None

def compute_psth_data_gpu(spikes, events):
    psth_data = {}
    events_cp = cp.asarray(events)

    inc = 10 * 30         
    window_size = 100 * 30  
    start_initial = -200 * 30  
    max_end = 500 * 30        

    num_windows = int((max_end - (start_initial + window_size)) // inc + 1)
    window_starts = cp.arange(num_windows) * inc + start_initial
    standard_x = cp.asnumpy(window_starts + window_size) / 30  # back to ms

    T = max((s[1] for s in spikes), default=0)

    for template in tqdm.tqdm(range(1, T + 1)):
        template_spikes = [s for s in spikes if s[1] == template]
        if not template_spikes:
            continue

        S = cp.asarray([s[0] for s in template_spikes], dtype=cp.float32)
        S = cp.sort(S)

        lower_bounds = events_cp[:, None] + window_starts[None, :]
        upper_bounds = lower_bounds + window_size

        if S.size > 0:
            left_indices = cp.searchsorted(S, lower_bounds, side='left')
            right_indices = cp.searchsorted(S, upper_bounds, side='right')
            counts = right_indices - left_indices
            agg_counts = cp.sum(counts, axis=0)
            aggregated_counts = cp.asnumpy(agg_counts)
        else:
            aggregated_counts = np.zeros(num_windows, dtype=int)

        psth_data[template] = {
            'standard_x': standard_x.tolist(),
            'oss_y': aggregated_counts.tolist()
        }

    return psth_data

def pearson_corr_numpy(x, y):
    x_arr = np.array(x)
    y_arr = np.array(y)
    return np.corrcoef(x_arr, y_arr)[0, 1]

def avg_ratio(x, y):
    """
    Compute the average of (x[i] - y[i]) / y[i].
    Assumes len(x) == len(y) and all y[i] != 0.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = 0
    total = 0.0
    for xi, yi in zip(x, y):
        if yi == 0:
            continue
        total += abs(xi - yi) / yi
        n += 1
    return total / n

def mass_difference(x, y):
    """
    Compute ‖x - y‖₁ / ‖y‖₁.
    Assumes len(x) == len(y) and sum(abs(y)) != 0.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    num = 0.0
    denom = 0.0
    for xi, yi in zip(x, y):
        num   += abs(xi - yi)
        denom += abs(yi)
    if denom == 0:
        raise ZeroDivisionError("L1(y) is zero, cannot divide")
    return num / denom

def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the discrete Fréchet distance between two 1D curves x and y.

    Parameters
    ----------
    x : array-like, shape (n,)
        First curve (sequence of scalar points).
    y : array-like, shape (m,)
        Second curve.

    Returns
    -------
    float
        The Fréchet distance between x and y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    # cache array, initialized to -1 (means “uncomputed”)
    ca = np.full((n, m), -1.0)

    def _c(i: int, j: int) -> float:
        # helper recursion with memoization
        if ca[i, j] >= 0:
            return ca[i, j]

        d = abs(x[i] - y[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i-1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j-1), d)
        else:
            ca[i, j] = max(
                min(_c(i-1, j), _c(i-1, j-1), _c(i, j-1)),
                d
            )
        return ca[i, j]

    return _c(n-1, m-1)

def run(directory,
        spike_file1: Path,
        spike_file2: Path,
        eventfile: Path,
        k: int = -1):
    # Load fonts
    script_dir = Path(__file__).resolve().parent
    font_dir = script_dir / "fonts" / "gill-sans-2"
    font_prop = FontProperties(
        fname=font_dir / "Gill Sans.otf",
        size=14,
    )
    bold_font_prop = FontProperties(
        fname=font_dir / "Gill Sans Bold.otf",
        size=14,
    )
    # make all text thicker
    font_prop.set_weight('bold')

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    spikes1 = get_spikes(spike_file1)
    spikes2 = get_spikes(spike_file2)
    if not spikes1 or not spikes2:
        print("Error: one of the spike files did not load any spikes.")
        return

    T1 = max(temp for _, temp in spikes1)
    T2 = max(temp for _, temp in spikes2)
    print(f"T1 = {T1}, T2 = {T2}")

    events = get_events(eventfile)
    min_s = min(s for (s, _) in spikes1 + spikes2)
    max_s = max(s for (s, _) in spikes1 + spikes2)
    events = cropped_events(events, min_s, max_s, k)
    if not events:
        print("No events left after cropping.")
        return

    psth1 = compute_psth_data_gpu(spikes1, events)
    psth2 = compute_psth_data_gpu(spikes2, events)

    # compute correlations + normalized curves
    corr_data = {}
    for tmpl in sorted(set(psth1) | set(psth2)):
        d1 = psth1.get(tmpl, {'standard_x': [], 'oss_y': []})
        d2 = psth2.get(tmpl, {'standard_x': [], 'oss_y': []})
        x1, y1 = d1['standard_x'], d1['oss_y']
        x2, y2 = d2['standard_x'], d2['oss_y']
        if x1 != x2 or not x1:
            continue

        y1_arr = np.array(y1, dtype=float)
        y2_arr = np.array(y2, dtype=float)
        max1 = y1_arr.max() if y1_arr.size else 0
        max2 = y2_arr.max() if y2_arr.size else 0
        y1_norm = y1_arr / max1 if max1 > 0 else y1_arr
        y2_norm = y2_arr / max2 if max2 > 0 else y2_arr

        corr = pearson_corr_numpy(y1_norm, y2_norm)
        corr_data[tmpl] = {
            'x': x1,
            'y1': y1_norm.tolist(),
            'y2': y2_norm.tolist(),
            'corr': corr
        }

    if not corr_data:
        print("No templates to plot.")
        return

    corr_data_outname = directory / "psth_corr_data.txt"
    with open(corr_data_outname, 'w') as f:
        for t, d in corr_data.items():
            f.write(f"{t},{d['corr']}\n")

    fs = 30000
    start_sample = -200 * (fs // 1000)   # –200 ms
    end_sample   =  500 * (fs // 1000)   # +500 ms
    win_dur_s    = (end_sample - start_sample) / fs

    # 2) Build a lookup of spike times per template (using spikes1)
    times_by_tmpl = {}
    for t, tmpl in spikes1:
        times_by_tmpl.setdefault(tmpl, []).append(t)
    for tmpl in times_by_tmpl:
        times_by_tmpl[tmpl].sort()

    # 3) Select only templates with mean rate ≥ 1 Hz
    active_templates = []
    for tmpl, times in times_by_tmpl.items():
        counts = []
        for evt in events:
            lo = evt + start_sample
            hi = evt + end_sample
            left  = bisect.bisect_left(times, lo)
            right = bisect.bisect_right(times, hi)
            counts.append(right - left)
        mean_rate = (sum(counts) / len(counts)) / win_dur_s
        if mean_rate >= 2.5:
            active_templates.append(tmpl)

    # 4) Now plot only those correlations
    corr_histogram_outname = directory / "psth_corr_histogram_active.pdf"
    correlations = [
        d['corr']
        for tmpl, d in corr_data.items()
        if tmpl in active_templates
    ]
    fig, ax = plt.subplots()
    ax.hist(correlations,
            bins=np.arange(0, 1, 0.025),
            edgecolor='black')

    # hide the top/right spines properly
    for spine in ['right','top']:
        ax.spines[spine].set_visible(False)

    ax.set_xlabel("Pearson correlation between OSS PSTH and KS PSTH", fontproperties=font_prop)
    ax.set_ylabel("Number of templates",                         fontproperties=font_prop)
    ax.set_title(f"Templates ≥ 2.5 Hz (n={len(correlations)}) for Closest Cluster (PCA space)",    fontproperties=font_prop)
    print(np.array(correlations).mean())
    fig.tight_layout()
    fig.savefig(corr_histogram_outname)
    plt.close(fig)

    # threshold by mass
    masses = {t: sum(d['y1']) + sum(d['y2']) for t, d in corr_data.items()}
    max_mass = max(masses.values())
    threshold = 0.2 * max_mass
    filtered = {t: d for t, d in corr_data.items() if masses[t] >= threshold}
    if not filtered:
        print("No templates exceed the mass threshold; nothing to plot.")
        return

    # pick top-9 by correlation
    top9 = sorted(filtered.items(),
                  key=lambda kv: kv[1]['corr'],
                  reverse=True)[:9]
    bottom9 = sorted(filtered.items(),key=lambda kv: kv[1]['corr'], reverse=True)[-9:]

    print(f"Top 9 templates are {[t for t, d in top9]}.")

    # 3×3 grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    for ax, (tmpl, data) in zip(axes.flat, top9):
        ax.plot(data['x'], data['y1'], label='OSS', alpha=0.7)
        ax.plot(data['x'], data['y2'], label='KS',  alpha=0.7)
        ax.set_xlabel("Time (ms)",             fontproperties=font_prop)
        ax.set_ylabel("Normalized spike count", fontproperties=font_prop)
        ax.set_title("")  # no per-subplot title

    # single legend for the whole figure, top-right
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper right',
               prop=font_prop)

    # master title, larger + bold
    fig.suptitle(
        "Example Templates for Closest Cluster (PCA space)",
        fontproperties=bold_font_prop,
        fontsize=20,
        y=0.95
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    outname = directory / "top9_single_neuron_comparisons.pdf"
    plt.savefig(outname)
    plt.clf()
    print(f"Saved example templates comparison to {outname}")

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    for ax, (tmpl, data) in zip(axes.flat, bottom9):
        ax.plot(data['x'], data['y1'], label='OSS', alpha=0.7)
        ax.plot(data['x'], data['y2'], label='KS',  alpha=0.7)
        ax.set_xlabel("Time (ms)",             fontproperties=font_prop)
        ax.set_ylabel("Normalized spike count", fontproperties=font_prop)
        ax.set_title("")  # no per-subplot title

    # single legend for the whole figure, top-right
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper right',
               prop=font_prop)

    # master title, larger + bold
    fig.suptitle(
        "Example Templates for Closest Cluster (PCA space)",
        fontproperties=bold_font_prop,
        fontsize=20,
        y=0.95
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    outname = directory / "bottom9_single_neuron_comparisons.pdf"
    plt.savefig(outname)
    plt.clf()
    print(f"Saved example templates comparison to {outname}")