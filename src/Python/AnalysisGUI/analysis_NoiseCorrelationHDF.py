#!/usr/bin/env python3
import os
import sys
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import f_oneway
from itertools import product

def load_selected_tensors(hdf_path, names):
    if not os.path.exists(hdf_path):
        print(f"Error: '{hdf_path}' not found.", file=sys.stderr)
        sys.exit(1)
    data = {}
    print(f"→ attempting to open HDF5 file: {hdf_path!r}")
    print("  exists:", os.path.exists(hdf_path))
    print("  size  :", os.path.getsize(hdf_path), "bytes")
    with h5py.File(hdf_path, 'r') as f:
        for name in names:
            if name in f:
                data[name] = f[name][()]
            else:
                print(f"Warning: '{name}' missing.", file=sys.stderr)
    return data

def plot_noisecorr_timecourse(events, align_to, spikes, rads, selected, correct,
    left_ms=-600, right_ms=400, bin_size=100, step_size=10, pre_event_offset=-600):
    N, T, M = spikes.shape # N = num trials, T = num samples, M = num channels

    offsets_all = events - align_to

    # cumspk[i, j, k] = num spikes for trial i at time indices [0, j] on channel k
    cumspk = np.concatenate([
        np.zeros((N,1,M), dtype=spikes.dtype),
        spikes.cumsum(axis=1)
    ], axis=1)

    # time bins left
    starts = np.arange(
        left_ms,
        right_ms - bin_size + 1,
        step_size
    )

    # time bins for plot
    times = starts

    plt.figure(figsize=(8,5))

    for r in np.unique(rads):
        trials = np.where(correct & (rads == r))[0]
        if idxs.size < 2:
            continue

        means = np.zeros_like(times, dtype=float)
        sems  = np.zeros_like(times, dtype=float)

        offsets = offsets[trials]
        for b, w0 in enumerate(starts):
            idx0 = (offsets + w0 + pre_event_offset).astype(int)
            idx1 = idx0 + bin_size
            np.clip(idx0, 0, T, out=idx0)
            np.clip(idx1, 0, T, out=idx1)
            counts = cumspk[:, idx1, idxs] - cumspk[:, idx0, idxs]
            R      = np.corrcoef(counts[selected_neurons])**2
            vals   = R[np.triu_indices(R.shape[0], k=1)]
            vals   = vals[~np.isnan(vals)]

            # store for histogram if this is the requested bin
            if hist_time_ms is not None and b == bin_idx:
                hist_vals[r] = vals.copy()
            means[b] = vals.mean()
            sems[b]  = vals.std(ddof=0) / np.sqrt(vals.size)

        plt.errorbar(
            times, means, yerr=sems,
            capsize=2, elinewidth=1, marker='o', linestyle='-',
            label=f"rad = {r}"
        )

def filter_rads(unique_rads, rad, correct):
    return np.arange(0, unique_rads.shape[0])

    num_trials_for_rad = np.array([ len(np.where(correct & (rad == r))[0]) for r in unique_rads ])
    print(f"num_trials_for_rad = {num_trials_for_rad}")
    max_num_trials = max(num_trials_for_rad)
    min_num_trials = min(num_trials_for_rad)
    mean = num_trials_for_rad.mean()
    std = num_trials_for_rad.std()
    filtered_rads = np.where(num_trials_for_rad >= mean - 1 * std)[0]
    return filtered_rads

def filter_trials_equal_rads(unique_rads, rad, correct):
    filtered_rads = filter_rads(unique_rads, rad, correct)
    num_trials_for_rad = np.array([ len(np.where(correct & (rad == r))[0]) for r in unique_rads ])
    min_num_trials = min(num_trials_for_rad[filtered_rads])
    rad_filters = [ np.random.choice(np.where(correct & (rad == r))[0], min_num_trials, replace=False) for r in unique_rads[filtered_rads] ]
    rad_filter = np.sort(np.concatenate(rad_filters))
    idxs = np.arange(correct.shape[0])
    mask = np.isin(idxs, rad_filter)
    return mask

def run(base_dir, hdf_path,
        bin_left_ms, bin_right_ms,
        noise_corr_left_ms, noise_corr_right_ms,
        bin_size=50, step_size=10,
        hist_time_ms=None,
        filter_tuned=False,
        anova_p_thresh=0.05,
        area="mt"):

    # Load data
    names = [
        "imec0/spk",
        "imec1/spk",
        "trial_info/edata_st1_acquired",
        "trial_info/edata_fixation_stable",
        "trial_info/esetup_probe_on",
        "trial_info/edata_st1_on",
        "trial_info/esetup_st1_rad",
        "trial_info/edata_st1_maintained",
        "imec0/unit_info/y",
        "imec1/unit_info/y"
    ]

    data = load_selected_tensors(str(hdf_path), names)

    if area == "lip":
        spikes = data["imec0/spk"]  
        y      = data["imec0/unit_info/y"].squeeze().astype(float)                      
    elif area == "mt":
        spikes = data["imec1/spk"]
        y      = data["imec1/unit_info/y"].squeeze().astype(float)
    else:
        print(f"Warning: parameter 'area' not set to known value, skipping.")
        return

    ed_acq = data["trial_info/edata_st1_acquired"].squeeze().astype(float) * 1000.0
    ed_fix = data["trial_info/edata_fixation_stable"].squeeze().astype(float) * 1000.0
    setup  = data["trial_info/esetup_probe_on"].squeeze().astype(bool)
    st_on  = data["trial_info/edata_st1_on"].squeeze().astype(float) * 1000.0
    st_maintained = data["trial_info/edata_st1_maintained"].squeeze().astype(float) * 1000.0
    rad    = data["trial_info/esetup_st1_rad"].squeeze()

    N, T, M = spikes.shape

    # Filter for Correct trials
    correct = ~setup & ~np.isnan(st_maintained)

    if correct.sum() < 2:
        raise RuntimeError("Not enough correct trials.")

    rad_corr    = rad[correct]
    unique_rads = np.unique(rad_corr)
    valid_trials = np.where(correct)[0]

    # Alignment offsets
    rel_acq_all    = ed_acq - ed_fix   # for the original plot
    rel_on_all     = st_on  - ed_fix   # for the new plot

    assert len(np.where(np.isnan(rel_acq_all[correct]))[0]) == 0
    assert len(np.where(np.isnan(rel_on_all[correct]))[0]) == 0

    rad_filter = filter_trials_equal_rads(unique_rads, rad, correct)
    pre_event_offset = -bin_left_ms

    # Build cumulative‐sum along time
    cumspk = np.concatenate([
        np.zeros((N,1,M), dtype=spikes.dtype),
        spikes.cumsum(axis=1)
    ], axis=1)

    # ANOVA‐based filtering
    anova_window = 500  # ms
    with np.errstate(invalid='ignore'):
        idx0 = (rel_acq_all + (-anova_window) + pre_event_offset).astype(int)
        idx1 = (rel_acq_all + pre_event_offset).astype(int)
    np.clip(idx0, 0, T, out=idx0)
    np.clip(idx1, 0, T, out=idx1)

    anova_pvals = np.zeros(N)
    for i in range(N):
        counts_i = cumspk[i, idx1[valid_trials], valid_trials] \
                 - cumspk[i, idx0[valid_trials], valid_trials]
        groups   = [counts_i[rad_corr == r] for r in unique_rads]
        anova_pvals[i] = f_oneway(*groups)[1]

    tuned = np.where(anova_pvals <= anova_p_thresh)
    untuned = np.where(anova_pvals > anova_p_thresh)
    if filter_tuned:
        selected_neurons = tuned[0]
        print(f"Filtering for neurons with ANOVA p <= {anova_p_thresh}. "
              f"Kept {len(selected_neurons)} / {N}")
    else:
        selected_neurons = untuned[0]
        print(f"Filtering for neurons with ANOVA p > {anova_p_thresh}. "
              f"Kept {len(selected_neurons)} / {N}")

    # Sliding‐window setup
    starts = np.arange(
        noise_corr_left_ms,
        noise_corr_right_ms - bin_size + 1,
        step_size
    )
    times  = starts

    # ——— histogram setup ———
    if hist_time_ms is not None:
        # pick the sliding‐window bin closest to hist_time_ms
        bin_idx   = int(np.argmin(np.abs(starts - hist_time_ms)))
        # initialize per‐radius storage for that bin
        hist_vals = {r: None for r in unique_rads}

    # ——— First figure: relative to edata_st1_acquired ———
    plt.figure(figsize=(8,5))
    for r in unique_rads:
        idxs = np.where(rad_filter & correct & (rad == r))[0]
        if idxs.size < 2:
            continue

        print(f"rad = {r} : {idxs.shape}")
        means = np.zeros_like(times, dtype=float)
        sems  = np.zeros_like(times, dtype=float)

        rel_acq = rel_acq_all[idxs]
        for b, w0 in enumerate(starts):
            idx0 = (rel_acq + w0 + pre_event_offset).astype(int)
            idx1 = idx0 + bin_size
            np.clip(idx0, 0, T, out=idx0)
            np.clip(idx1, 0, T, out=idx1)
            counts = cumspk[:, idx1, idxs] - cumspk[:, idx0, idxs]
            R      = np.corrcoef(counts[selected_neurons])**2
            vals   = R[np.triu_indices(R.shape[0], k=1)]
            vals   = vals[~np.isnan(vals)]

            # store for histogram if this is the requested bin
            if hist_time_ms is not None and b == bin_idx:
                hist_vals[r] = vals.copy()
            means[b] = vals.mean()
            sems[b]  = vals.std(ddof=0) / np.sqrt(vals.size)

        plt.errorbar(
            times, means, yerr=sems,
            capsize=2, elinewidth=1, marker='o', linestyle='-',
            label=f"rad = {r}"
        )

    plt.xlabel("Time (ms) relative to edata_st1_acquired")
    plt.ylabel("Mean noise-corr (upper triangle)")
    if filter_tuned:
        plt.title(f"Noise-Correlation by Direction\nKept {len(selected_neurons)} / {N} neurons w/ ANOVA p <= {anova_p_thresh}\nSession: {Path(hdf_path).name}")
    else:
        plt.title(f"Noise-Correlation by Direction\nKept {len(selected_neurons)} / {N} neurons w/ ANOVA p > {anova_p_thresh}\nSession: {Path(hdf_path).name}")
    plt.legend(title="esetup_st1_rad")
    plt.tight_layout()
    plt.savefig(f"{hdf_path}_{filter_tuned}_{anova_p_thresh}_{area}_ncorr_timecourse_acquired.png")

    # ——— PSTH: relative to edata_st1_acquired ———
    plt.figure(figsize=(8,5))
    for r in unique_rads:
        idxs = np.where(rad_filter & correct & (rad == r))[0]
        if idxs.size < 2:
            continue

        counts_all = np.zeros_like(times, dtype=float)

        rel_acq = rel_acq_all[idxs]
        for b, w0 in enumerate(starts):
            idx0 = (rel_acq + w0 + pre_event_offset).astype(int)
            idx1 = idx0 + bin_size
            np.clip(idx0, 0, T, out=idx0)
            np.clip(idx1, 0, T, out=idx1)

            counts = cumspk[:, idx1, idxs] - cumspk[:, idx0, idxs]
            counts_all[b] = counts[selected_neurons].squeeze().sum() / len(idxs) / (bin_size / 1000) / len(selected_neurons)

        plt.errorbar(
            times, counts_all, yerr=sems,
            capsize=2, elinewidth=1, marker='o', linestyle='-',
            label=f"rad = {r}"
        )

    plt.xlabel("Time (ms) relative to edata_st1_acquired")
    plt.ylabel("Spike Count")
    if filter_tuned:
        plt.title(f"PSTH by Direction\nKept {len(selected_neurons)} / {N} neurons w/ ANOVA p <= {anova_p_thresh}\nSession: {Path(hdf_path).name}")
    else:
        plt.title(f"PSTH by Direction\nKept {len(selected_neurons)} / {N} neurons w/ ANOVA p > {anova_p_thresh}\nSession: {Path(hdf_path).name}")
    plt.legend(title="esetup_st1_rad")
    plt.tight_layout()
    plt.savefig(f"{hdf_path}_{filter_tuned}_{anova_p_thresh}_{area}_psth_acquired.png")

    # ——— Second figure: relative to edata_st1_on ———
    starts_on = np.arange(-500, 1800 - bin_size + 1, step_size)
    times_on  = starts_on

    # Compute a new offset so that w0=0 really lands at the onset index
    pre_event_offset_on = -bin_left_ms

    plt.figure(figsize=(8,5))
    for r in unique_rads:
        idxs = np.where(rad_filter & correct & (rad == r))[0]
        if idxs.size < 2:
            continue

        means = np.zeros_like(times_on, dtype=float)
        sems  = np.zeros_like(times_on, dtype=float)

        # rel_on_all was defined up top as (st_on – ed_fix)
        rel_on = rel_on_all[idxs]

        for b, w0 in enumerate(starts_on):
            # **here** use pre_event_offset_on instead of the old pre_event_offset
            idx0 = (rel_on + w0 + pre_event_offset_on).astype(int)
            idx1 = idx0 + bin_size
            np.clip(idx0, 0, T, out=idx0)
            np.clip(idx1, 0, T, out=idx1)

            counts = cumspk[:, idx1, idxs] - cumspk[:, idx0, idxs]
            R      = np.corrcoef(counts[selected_neurons])**2
            vals   = R[np.triu_indices(R.shape[0], k=1)]
            vals   = vals[~np.isnan(vals)]

            means[b] = vals.mean()
            sems[b]  = vals.std(ddof=0) / np.sqrt(vals.size)

        plt.errorbar(
            times_on, means, yerr=sems,
            capsize=2, elinewidth=1, marker='o', linestyle='-',
            label=f"rad = {r}"
        )

    plt.xlabel("Time (ms) relative to edata_st1_on")
    plt.ylabel("Mean noise-corr (upper triangle)")
    plt.title((
        f"Noise-Correlation by Direction\n"
        f"Kept {len(selected_neurons)} / {N} neurons "
        f"w/ ANOVA p {'<=' if filter_tuned else '>'} {anova_p_thresh}\n"
        f"Session: {Path(hdf_path).name}"
    ))
    plt.legend(title="esetup_st1_rad")
    plt.tight_layout()
    plt.savefig(f"{hdf_path}_{filter_tuned}_{anova_p_thresh}_{area}_ncorr_timecourse_cue_onset.png")
    plt.clf()

    # ——— PSTH: relative to edata_st1_acquired ———
    starts_on = np.arange(-500, 1800 - bin_size + 1, step_size)
    times_on  = starts_on
    plt.figure(figsize=(8,5))
    for r in unique_rads:
        idxs = np.where(rad_filter & correct & (rad == r))[0]
        if idxs.size < 2:
            continue

        counts_all = np.zeros_like(times_on, dtype=float)

        rel_acq = rel_on_all[idxs]
        for b, w0 in enumerate(starts_on):
            idx0 = (rel_acq + w0 + pre_event_offset_on).astype(int)
            idx1 = idx0 + bin_size
            np.clip(idx0, 0, T, out=idx0)
            np.clip(idx1, 0, T, out=idx1)

            counts = cumspk[:, idx1, idxs] - cumspk[:, idx0, idxs]
            counts_all[b] = counts[selected_neurons].squeeze().sum() / len(idxs) / (bin_size / 1000) / len(selected_neurons)

        plt.errorbar(
            times_on, counts_all, yerr=sems,
            capsize=2, elinewidth=1, marker='o', linestyle='-',
            label=f"rad = {r}"
        )

    plt.xlabel("Time (ms) relative to edata_st1_on")
    plt.ylabel("Spike Count")
    if filter_tuned:
        plt.title(f"PSTH by Direction\nKept {len(selected_neurons)} / {N} neurons w/ ANOVA p <= {anova_p_thresh}\nSession: {Path(hdf_path).name}")
    else:
        plt.title(f"PSTH by Direction\nKept {len(selected_neurons)} / {N} neurons w/ ANOVA p > {anova_p_thresh}\nSession: {Path(hdf_path).name}")
    plt.legend(title="esetup_st1_rad")
    plt.tight_layout()
    plt.savefig(f"{hdf_path}_{filter_tuned}_{anova_p_thresh}_{area}_psth_on.png")

    if hist_time_ms is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10,8), sharex=True, sharey=True)
        if filter_tuned:
            fig.suptitle(f"Noise‐Corr Histograms at t≈{starts[bin_idx]} ms\nKept {len(selected_neurons)} / {N} neurons w/ ANOVA p <= {anova_p_thresh}\nSession: {Path(hdf_path).name}")
        else:
            fig.suptitle(f"Noise‐Corr Histograms at t≈{starts[bin_idx]} ms\nKept {len(selected_neurons)} / {N} neurons w/ ANOVA p > {anova_p_thresh}\nSession: {Path(hdf_path).name}")
        for ax, r in zip(axes.flatten(), unique_rads):
            vals = hist_vals.get(r)
            if vals is None or len(vals) == 0:
                ax.text(0.5, 0.5, "no data", ha='center')
            else:
                ax.hist(vals, bins=50)
            ax.set_title(f"rad = {r}")
            ax.set_xlabel("Pearson r")
            ax.set_ylabel("Count")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{hdf_path}_{filter_tuned}_{anova_p_thresh}_{area}_ncorr_histogram.png")
        plt.clf()

    # Define window relative to edata_st1_acquired
    w0_bar, w1_bar = -200, 100
    idx0_bar = (rel_acq_all + w0_bar + pre_event_offset).astype(int)
    idx1_bar = (rel_acq_all + w1_bar + pre_event_offset).astype(int)
    np.clip(idx0_bar, 0, T, out=idx0_bar)
    np.clip(idx1_bar, 0, T, out=idx1_bar)

    mean_corrs, sem_corrs = [], []
    for r in unique_rads:
        idxs = np.where(rad_filter & correct & (rad == r))[0]
        if idxs.size < 2:
            mean_corrs.append(np.nan)
            sem_corrs.append(np.nan)
            continue
        # compute counts in the window for each trial of this direction
        counts = cumspk[:, idx1_bar[idxs], idxs] - cumspk[:, idx0_bar[idxs], idxs]
        R = np.corrcoef(counts[selected_neurons])
        vals = R[np.triu_indices(R.shape[0], k=1)]
        vals = vals[~np.isnan(vals)]
        mean_corrs.append(vals.mean())
        sem_corrs.append(vals.std(ddof=0) / np.sqrt(vals.size))

    # plot bar graph
    plt.figure(figsize=(6,4))
    plt.bar([str(r) for r in unique_rads], mean_corrs, yerr=sem_corrs, capsize=5)
    plt.xlabel("Direction (rad)")
    plt.ylabel("Mean noise-corr (upper triangle)")
    plt.title(
        f"Avg Noise-Corr [-200,100]ms after edata_st1_acquired\n"
        f"Kept {len(selected_neurons)} / {N} neurons w/ ANOVA p {'<=' if filter_tuned else '>'} {anova_p_thresh}\n"
        f"Session: {hdf_path}"
    )
    plt.tight_layout()
    plt.savefig(f"{hdf_path}_{filter_tuned}_{anova_p_thresh}_{area}_ncorr_bar_acquired.png")
    plt.clf()

    # ——— 1) Scatter + sliding‐window mean SEM by direction ———
    # window [-200, -50] ms relative to edata_st1_acquired
    w0_dist, w1_dist = -200, -100
    idx0_dist = (rel_acq_all + w0_dist + pre_event_offset).astype(int)
    idx1_dist = (rel_acq_all + w1_dist + pre_event_offset).astype(int)
    np.clip(idx0_dist, 0, T, out=idx0_dist)
    np.clip(idx1_dist, 0, T, out=idx1_dist)

    window_size = 100  # µm
    step        = 100 # µm

    fig, axes = plt.subplots(2, 2, figsize=(12,10), sharex=True, sharey=True)
    for ax, r in zip(axes.flatten(), unique_rads):
        # 1) collect pairwise distances & correlations for this direction
        idxs_r  = np.where(rad_filter & correct & (rad == r))[0]
        counts_r = cumspk[:, idx1_dist[idxs_r], idxs_r] \
                 - cumspk[:, idx0_dist[idxs_r], idxs_r]
        Rr = np.corrcoef(counts_r[selected_neurons])

        # neuron y‐positions (µm)
        y_sel = y[selected_neurons]
        dists_r, corrs_r = [], []
        for i in range(len(y_sel)):
            for j in range(i+1, len(y_sel)):
                dists_r.append(abs(y_sel[i] - y_sel[j]))
                corrs_r.append(Rr[i, j])
        dists_r = np.array(dists_r)
        corrs_r = np.array(corrs_r)

        # 2) raw scatter
        ax.scatter(dists_r, corrs_r**2, alpha=0.3)

        # 3) sliding‐window bins for mean ± SEM
        max_d  = dists_r.max()
        starts = np.arange(0, max_d - window_size + step, step)
        centers = starts
        mean_corr = np.full_like(centers, np.nan, dtype=float)
        sem_corr  = np.full_like(centers, np.nan, dtype=float)
        for k, s in enumerate(starts):
            mask = (dists_r >= s) & (dists_r <= s + window_size)

            if mask.any():
                vals = corrs_r[mask]**2
                vals = vals[~np.isnan(vals)]
                if vals.size > 0:
                    mean_corr[k] = vals.mean()
                    sem_corr[k]  = vals.std(ddof=0) / np.sqrt(vals.size)

        # 4) overlay mean ± SEM (in red for visibility)
        valid = ~np.isnan(mean_corr)
        ax.errorbar(
            centers[valid], mean_corr[valid], yerr=sem_corr[valid],
            fmt='o-', linewidth=2, zorder=10,
            capsize=4, elinewidth=1, markeredgewidth=1,
            color='red',
            label="binned mean ± SEM"
        )

        ax.set_title(f"rad = {r}")
        ax.set_xlabel("Distance (µm)")
        ax.set_ylabel("Noise corr (r^2)")

    plt.suptitle(
        f"Noise Corr vs Distance by Direction [{w0_dist},{w1_dist}] ms\n"
        f"Kept {len(selected_neurons)} / {N} neurons w/ ANOVA p {'<=' if filter_tuned else '>'} {anova_p_thresh}\n"
        f"Session: {hdf_path}"
    )
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(f"{hdf_path}_{filter_tuned}_{anova_p_thresh}_{area}_ncorr_dist_scatter.png")
    plt.clf()

    # ——— Sliding‐window binned plot by direction ———
    window_size = 100  # µm
    step        = 10  # µm

    fig, ax = plt.subplots(figsize=(8,5))
    for r in unique_rads:
        # get counts & corr matrix for this direction
        idxs_r  = np.where(rad_filter & correct & (rad == r))[0]
        counts_r = cumspk[:, idx1_dist[idxs_r], idxs_r] \
                 - cumspk[:, idx0_dist[idxs_r], idxs_r]
        Rr = np.corrcoef(counts_r[selected_neurons])

        # build pairwise distances & correlations
        y_sel = y[selected_neurons]            # true y‐coords
        dists_r, corrs_r = [], []
        for i in range(len(y_sel)):
            for j in range(i+1, len(y_sel)):
                dists_r.append(abs(y_sel[i] - y_sel[j]))
                corrs_r.append(Rr[i, j])
        dists_r = np.array(dists_r)
        corrs_r = np.array(corrs_r)

        # define sliding windows
        max_d  = dists_r.max()
        starts = np.arange(0, max_d - 2 * window_size + step, step)
        centers = starts

        # allocate stats
        mean_corr = np.full(starts.shape, np.nan, dtype=float)
        sem_corr  = np.full(starts.shape, np.nan, dtype=float)

        # compute mean±SEM in each window
        for k, s in enumerate(starts):
            mask = (dists_r >= s) & (dists_r < s + window_size)
            if np.any(mask):
                vals = corrs_r[mask]**2
                vals = vals[~np.isnan(vals)]
                if vals.size > 0:
                    mean_corr[k] = vals.mean()
                    sem_corr[k]  = vals.std(ddof=0) / np.sqrt(vals.size)

        # plot one line per direction
        ax.errorbar(
            centers, mean_corr, yerr=sem_corr,
            fmt='o-', capsize=4, elinewidth=1, markeredgewidth=1,
            label=f"rad = {r}"
        )

    ax.set_xlabel("Inter-neuron distance (µm)")
    ax.set_ylabel("Mean noise-corr ± SEM")
    ax.set_title(
        f"Sliding-window (50 µm step 10 µm) Noise Corr by Distance [{w0_dist},{w1_dist}] ms\n"
        f"{len(selected_neurons)}/{N} neurons, session {Path(hdf_path).name}"
    )
    ax.legend(title="Direction")
    plt.tight_layout()
    plt.savefig(f"{hdf_path}_{filter_tuned}_{anova_p_thresh}_{area}_ncorr_distance_binned.png")
    plt.clf()

if __name__ == "__main__":
    session_paths = [
        Path("C:/SGL_DATA/tg_20231206_MGS_RF.h5"),
        Path("C:/SGL_DATA/tg_20231121_MGS_RF.h5"),
      #  Path("C:/SGL_DATA/tg_20231215_MGS_RF.h5"),
       # Path("C:/SGL_DATA/tg_20231201_MGS_RF.h5")
    ]

    areas = [
        "mt",
        "lip"
    ]

    # True : keep p <= anova_p_thresh
    # False: keep p  > anova_p_thresh
    filter_tuned_options = [
        True,
        False
    ]

    for filter_tuned, area, session_path in product(filter_tuned_options, areas, session_paths):
        run(
            Path("C:/MyCodeOutput"),
            session_path,
            bin_left_ms         = -600,
            bin_right_ms        = 2400,   
            noise_corr_left_ms  = -800,
            noise_corr_right_ms = 300,
            bin_size            = 100,
            step_size           = 10,
            hist_time_ms        = -400,
            filter_tuned        = filter_tuned,
            anova_p_thresh      = 0.2,
            area                = area
        )