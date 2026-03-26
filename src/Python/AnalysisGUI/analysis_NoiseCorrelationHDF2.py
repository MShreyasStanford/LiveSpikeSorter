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
    with h5py.File(hdf_path, 'r') as f:
        for name in names:
            if name in f:
                data[name] = f[name][()]
            else:
                print(f"Warning: '{name}' missing.", file=sys.stderr)
    return data


def preprocess_events(data):
    """
    Convert edata timestamps (s) to ms and compute relative acquisition and onset arrays.
    """
    ed_acq = data["trial_info/edata_st1_acquired"].squeeze().astype(float) * 1000.0
    ed_fix = data["trial_info/edata_fixation_stable"].squeeze().astype(float) * 1000.0
    st_on  = data["trial_info/edata_st1_on"].squeeze().astype(float) * 1000.0
    rel_acq_all = ed_acq - ed_fix
    rel_on_all  = st_on  - ed_fix
    return ed_acq, ed_fix, st_on, rel_acq_all, rel_on_all


def build_cumspk(spikes):
    """
    Build cumulative-sum of spikes along time axis.
    """
    N, T, M = spikes.shape
    return np.concatenate([
        np.zeros((N,1,M), dtype=spikes.dtype),
        spikes.cumsum(axis=1)
    ], axis=1)


def select_neurons(cumspk, rel_acq_all, valid_trials, rad_corr,
                   unique_rads, anova_window, pre_event_offset,
                   bin_size, anova_p_thresh, filter_tuned):
    """
    Run ANOVA on counts in a window around acquisition, grouping by direction.
    Returns indices of tuned or untuned neurons.
    """
    T_max = cumspk.shape[1] - 1
    rel_valid = rel_acq_all[valid_trials]
    idx0 = (rel_valid - anova_window + pre_event_offset).astype(int)
    idx1 = (rel_valid + pre_event_offset).astype(int)
    np.clip(idx0, 0, T_max, out=idx0)
    np.clip(idx1, 0, T_max, out=idx1)

    N = cumspk.shape[0]
    pvals = np.zeros(N)
    for i in range(N):
        counts_i = cumspk[i, idx1, valid_trials] - cumspk[i, idx0, valid_trials]
        groups = [counts_i[rad_corr == r] for r in unique_rads]
        if any(len(g) < 2 for g in groups):
            pvals[i] = np.nan
        else:
            pvals[i] = f_oneway(*groups)[1]

    tuned   = np.where(pvals <= anova_p_thresh)[0]
    untuned = np.where(pvals >  anova_p_thresh)[0]
    return tuned if filter_tuned else untuned


def compute_timecourse(cumspk, rel_event_all, valid_trials, rad_corr,
                        unique_rads, selected_neurons, starts,
                        pre_event_offset, bin_size):
    """
    Compute mean and SEM of pairwise noise correlations over sliding windows.
    """
    times = starts
    means_dict = {}
    sems_dict  = {}
    T_max = cumspk.shape[1] - 1

    for r in unique_rads:
        trials = valid_trials[rad_corr[valid_trials] == r]
        if trials.size < 2:
            means_dict[r] = np.full_like(times, np.nan, dtype=float)
            sems_dict[r]  = np.full_like(times, np.nan, dtype=float)
            continue

        means = np.zeros_like(times, dtype=float)
        sems  = np.zeros_like(times, dtype=float)
        rel_evt = rel_event_all[trials]

        for bi, w0 in enumerate(starts):
            idx0 = (rel_evt + w0 + pre_event_offset).astype(int)
            idx1 = idx0 + bin_size
            np.clip(idx0, 0, T_max, out=idx0)
            np.clip(idx1, 0, T_max, out=idx1)

            counts = cumspk[:, idx1, trials] - cumspk[:, idx0, trials]
            R      = np.corrcoef(counts[selected_neurons])
            vals   = R[np.triu_indices(R.shape[0], k=1)]
            vals   = vals[~np.isnan(vals)]
            means[bi] = vals.mean()
            sems[bi]  = vals.std(ddof=0) / np.sqrt(vals.size)

        means_dict[r] = means
        sems_dict[r]  = sems
    return times, means_dict, sems_dict


def add_event_lines(ax, times, rel_acq_all, rel_on_all):
    """
    Draw vertical dashed lines for fixation_stable, acquired, and st1_on.
    """
    x_fix = -np.nanmean(rel_acq_all)
    x_acq = 0.0
    x_on  = np.nanmean(rel_on_all - rel_acq_all)
    for x, label in [(x_fix, 'fixation_stable'), (x_acq, 'acquired'), (x_on, 'st1_on')]:
        if times[0] <= x <= times[-1]:
            ax.axvline(x, linestyle='--', label=label)


def plot_timecourse(cumspk, rel_event_all, rel_acq_all, rel_on_all,
                    valid_trials, rad_corr, unique_rads, selected_neurons,
                    starts, pre_event_offset, bin_size,
                    anova_p_thresh, filter_tuned, hdf_path, area,
                    title_event, x_label, suffix):
    """
    Generic time-course plotting aligned to a given event.
    """
    times, means_dict, sems_dict = compute_timecourse(
        cumspk, rel_event_all, valid_trials, rad_corr,
        unique_rads, selected_neurons, starts,
        pre_event_offset, bin_size
    )
    fig, ax = plt.subplots(figsize=(8,5))
    for r in unique_rads:
        ax.errorbar(
            times, means_dict[r], yerr=sems_dict[r],
            capsize=2, elinewidth=1, marker='o', linestyle='-',
            label=f'rad = {r}'
        )
    add_event_lines(ax, times, rel_acq_all, rel_on_all)
    ax.set(xlabel=x_label,
           ylabel='Mean noise-corr (upper triangle)')
    title = (
        f'Noise-Correlation by Direction (aligned to {title_event})\n'
        f'Kept {len(selected_neurons)} / {cumspk.shape[0]} neurons '
        f'w/ ANOVA p {"<=" if filter_tuned else ">"} {anova_p_thresh}\n'
        f'Session: {Path(hdf_path).name}'
    )
    ax.set_title(title)
    ax.legend(title='esetup_st1_rad')
    fig.tight_layout()
    fig.savefig(f"{hdf_path}_{filter_tuned}_{anova_p_thresh}_{area}_{suffix}.png")
    plt.close(fig)


def run(base_dir, hdf_path,
        bin_left_ms, bin_right_ms,
        noise_corr_left_ms, noise_corr_right_ms,
        bin_size=50, step_size=10,
        hist_time_ms=None,
        filter_tuned=False,
        anova_p_thresh=0.05,
        area="mt"):
    names = [
        "imec0/spk", "imec1/spk",
        "trial_info/edata_st1_acquired",
        "trial_info/edata_fixation_stable",
        "trial_info/esetup_probe_on",
        "trial_info/edata_st1_on",
        "trial_info/esetup_st1_rad",
    ]
    data = load_selected_tensors(str(hdf_path), names)

    spikes = data[f"imec{0 if area=='lip' else 1}/spk"]
    ed_acq, ed_fix, st_on, rel_acq_all, rel_on_all = preprocess_events(data)
    setup = data["trial_info/esetup_probe_on"].squeeze().astype(bool)
    rad   = data["trial_info/esetup_st1_rad"].squeeze()

    correct = ~setup & ~np.isnan(st_on)
    valid_trials = np.where(correct)[0]
    rad_corr      = rad[valid_trials]
    unique_rads   = np.unique(rad_corr)

    cumspk = build_cumspk(spikes)
    pre_event_offset = -bin_left_ms

    selected_neurons = select_neurons(
        cumspk, rel_acq_all, valid_trials, rad_corr,
        unique_rads, anova_window=500, pre_event_offset=pre_event_offset,
        bin_size=bin_size, anova_p_thresh=anova_p_thresh,
        filter_tuned=filter_tuned
    )
    print(f"Filtering result: kept {len(selected_neurons)} / {cumspk.shape[0]} neurons")

    # acquisition-aligned
    starts_acq = np.arange(noise_corr_left_ms,
                           noise_corr_right_ms - bin_size + 1,
                           step_size)
    plot_timecourse(
        cumspk, rel_acq_all, rel_acq_all, rel_on_all,
        valid_trials, rad_corr, unique_rads, selected_neurons,
        starts_acq, pre_event_offset, bin_size,
        anova_p_thresh, filter_tuned, hdf_path, area,
        title_event='edata_st1_acquired',
        x_label='Time (ms) relative to edata_st1_acquired',
        suffix='timecourse_acquired'
    )

    # cue-onset-aligned
    starts_on = np.arange(-500,
                          noise_corr_right_ms - bin_size + 1,
                          step_size)
    pre_event_offset_on = -starts_on[0]
    plot_timecourse(
        cumspk, rel_on_all, rel_acq_all, rel_on_all,
        valid_trials, rad_corr, unique_rads, selected_neurons,
        starts_on, pre_event_offset_on, bin_size,
        anova_p_thresh, filter_tuned, hdf_path, area,
        title_event='edata_st1_on',
        x_label='Time (ms) relative to edata_st1_on',
        suffix='timecourse_cue_onset'
    )

    # ——— Histogram at specified time ———
    if hist_time_ms is not None:
        # find the bin index closest to hist_time_ms
        bin_idx = int(np.argmin(np.abs(starts_acq - hist_time_ms)))
        hist_vals = {}
        for r in unique_rads:
            trials = valid_trials[rad_corr == r]
            if trials.size < 2:
                hist_vals[r] = np.array([])
                continue
            # compute counts for this direction in the selected bin
            idx0 = (rel_acq_all[trials] + starts_acq[bin_idx] + pre_event_offset).astype(int)
            idx1 = idx0 + bin_size
            np.clip(idx0, 0, cumspk.shape[1]-1, out=idx0)
            np.clip(idx1, 0, cumspk.shape[1]-1, out=idx1)
            counts = cumspk[:, idx1, trials] - cumspk[:, idx0, trials]
            R      = np.corrcoef(counts[selected_neurons])
            vals   = R[np.triu_indices(R.shape[0], k=1)]
            hist_vals[r] = vals[~np.isnan(vals)]
        # plot the 2×2 histogram grid
        plot_histogram(hist_vals, starts_acq, bin_idx, anova_p_thresh, filter_tuned, hdf_path, area)

    # ——— Bar graph over fixed window ———
    plot_bar_graph(
        cumspk, rel_acq_all, valid_trials, rad_corr, unique_rads,
        selected_neurons, pre_event_offset, bin_size,
        anova_p_thresh, filter_tuned, hdf_path, area
    )

if __name__ == "__main__":
    session_paths = [Path("C:/SGL_DATA/tg_20231121_MGS_RF.h5")]
    areas = ["mt", "lip"]
    filter_opts = [True, False]
    for filt, area, sess in product(filter_opts, areas, session_paths):
        run(Path("C:/MyCodeOutput"), sess,
            bin_left_ms=-800, bin_right_ms=2400,
            noise_corr_left_ms=-800, noise_corr_right_ms=200,
            bin_size=100, step_size=10,
            hist_time_ms=-400,
            filter_tuned=filt,
            anova_p_thresh=0.2,
            area=area)
