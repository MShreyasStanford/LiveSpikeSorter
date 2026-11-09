#!/usr/bin/env python3
"""
closedloop_baseline.py -- build per-population (FS/RS) baseline statistics from
the TRAINING recording, for the LSS closed-loop SDM trigger.

Produces closedloop_stats.txt (plain key/value, trivially parsed by
ClosedLoopSdmProcessor in C++). The test run loads it via --sdm_stats.

Spike source (default): Kilosort's OWN training sort, already in oss_input/ after
Kilosort runs -- spike_times.npy + spike_templates.npy, in the same final-cluster
index space LSS emits live and rs_fs_labels.csv uses. No need to re-stream the
training data through LSS. (--spikes overrides this with an LSS spikeOutput.txt, which
matches the live detector exactly but requires an LSS pass over the training file.)

Inputs
  spike_times.npy / spike_templates.npy   Kilosort training sort (default source)
  rs_fs_labels.csv  from waveform_classification.py (template_index,label,width_us)
  --trial-events    (optional) CSV from nidq_events.py with a
                    'trial_start_imec_sample' column. When given, only spikes
                    inside each trial's [onset, onset+epoch_s] window are used
                    (the delay period the trigger actually runs in). This is how
                    nidq_events.py feeds the baseline. Without it, the whole
                    recording is used.

Two statistics, both PER POPULATION (FS and RS separately), both built on the
same live-comparable value V_P(bin) = (total spikes of pop P in the bin) / n_P,
where n_P is the number of labelled neurons in P:

  median-score : per neuron, median over its bins; mean of those across the pop.
                 Live rule: FS-low when V_FS(bin) < fs_median (+ offset).
  z-score      : mean and sd of V_P(bin) over training bins.
                 Live rule: FS-low when (V_FS(bin)-fs_mean)/fs_sd < -trigger_z.

Both are written; the processor's --sdm_mode picks which to use.

Usage
  py -3.10 closedloop_baseline.py <oss_input_dir> [--spikes spikeOutput.txt]
       [--labels rs_fs_labels.csv] [--bin-ms 100]
       [--trial-events nidq_trial_events.csv] [--epoch-s 1.5]
       [--fs 30000] [--out closedloop_stats.txt]
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def load_labels(path):
    """template_index -> 'FS'/'RS'/'NONE'. Returns (fs_ids, rs_ids) as sets."""
    fs, rs = set(), set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            idx = int(row["template_index"])
            lab = row["label"].strip().upper()
            if lab == "FS":
                fs.add(idx)
            elif lab == "RS":
                rs.add(idx)
    return fs, rs


def load_ks_spikes(oss_dir):
    """Kilosort training sort: (samples, templates) from spike_times.npy +
    spike_templates.npy in oss_input/. Both are final-cluster space, matching
    what LSS emits live and what rs_fs_labels.csv is keyed on."""
    st = np.load(Path(oss_dir) / "spike_times.npy").astype(np.int64).ravel()
    sc = np.load(Path(oss_dir) / "spike_templates.npy").astype(np.int64).ravel()
    n = min(st.size, sc.size)
    return st[:n], sc[:n]


def load_spikes(path):
    """Return (samples, templates) int64 arrays from an LSS spikeOutput.txt."""
    samples, templates = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            samples.append(int(parts[0]))
            templates.append(int(parts[1]))
    return np.asarray(samples, dtype=np.int64), np.asarray(templates, dtype=np.int64)


def epoch_mask(samples, trial_events, epoch_samples):
    """Keep spikes falling in any [onset, onset+epoch_samples) trial window."""
    onsets = []
    with open(trial_events, newline="") as fh:
        for row in csv.DictReader(fh):
            key = "trial_start_imec_sample" if "trial_start_imec_sample" in row else None
            if key is None:  # be lenient about column naming
                key = next((k for k in row if "imec" in k and "sample" in k), None)
            if key:
                onsets.append(int(float(row[key])))
    if not onsets:
        raise ValueError(f"No trial onset samples found in {trial_events}")
    onsets = np.asarray(sorted(onsets), dtype=np.int64)
    # a spike is in-epoch if the nearest onset at or before it is within epoch
    idx = np.searchsorted(onsets, samples, side="right") - 1
    in_epoch = np.zeros(samples.shape, dtype=bool)
    valid = idx >= 0
    in_epoch[valid] = (samples[valid] - onsets[idx[valid]]) < epoch_samples
    return in_epoch, len(onsets)


def pop_stats(samples, templates, pop_ids, bin_samples, n_bins, bin0):
    """Per-population median-score and z-score over binned spike counts.

    Returns dict with n, median (mean-of-per-neuron-medians), mean, sd.
    Population value per bin = total pop spikes in bin / n_pop.
    """
    n_pop = len(pop_ids)
    if n_pop == 0 or n_bins <= 0:
        return dict(n=n_pop, median=0.0, mean=0.0, sd=1.0)

    # per-neuron per-bin counts (dense over the analysed bins)
    ids = sorted(pop_ids)
    id_row = {tid: r for r, tid in enumerate(ids)}
    counts = np.zeros((n_pop, n_bins), dtype=np.int64)
    for s, t in zip(samples, templates):
        if t in id_row:
            b = (s - bin0) // bin_samples
            if 0 <= b < n_bins:
                counts[id_row[t], b] += 1

    # Population value per bin = mean over neurons (== total pop spikes / n_pop).
    # This is the single live-comparable value both statistics are built on.
    pop_value = counts.mean(axis=0)

    # median-score: median of the population value over bins. By construction
    # ~50% of training bins fall below it -> the "center in half" the trigger
    # wants. (Do NOT use mean-of-per-neuron-medians: 100 ms bins are mostly 0/1
    # per neuron, so that collapses far below the population median.)
    median_score = float(np.median(pop_value))

    # z-score: mean and sd of the same population value over bins.
    mean = float(pop_value.mean())
    sd = float(pop_value.std(ddof=1)) if n_bins > 1 else 0.0
    if not (sd > 0.0):
        sd = 1.0
    return dict(n=n_pop, median=median_score, mean=mean, sd=sd)


def main():
    ap = argparse.ArgumentParser(description="Build FS/RS closed-loop baseline stats from training data.")
    ap.add_argument("oss_input", help="oss_input dir (spike_times.npy, spike_templates.npy, rs_fs_labels.csv)")
    ap.add_argument("--spikes", default=None,
                    help="optional LSS spikeOutput.txt; default is Kilosort's spike_times/spike_templates in oss_input")
    ap.add_argument("--labels", default="rs_fs_labels.csv")
    ap.add_argument("--bin-ms", type=float, default=100.0)
    ap.add_argument("--trial-events", default=None,
                    help="nidq_events.py trial CSV; restricts baseline to trial delay windows")
    ap.add_argument("--epoch-s", type=float, default=1.5,
                    help="seconds after each trial onset to include (default 1.5 = the delay)")
    ap.add_argument("--fs", type=float, default=30000.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = Path(args.oss_input)
    fs_ids, rs_ids = load_labels(base / args.labels)
    if args.spikes:
        samples, templates = load_spikes(base / args.spikes)   # LSS detector-matched
        source = "LSS spikeOutput"
    else:
        samples, templates = load_ks_spikes(base)              # Kilosort training sort
        source = "Kilosort training sort"
    if samples.size == 0:
        raise SystemExit("No spikes found.")
    print(f"  source: {source} ({samples.size} spikes)")

    bin_samples = int(round(args.bin_ms / 1000.0 * args.fs))
    n_trials = 0
    if args.trial_events:
        mask, n_trials = epoch_mask(samples, Path(args.trial_events),
                                    int(round(args.epoch_s * args.fs)))
        samples, templates = samples[mask], templates[mask]
        if samples.size == 0:
            raise SystemExit("No spikes fell inside the trial epochs.")

    bin0 = int(samples.min())
    n_bins = int((samples.max() - bin0) // bin_samples) + 1

    fs_stat = pop_stats(samples, templates, fs_ids, bin_samples, n_bins, bin0)
    rs_stat = pop_stats(samples, templates, rs_ids, bin_samples, n_bins, bin0)

    out = Path(args.out) if args.out else base / "closedloop_stats.txt"
    with open(out, "w") as fh:
        fh.write(f"bin_ms {args.bin_ms:g}\n")
        fh.write(f"n_bins {n_bins}\n")
        fh.write(f"n_trials {n_trials}\n")
        fh.write(f"fs_n {fs_stat['n']}\n")
        fh.write(f"rs_n {rs_stat['n']}\n")
        fh.write(f"fs_median {fs_stat['median']:.6f}\n")
        fh.write(f"rs_median {rs_stat['median']:.6f}\n")
        fh.write(f"fs_mean {fs_stat['mean']:.6f}\n")
        fh.write(f"fs_sd {fs_stat['sd']:.6f}\n")
        fh.write(f"rs_mean {rs_stat['mean']:.6f}\n")
        fh.write(f"rs_sd {rs_stat['sd']:.6f}\n")

    print(f"  bins={n_bins} ({args.bin_ms:g} ms)  trials={n_trials}")
    print(f"  FS n={fs_stat['n']}  median={fs_stat['median']:.3f}  mean={fs_stat['mean']:.3f}  sd={fs_stat['sd']:.3f}")
    print(f"  RS n={rs_stat['n']}  median={rs_stat['median']:.3f}  mean={rs_stat['mean']:.3f}  sd={rs_stat['sd']:.3f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
