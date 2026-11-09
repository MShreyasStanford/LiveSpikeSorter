#!/usr/bin/env python3
"""
waveform_classification.py -- label each FINAL cluster as RS / FS / NONE.

Standalone CLI wrapper around the trough-to-peak method in
AnalysisGUI/analysis_WaveformClassification.py, producing the RS/FS label file
the closed-loop SDM processor (ClosedLoopSdmProcessor) reads at runtime.

Index space: operates on templates.npy in FINAL-cluster space (length T), so
`template_index` here == the template_id LSS emits from closestCluster() and
sends in each spike. That is the same index the label lookup uses live.

Mapping (trough-to-peak width on the primary = max-p2p channel):
    trough-peak-short  -> FS   (narrow spike, fast-spiking interneuron)
    trough-peak-long   -> RS   (broad spike,  regular-spiking pyramidal)
    peak-trough        -> NONE (positive-going / atypical; excluded)

The RS/FS boundary defaults to 200 us, matching AnalysisGUI/analysis_WaveformClassification.py
(threshold_us=200). Override with --boundary-us if your prior analysis differs.

Usage:
    py -3.10 waveform_classification.py <oss_input_dir> [--templates templates.npy]
             [--boundary-us 400] [--fs 30000] [--out rs_fs_labels.csv]
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def classify(templates, fs_hz, boundary_us):
    """Return (labels, widths_us) for each final cluster.

    labels[i] in {'FS','RS','NONE'}; widths_us[i] is trough->peak in us
    (nan when the peak precedes the trough, i.e. 'peak-trough').
    """
    if templates.ndim != 3:
        raise ValueError(f"Expected templates (T, N, C), got {templates.shape}")
    T = templates.shape[0]
    labels, widths = [], []
    for idx in range(T):
        wave = templates[idx]                 # (n_samples, n_channels)
        ptp_by_ch = wave.ptp(axis=0)
        primary = int(np.argmax(ptp_by_ch))
        trace = wave[:, primary]
        peak_idx = int(np.argmax(trace))
        trough_idx = int(np.argmin(trace))
        if peak_idx < trough_idx:
            labels.append('NONE')             # positive-going / atypical
            widths.append(float('nan'))
            continue
        width_us = (peak_idx - trough_idx) / fs_hz * 1e6
        widths.append(width_us)
        labels.append('FS' if width_us <= boundary_us else 'RS')
    return labels, widths


def main():
    ap = argparse.ArgumentParser(description="Label final clusters RS/FS/NONE for LSS closed loop.")
    ap.add_argument("oss_input", help="oss_input/ directory (holds templates.npy)")
    ap.add_argument("--templates", default="templates.npy",
                    help="templates file name inside oss_input (final-cluster space)")
    ap.add_argument("--boundary-us", type=float, default=200.0,
                    help="trough->peak width boundary FS/RS in us (default 200, matching AnalysisGUI)")
    ap.add_argument("--fs", type=float, default=30000.0, help="sampling rate Hz (default 30000)")
    ap.add_argument("--out", default=None,
                    help="output CSV (default <oss_input>/rs_fs_labels.csv)")
    args = ap.parse_args()

    base = Path(args.oss_input)
    templates = np.load(base / args.templates)
    labels, widths = classify(templates, args.fs, args.boundary_us)

    out = Path(args.out) if args.out else base / "rs_fs_labels.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["template_index", "label", "width_us"])
        for i, (lab, wid) in enumerate(zip(labels, widths)):
            w.writerow([i, lab, f"{wid:.1f}" if wid == wid else ""])

    n_fs = labels.count('FS')
    n_rs = labels.count('RS')
    n_none = labels.count('NONE')
    print(f"  {len(labels)} clusters -> FS={n_fs}  RS={n_rs}  NONE={n_none}  (boundary {args.boundary_us:.0f} us)")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
