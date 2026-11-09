#!/usr/bin/env python3
"""
plot_template_cli.py -- inspect Kilosort templates by index (no hardcoded paths).

Usage:
    py -3.10 plot_template_cli.py <oss_input_dir> --index 4 17 21
    py -3.10 plot_template_cli.py <oss_input_dir> --index 4 --save        # PNG instead of window
    py -3.10 plot_template_cli.py <oss_input_dir> --index 0 4 --metrics   # print trough/peak numbers only

Prints, per template: primary channel, trough-to-peak duration (us), and order
(trough-first vs peak-first) -- the same features the waveform classifier uses,
so you can see whether a 'peak-trough' label is real or a mislabel.
"""
import argparse
from pathlib import Path
import numpy as np

FS = 30000.0  # Hz


def metrics(templates, idx):
    wave = templates[idx]                    # (num_samples, num_channels)
    ptp_by_ch = wave.ptp(axis=0)
    primary = int(np.argmax(ptp_by_ch))
    trace = wave[:, primary]
    trough_i = int(np.argmin(trace))
    peak_i = int(np.argmax(trace))
    t2p_us = abs(peak_i - trough_i) / FS * 1e6
    order = "trough-first" if trough_i < peak_i else "peak-first"
    return primary, trough_i, peak_i, t2p_us, order, ptp_by_ch[primary]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oss_input_dir", help="folder containing templates.npy")
    ap.add_argument("--index", type=int, nargs="+", help="template indices to inspect")
    ap.add_argument("--all", action="store_true",
                    help="dump trough-to-peak metrics for every template to template_metrics.csv")
    ap.add_argument("--save", action="store_true", help="save PNG instead of showing a window")
    ap.add_argument("--metrics", action="store_true", help="print numbers only, no plot")
    args = ap.parse_args()

    templates = np.load(Path(args.oss_input_dir) / "templates.npy")
    T = templates.shape[0]

    if args.all:
        import csv
        out = Path(args.oss_input_dir) / "template_metrics.csv"
        with open(out, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["template", "primary_ch", "order", "trough_to_peak_us", "amp"])
            for idx in range(T):
                primary, ti, pi, t2p, order, amp = metrics(templates, idx)
                wtr.writerow([idx, primary, order, round(t2p, 1), round(float(amp), 3)])
        print(f"Wrote {out} ({T} templates)")
        return

    if not args.index:
        ap.error("give --index <...> or --all")

    for idx in args.index:
        if not (0 <= idx < T):
            print(f"index {idx} out of range 0..{T-1}")
            continue
        primary, ti, pi, t2p, order, amp = metrics(templates, idx)
        print(f"template {idx:>4}: primary_ch={primary:>3}  order={order:<12} "
              f"trough_to_peak={t2p:6.1f} us  amp={amp:8.2f}")

        if args.metrics:
            continue

        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wave = templates[idx]
        support = [j for j in range(wave.shape[1]) if np.any(wave[:, j] != 0)]
        block = wave[:, min(support):max(support) + 1] if support else wave
        plt.figure()
        for i in range(block.shape[1]):
            plt.plot(block[:, i] * 5 + 10 * i)
        plt.title(f"template {idx}  ({order}, {t2p:.0f} us)")
        if args.save:
            out = Path(args.oss_input_dir) / f"template_{idx}.png"
            plt.savefig(out, dpi=110, bbox_inches="tight")
            plt.close()
            print(f"  saved {out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
