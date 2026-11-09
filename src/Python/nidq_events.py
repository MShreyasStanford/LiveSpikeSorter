#!/usr/bin/env python3
"""
nidq_events.py -- extract trial onsets from a recorded SpikeGLX run, offline.

Finds rising edges on a NIDQ digital line and reports them as IMEC AP sample
counts, so trial times line up with a Kilosort/LSS spike file taken from the
same run. Unlike the live path (online_data_viewer_ET.m / dataSocket.cpp) this
edge-detects a single masked bit rather than the composite digital word, and
aligns NI to IMEC by fitting the shared 1 Hz sync square wave instead of
assuming a fixed 25000/30000 sample-rate ratio. On a 100 minute run the fixed
ratio drifts ~30 ms by the end; the sync fit holds to well under a millisecond.

Usage:
    py -3.10 nidq_events.py <run_dir>                        # report + trial CSV
    py -3.10 nidq_events.py <run_dir> --report               # survey lines, write nothing
    py -3.10 nidq_events.py <run_dir> --trial-bit 3
    py -3.10 nidq_events.py <run_dir> --probe imec1
    py -3.10 nidq_events.py <run_dir> --eventfile eventfile.txt   # LSS "<sample> <label>" format
    py -3.10 nidq_events.py <run_dir> --pulses-out all_pulses.csv # every pulse, every bit
    py -3.10 nidq_events.py                                  # omit run_dir -> folder-picker dialog

Always read the per-bit report before trusting --trial-bit: line assignments are
paradigm-dependent. The trial bit is normally the most numerous one, with the
later stages of the trial showing progressively fewer pulses as trials drop out.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

DIGITAL_BITS = 16          # int16 digital word
SCAN_CHUNK = 4_000_000     # frames per memmap read when scanning the whole file
SYNC_WINDOWS = 8           # sync sample windows spread across the run
SYNC_WINDOW_SEC = 10.0


# ---------------------------------------------------------------- meta parsing

def read_meta(path):
    """Parse a SpikeGLX .meta file into a dict (leading '~' stripped from keys)."""
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            meta[key.lstrip("~")] = val
    return meta


def channel_counts(meta, key):
    """snsMnMaXaDw / snsApLfSy -> tuple of ints (already total saved counts)."""
    return tuple(int(x) for x in meta[key].split(","))


def find_run_files(run_dir, probe):
    """Locate the nidq and imec AP bin/meta pairs inside a SpikeGLX run folder."""
    run_dir = Path(run_dir)
    nidq = sorted(run_dir.glob("*.nidq.bin"))
    if not nidq:
        sys.exit(f"No .nidq.bin found in {run_dir}")
    ap = sorted(run_dir.glob(f"*{probe}*/*.{probe}.ap.bin")) or \
         sorted(run_dir.glob(f"*.{probe}.ap.bin"))
    if not ap:
        sys.exit(f"No .{probe}.ap.bin found under {run_dir} (try --probe imec1)")
    return nidq[0], ap[0]


def open_stream(bin_path):
    """Memmap a SpikeGLX .bin as (frames, channels) int16 using its .meta."""
    meta = read_meta(bin_path.with_suffix(".meta"))
    nchan = int(meta["nSavedChans"])
    nframes = bin_path.stat().st_size // (nchan * 2)
    mm = np.memmap(bin_path, dtype=np.int16, mode="r", shape=(nframes, nchan))
    return mm, meta, nframes


# ------------------------------------------------------------- digital scanning

def scan_digital_word(mm, dw_index, nframes, min_width=2):
    """Rising/falling edges of every bit of the digital word, over the whole file.

    Returns {bit: (rises, widths)} with sample indices relative to file start.
    Pulses narrower than min_width samples are dropped as chatter.
    """
    rise_parts = [[] for _ in range(DIGITAL_BITS)]
    fall_parts = [[] for _ in range(DIGITAL_BITS)]
    prev = np.zeros(DIGITAL_BITS, dtype=np.int8)

    for start in range(0, nframes, SCAN_CHUNK):
        word = np.asarray(mm[start:start + SCAN_CHUNK, dw_index]).astype(np.uint16)
        if word.size == 0:
            continue
        for bit in range(DIGITAL_BITS):
            state = ((word >> bit) & 1).astype(np.int8)
            delta = np.diff(np.concatenate(([prev[bit]], state)))
            rise_parts[bit].append(np.flatnonzero(delta > 0) + start)
            fall_parts[bit].append(np.flatnonzero(delta < 0) + start)
            prev[bit] = state[-1]

    out = {}
    for bit in range(DIGITAL_BITS):
        rises = np.concatenate(rise_parts[bit]) if rise_parts[bit] else np.empty(0, int)
        falls = np.concatenate(fall_parts[bit]) if fall_parts[bit] else np.empty(0, int)
        if rises.size == 0:
            continue
        # Pair each rise with the next fall to get pulse widths.
        idx = np.searchsorted(falls, rises, side="right")
        valid = idx < falls.size
        widths = np.full(rises.size, -1, dtype=np.int64)
        widths[valid] = falls[idx[valid]] - rises[valid]
        keep = (widths >= min_width) | (widths < 0)   # keep a final unterminated pulse
        out[bit] = (rises[keep], widths[keep])
    return out


def print_report(pulses, fs_ni):
    """Per-bit survey of the digital word, so a line assignment can be eyeballed."""
    print("\n  NIDQ digital word -- pulses per bit")
    print(f"  {'bit':>3} {'value':>6} {'pulses':>8} {'width(ms)':>10} "
          f"{'first(s)':>10} {'last(s)':>10}")
    for bit in sorted(pulses):
        rises, widths = pulses[bit]
        good = widths[widths >= 0]
        width_ms = np.median(good) / fs_ni * 1000 if good.size else float("nan")
        print(f"  {bit:>3} {1 << bit:>6} {rises.size:>8} {width_ms:>10.2f} "
              f"{rises[0] / fs_ni:>10.1f} {rises[-1] / fs_ni:>10.1f}")
    print()


# ------------------------------------------------------------ sync-based fitting

def find_ni_sync(mm, meta, nframes, xa_offset, xa_count, dw_index, period):
    """Locate the NI sync waveform, as (kind, channel, thresh_or_bit).

    Honours syncNiChan when it looks right, but verifies against the data and
    falls back to a scan, because syncNiChanType is not consistently readable
    across SpikeGLX versions.
    """
    probe_n = min(int(20 * float(meta["niSampRate"])), nframes)
    thresh = float(meta.get("syncNiThresh", 1.1)) / float(meta["niAiRangeMax"]) \
        * float(meta["niMaxInt"])
    named = int(meta.get("syncNiChan", -1))

    analog = np.asarray(mm[:probe_n, xa_offset:xa_offset + xa_count]).astype(np.float32)
    order = ([named] if 0 <= named < xa_count else []) + \
            [c for c in range(xa_count) if c != named]
    for ch in order:
        high = analog[:, ch] > thresh
        if _looks_like_sync(high, float(meta["niSampRate"]), period):
            return "analog", xa_offset + ch, thresh

    word = np.asarray(mm[:probe_n, dw_index]).astype(np.uint16)
    for bit in range(DIGITAL_BITS):
        high = ((word >> bit) & 1).astype(bool)
        if _looks_like_sync(high, float(meta["niSampRate"]), period):
            return "digital", dw_index, bit
    return None, None, None


def find_ap_sync_bit(mm, sy_index, fs_im, nframes, period):
    """Which bit of the imec SY word carries the sync square wave."""
    probe_n = min(int(20 * fs_im), nframes)
    word = np.asarray(mm[:probe_n, sy_index]).astype(np.uint16)
    for bit in range(DIGITAL_BITS):
        high = ((word >> bit) & 1).astype(bool)
        if _looks_like_sync(high, fs_im, period):
            return bit
    return None


def _looks_like_sync(high, fs, period):
    """A sync square wave: ~50% duty, edges spaced by the sync period."""
    if not 0.3 < high.mean() < 0.7:
        return False
    rises = np.flatnonzero(np.diff(high.astype(np.int8)) > 0)
    if rises.size < 3:
        return False
    return abs(np.median(np.diff(rises)) / fs - period) < 0.05 * period


def _edges_ni(mm, kind, chan, thresh_or_bit, lo, hi):
    seg = np.asarray(mm[lo:hi, chan])
    if kind == "analog":
        high = seg.astype(np.float32) > thresh_or_bit
    else:
        high = ((seg.astype(np.uint16) >> int(thresh_or_bit)) & 1).astype(bool)
    return np.flatnonzero(np.diff(high.astype(np.int8)) > 0) + lo


def _edges_ap(mm, sy_index, bit, lo, hi):
    word = np.asarray(mm[lo:hi, sy_index]).astype(np.uint16)
    state = ((word >> bit) & 1).astype(np.int8)
    return np.flatnonzero(np.diff(state) > 0) + lo


def fit_ni_to_im(ni_mm, ni_meta, ni_frames, ap_mm, ap_meta, ap_frames,
                 ni_sync, ap_sync_bit, sy_index):
    """Least-squares map from NI sample index to IMEC AP sample index.

    Matches individual sync pulses between the two streams across windows spread
    over the run, then fits a line. Returns (slope, intercept, max_residual).
    """
    fs_ni = float(ni_meta["niSampRate"])
    fs_im = float(ap_meta["imSampRate"])
    first_ni = float(ni_meta.get("firstSample", 0))
    first_im = float(ap_meta.get("firstSample", 0))
    period = float(ni_meta.get("syncSourcePeriod", 1))
    kind, chan, thresh_or_bit = ni_sync

    # Streams are counted from their own start, so the meta model has to undo
    # the difference in firstSample as well as the rate ratio.
    def predict(ni_idx):
        return (first_ni + ni_idx) * fs_im / fs_ni - first_im

    span = min(ni_frames / fs_ni, ap_frames / fs_im) - SYNC_WINDOW_SEC
    starts = np.linspace(SYNC_WINDOW_SEC, span, SYNC_WINDOWS)

    ni_pts, ap_pts = [], []
    for t0 in starts:
        lo_ni, hi_ni = int(t0 * fs_ni), int((t0 + SYNC_WINDOW_SEC) * fs_ni)
        lo_ap, hi_ap = int(t0 * fs_im), int((t0 + SYNC_WINDOW_SEC) * fs_im)
        en = _edges_ni(ni_mm, kind, chan, thresh_or_bit, lo_ni, min(hi_ni, ni_frames))
        ea = _edges_ap(ap_mm, sy_index, ap_sync_bit, lo_ap, min(hi_ap, ap_frames))
        if en.size == 0 or ea.size == 0:
            continue
        for edge in en:
            pred = predict(edge)
            nearest = ea[np.argmin(np.abs(ea - pred))]
            # Unambiguous only if we are well inside half a sync period.
            if abs(nearest - pred) < 0.4 * period * fs_im:
                ni_pts.append(edge)
                ap_pts.append(nearest)

    if len(ni_pts) < 4:
        return None
    ni_pts = np.asarray(ni_pts, dtype=float)
    ap_pts = np.asarray(ap_pts, dtype=float)

    # Second pass: re-match using the fitted model, which removes any pulse that
    # the coarser meta model paired with a neighbour on a very long recording.
    slope, intercept = np.polyfit(ni_pts, ap_pts, 1)
    resid = ap_pts - (slope * ni_pts + intercept)
    keep = np.abs(resid) < 0.4 * period * fs_im
    slope, intercept = np.polyfit(ni_pts[keep], ap_pts[keep], 1)
    resid = ap_pts[keep] - (slope * ni_pts[keep] + intercept)
    return slope, intercept, float(np.abs(resid).max()), int(keep.sum())


# ------------------------------------------------------------------------ main

def pick_folder(title):
    """Blocking folder-select dialog. Exits with a clear message if unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        sys.exit("No folder given and tkinter is unavailable for a picker dialog -- "
                  "pass the path directly as a command-line argument instead.")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title=title, mustexist=True)
    root.destroy()
    if not path:
        sys.exit("No folder selected.")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Extract trial onsets from a recorded SpikeGLX run.")
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="SpikeGLX run folder (holds *.nidq.bin and *_imec0/); "
                         "omit to pick it with a folder dialog")
    ap.add_argument("--trial-bit", type=int, default=1,
                    help="digital word bit carrying trial onset (default 1)")
    ap.add_argument("--probe", default="imec0", help="which probe to align to (default imec0)")
    ap.add_argument("--out", help="trial CSV path (default <run>_trial_events.csv)")
    ap.add_argument("--eventfile", help="also write LSS '<imec_sample> <label>' event file")
    ap.add_argument("--label", type=int, default=0, help="label written to --eventfile (default 0)")
    ap.add_argument("--pulses-out", help="write every pulse of every bit to this CSV")
    ap.add_argument("--report", action="store_true", help="survey the lines and exit")
    ap.add_argument("--min-width", type=int, default=2,
                    help="reject pulses narrower than this many NI samples (default 2)")
    args = ap.parse_args()

    run_dir = args.run_dir or pick_folder("Select SpikeGLX run folder (contains *.nidq.bin)")

    ni_path, ap_path = find_run_files(run_dir, args.probe)
    ni_mm, ni_meta, ni_frames = open_stream(ni_path)
    fs_ni = float(ni_meta["niSampRate"])

    mn, ma, xa, dw = channel_counts(ni_meta, "snsMnMaXaDw")
    if dw < 1:
        sys.exit("This run saved no NIDQ digital word (snsMnMaXaDw ends in 0).")
    dw_index = mn + ma + xa

    print(f"  nidq : {ni_path.name}")
    print(f"         {ni_frames} frames, {ni_frames / fs_ni:.1f} s, {fs_ni} Hz, "
          f"digital word at channel {dw_index}")

    pulses = scan_digital_word(ni_mm, dw_index, ni_frames, args.min_width)
    if not pulses:
        sys.exit("No digital pulses found anywhere in the word.")
    print_report(pulses, fs_ni)

    if args.pulses_out:
        rows = sorted((int(s), b) for b in pulses for s in pulses[b][0])
        with open(args.pulses_out, "w") as fh:
            fh.write("ni_sample,bit,value,t_sec\n")
            for sample, bit in rows:
                fh.write(f"{sample},{bit},{1 << bit},{sample / fs_ni:.6f}\n")
        print(f"  wrote {len(rows)} pulses -> {args.pulses_out}")

    if args.report:
        return

    if args.trial_bit not in pulses:
        sys.exit(f"Bit {args.trial_bit} has no pulses. Pick one from the report above.")
    trials = pulses[args.trial_bit][0]

    # --- align NI to IMEC using the shared sync square wave -------------------
    ap_mm, ap_meta, ap_frames = open_stream(ap_path)
    fs_im = float(ap_meta["imSampRate"])
    n_ap, n_lf, n_sy = channel_counts(ap_meta, "snsApLfSy")
    if n_sy < 1:
        sys.exit("The AP file has no SY sync channel; cannot align by sync.")
    sy_index = n_ap + n_lf
    period = float(ni_meta.get("syncSourcePeriod", 1))

    print(f"  imec : {ap_path.name}")
    print(f"         {ap_frames} frames, {ap_frames / fs_im:.1f} s, {fs_im:.4f} Hz, "
          f"SY at channel {sy_index}")

    ni_sync = find_ni_sync(ni_mm, ni_meta, ni_frames, mn + ma, xa, dw_index, period)
    ap_sync_bit = find_ap_sync_bit(ap_mm, sy_index, fs_im, ap_frames, period)

    fit = None
    if ni_sync[0] is not None and ap_sync_bit is not None:
        where = f"analog ch {ni_sync[1]}" if ni_sync[0] == "analog" else f"digital bit {ni_sync[1]}"
        print(f"  sync : NI on {where}, IMEC on SY bit {ap_sync_bit}")
        fit = fit_ni_to_im(ni_mm, ni_meta, ni_frames, ap_mm, ap_meta, ap_frames,
                           ni_sync, ap_sync_bit, sy_index)

    if fit:
        slope, intercept, max_resid, npts = fit
        print(f"         im = {slope:.9f} * ni + {intercept:.2f}   "
              f"({npts} pulses, max residual {max_resid:.1f} samples = "
              f"{max_resid / fs_im * 1e6:.0f} us)")
    else:
        first_ni = float(ni_meta.get("firstSample", 0))
        first_im = float(ap_meta.get("firstSample", 0))
        slope = fs_im / fs_ni
        intercept = first_ni * slope - first_im
        print("  WARNING: could not fit the sync waveform; falling back to the "
              "meta sample-rate ratio.\n"
              "           Expect a few ms of error by the end of a long run.")

    imec_samples = np.rint(slope * trials.astype(float) + intercept).astype(np.int64)
    in_range = (imec_samples >= 0) & (imec_samples < ap_frames)
    if not in_range.all():
        print(f"  note : {(~in_range).sum()} trial(s) fall outside the AP file and "
              f"were dropped.")
        trials, imec_samples = trials[in_range], imec_samples[in_range]

    out_path = Path(args.out) if args.out else \
        Path(run_dir) / f"{ni_path.name.split('.')[0]}_trial_events.csv"
    with open(out_path, "w") as fh:
        fh.write("trial,trial_start_ni_sample,trial_start_imec_sample,t_sec\n")
        for i, (ni_s, im_s) in enumerate(zip(trials, imec_samples)):
            fh.write(f"{i},{ni_s},{im_s},{im_s / fs_im:.6f}\n")
    print(f"\n  {len(trials)} trials on bit {args.trial_bit} "
          f"(value {1 << args.trial_bit}) -> {out_path}")

    if args.eventfile:
        with open(args.eventfile, "w") as fh:
            for im_s in imec_samples:
                fh.write(f"{im_s} {args.label}\n")
        print(f"  LSS event file -> {args.eventfile}")


if __name__ == "__main__":
    main()
