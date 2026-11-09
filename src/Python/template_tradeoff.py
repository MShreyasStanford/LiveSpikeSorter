#!/usr/bin/env python3
"""
template_tradeoff.py -- Measure the latency vs. accuracy tradeoff of running LSS
with reduced template counts, and plot it.

CONCEPT
-------
Offline replay is deterministic: the same .ap.bin is streamed every run, so a
run with N templates differs from the full-template run ONLY because of template
reduction (provided you disable skipping -- see PROTOCOL). That makes the
full-template run a clean reference for asking "what did dropping templates cost?"

Two distinct accuracy axes, which this tool separates:
  * FIDELITY  -- for neurons that are KEPT, how faithfully are their spike trains
                 reproduced vs. the reference (spike-matching score + firing-rate
                 correlation).
  * COVERAGE  -- what fraction of the reference's neurons still produce spikes
                 (yield).

Latency axis: per-batch processing time. In "never skip" runs the processTime
distribution is the true compute cost; the fraction of batches exceeding the
50 ms real-time deadline predicts the skip pressure you'd see in a real-time run.

PROTOCOL (run these once, save the outputs)
-------------------------------------------
For the reference and each N, run LSS on the SAME replay with skipping disabled
(input GUI: "Time behind" > 100000 = never skip) so no data is dropped:
  1. Reference: Max templates = 0 (all templates). Save cuda_output/spikeOutput.txt
     as spikeOutput_full.txt.
  2. For each N in e.g. {600,500,400,300,200,100}: set Max templates = N, run,
     save spikeOutput_N{N}.txt. Note the median/mean processing time from the GUI

USAGE
-----
  py -3.10 template_tradeoff.py `
  --reference spikeOutput_full.txt `
  --run 753:spikeOutput_full.txt:<lat_full> `
  --run 600:spikeOutput_N600.txt:11.6 `
  --run 500:spikeOutput_N500.txt:9.2 `
  --run 400:spikeOutput_N400.txt:<lat_400> `
  --run 300:spikeOutput_N300.txt:8.4 `
  --run 200:spikeOutput_N200.txt:<lat_200> `
  --fs 30000 --out tradeoff


Each --run is  N:spikeOutputPath[:latency]  where latency is a number or a path 
to a processTime log. The reference itself may be passed as a run (N = full 
count) to anchor the curve.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_spikes(path):
    """Load a spikeOutput.txt: columns sample_time, template_id, amplitude[, y].
    Returns (times[int64], templates[int64])."""
    times, temps = [], []
    with open(path) as f:
        for line in f:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                times.append(int(parts[0]))
                temps.append(int(parts[1]))
            except ValueError:
                continue
    return np.asarray(times, dtype=np.int64), np.asarray(temps, dtype=np.int64)


def by_neuron(times, temps):
    """dict: neuron_id -> sorted spike-time array."""
    out = {}
    order = np.argsort(times, kind="stable")
    times, temps = times[order], temps[order]
    for nid in np.unique(temps):
        out[int(nid)] = times[temps == nid]
    return out


def match_train(ref_t, test_t, tol):
    """Returns (precision, recall, f1, match_score)."""
    n_ref, n_test = len(ref_t), len(test_t)
    if n_ref == 0 or n_test == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    ref_t = np.sort(ref_t)
    idx = np.searchsorted(ref_t, test_t)
    matched_ref = np.zeros(n_ref, dtype=bool)
    tp = 0
    for k, t in enumerate(test_t):
        j = idx[k]
        best, bestd = -1, tol + 1
        for cand in (j - 1, j):
            if 0 <= cand < n_ref and not matched_ref[cand]:
                d = abs(int(ref_t[cand]) - int(t))
                if d <= tol and d < bestd:
                    best, bestd = cand, d
        if best >= 0:
            matched_ref[best] = True
            tp += 1
    precision = tp / n_test          # 1 - false-positive rate
    recall = tp / n_ref              # 1 - false-miss rate
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    match = precision + recall - 1.0  # paper's 1 - FP - FM
    return precision, recall, f1, match


def rate_corr(ref_t, test_t, binsize, tmax):
    """Pearson correlation of binned firing rates (PSTH-like)."""
    if len(ref_t) == 0 or len(test_t) == 0:
        return np.nan
    edges = np.arange(0, tmax + binsize, binsize)
    a, _ = np.histogram(ref_t, bins=edges)
    b, _ = np.histogram(test_t, bins=edges)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def refractory_rate(spike_times, refractory_samples):
    """Fraction of inter-spike intervals shorter than the refractory period."""
    if len(spike_times) < 2:
        return np.nan
    isi = np.diff(np.sort(spike_times))
    return float(np.mean(isi < refractory_samples))


def latency_stats(spec):
    """input number (median ms) or a path to a processTime log."""
    try:
        return {"median": float(spec), "mean": float(spec),
                "p95": np.nan, "frac_over_50": np.nan}
    except ValueError:
        pass
    vals = np.loadtxt(spec)
    vals = np.atleast_1d(vals).astype(float)
    return {"median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "p95": float(np.percentile(vals, 95)),
            "frac_over_50": float(np.mean(vals > 50.0))}


def _coarse_hist(times, edges):
    return np.histogram(times, bins=edges)[0].astype(float)


def _pearson_after_shift(ref_t, test_t, shift, coarse):
    """Population firing-rate Pearson after shifting test by `shift` samples."""
    t = test_t + shift
    lo = int(min(ref_t.min(), t.min()))
    hi = int(max(ref_t.max(), t.max()))
    edges = np.arange(lo, hi + coarse, coarse)
    a, b = _coarse_hist(ref_t, edges), _coarse_hist(t, edges)
    if a.std() == 0 or b.std() == 0:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


def estimate_global_lag(ref_t, test_t, coarse):
    """Find the constant sample offset to ADD to test_t so it best aligns to
    ref_t. Magnitude from cross-correlation of coarse population rates; sign
    chosen by directly scoring {0, +lag, -lag} so the numpy convention can't
    bite us. Returns (lag_samples, population_pearson_at_that_lag)."""
    if len(ref_t) == 0 or len(test_t) == 0:
        return 0, 0.0
    lo = int(min(ref_t.min(), test_t.min()))
    hi = int(max(ref_t.max(), test_t.max()))
    edges = np.arange(lo, hi + coarse, coarse)
    a, b = _coarse_hist(ref_t, edges), _coarse_hist(test_t, edges)
    a -= a.mean(); b -= b.mean()
    if a.std() == 0 or b.std() == 0:
        return 0, 0.0
    corr = np.correlate(a, b, mode="full")
    lag_bins = abs(int(np.argmax(corr) - (len(b) - 1)))
    cands = [0, lag_bins * coarse, -lag_bins * coarse]
    scored = [(_pearson_after_shift(ref_t, test_t, s, coarse), s) for s in cands]
    best_score, best_shift = max(scored)

    # --- refine to ~sample precision around the coarse estimate ---
    # A coarse (100 ms) lag of 0 can still hide a sub-bin offset of tens of ms,
    # which is huge vs the 0.5 ms match tolerance and ruins per-spike fidelity
    # even when the data is identical. Refine in two finer passes.
    step1 = max(1, coarse // 100)          # ~1 ms grid over +/- one coarse bin
    best_shift, best_score = _grid_refine(ref_t, test_t, best_shift,
                                          span=coarse, step=step1, bin_=step1)
    step2 = max(1, coarse // 500)          # ~0.2 ms grid over +/- one 1 ms step
    best_shift, best_score = _grid_refine(ref_t, test_t, best_shift,
                                          span=step1, step=step2, bin_=max(1, step2 * 3))
    return int(best_shift), float(best_score)


def _grid_refine(ref_t, test_t, center, span, step, bin_):
    """Grid-search the shift in [center-span, center+span] maximizing the
    population Pearson at resolution `bin_`. Returns (best_shift, best_score)."""
    best_shift, best_score = center, _pearson_after_shift(ref_t, test_t, center, bin_)
    for s in range(center - span, center + span + 1, step):
        sc = _pearson_after_shift(ref_t, test_t, s, bin_)
        if sc > best_score:
            best_score, best_shift = sc, s
    return best_shift, best_score


def _clip(times, lo, hi):
    """Keep spikes within [lo, hi] and re-base to start at 0."""
    m = (times >= lo) & (times <= hi)
    return times[m] - lo


def analyze(ref_path, runs, fs, tol_ms, bin_ms, refractory_ms, min_spikes, coarse_ms):
    tol = int(round(tol_ms * fs / 1000.0))
    binsize = int(round(bin_ms * fs / 1000.0))
    refr = int(round(refractory_ms * fs / 1000.0))
    coarse = int(round(coarse_ms * fs / 1000.0))

    rt, rtmp = load_spikes(ref_path)
    ref_all = by_neuron(rt, rtmp)
    print(f"Reference: {len(rt)} spikes, span "
          f"[{int(rt.min())}, {int(rt.max())}] samples")

    rows = []
    for N, sp_path, lat_spec in runs:
        tt, ttmp = load_spikes(sp_path)

        # --- align test to reference (handles differing recordingOffset / span) ---
        lag, lag_score = estimate_global_lag(rt, tt, coarse)
        tt_shift = tt + lag
        lo = int(max(rt.min(), tt_shift.min()))
        hi = int(min(rt.max(), tt_shift.max()))
        overlap_sec = max(0, hi - lo) / fs
        if hi <= lo:
            print(f"N={N}: NO overlap after alignment (lag={lag}) -- runs cover "
                  f"disjoint windows; cannot compare. Skipping fidelity.")
        # re-base both to the overlap window
        ref = {n: _clip(t, lo, hi) for n, t in ref_all.items()}
        test = {n: _clip(t + lag, lo, hi)
                for n, t in by_neuron(tt, ttmp).items()}
        tmax = max(1, hi - lo)

        ref_active = {n for n, t in ref.items() if len(t) >= min_spikes}
        test_active = {n for n, t in test.items() if len(t) >= min_spikes}

        precs, recs, f1s, matches, corrs, refrs = [], [], [], [], [], []
        for nid in sorted(ref_active):
            test_t = test.get(nid, np.empty(0, dtype=np.int64))
            p, r, f1, m = match_train(ref[nid], test_t, tol)
            c = rate_corr(ref[nid], test_t, binsize, tmax)
            for lst, v in ((precs, p), (recs, r), (f1s, f1), (matches, m), (corrs, c)):
                if not np.isnan(v):
                    lst.append(v)
        for nid in test_active:
            refrs.append(refractory_rate(test[nid], refr))

        lat = latency_stats(lat_spec)
        coverage = len(test_active & ref_active) / max(1, len(ref_active))
        if lag_score < 0.3:
            print(f"N={N}: WARNING low alignment confidence (pop r={lag_score:.2f}); "
                  f"fidelity numbers may be unreliable -- check the run protocol.")
        row = {
            "N": N,
            "total_spikes": len(tt),
            "global_lag_samples": lag,
            "align_pop_r": round(lag_score, 3),
            "overlap_sec": round(overlap_sec, 1),
            "yield": len(test_active),
            "coverage": coverage,
            "median_match": float(np.nanmedian(matches)) if matches else np.nan,
            "median_recall": float(np.nanmedian(recs)) if recs else np.nan,
            "median_precision": float(np.nanmedian(precs)) if precs else np.nan,
            "median_f1": float(np.nanmedian(f1s)) if f1s else np.nan,
            "median_rate_corr": float(np.nanmedian(corrs)) if corrs else np.nan,
            "median_refractory": float(np.nanmedian(refrs)) if refrs else np.nan,
            "latency_median": lat["median"],
            "latency_p95": lat["p95"],
            "frac_over_50": lat["frac_over_50"],
        }
        rows.append(row)
        print(f"N={N:>5}: lag={lag:>8}smp pop_r={lag_score:4.2f} overlap={overlap_sec:6.1f}s "
              f"yield={row['yield']:>4} cov={coverage:4.2f} "
              f"f1={row['median_f1']:.3f} rate_r={row['median_rate_corr']:.3f} "
              f"lat={lat['median']:.1f}ms")
    rows.sort(key=lambda r: r["N"])
    return rows


def plot(rows, out_prefix):
    Ns = [r["N"] for r in rows]
    lat = [r["latency_median"] for r in rows]
    match = [r["median_match"] for r in rows]
    f1 = [r["median_f1"] for r in rows]
    corr = [r["median_rate_corr"] for r in rows]
    cov = [r["coverage"] for r in rows]
    over = [r["frac_over_50"] for r in rows]

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) The headline Pareto curve: latency vs fidelity, annotated by N.
    a = ax[0, 0]
    a.plot(lat, f1, "o-", color="C0", label="kept-neuron F1")
    a.plot(lat, corr, "s--", color="C1", label="firing-rate corr")
    for x, y, n in zip(lat, f1, Ns):
        a.annotate(f"N={n}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    a.set_xlabel("median batch latency (ms)")
    a.set_ylabel("fidelity (vs full-template reference)")
    a.set_title("Latency vs. accuracy tradeoff")
    a.axvline(50, color="r", ls=":", lw=1, label="50 ms deadline")
    a.legend(); a.grid(alpha=0.3)

    # (0,1) Accuracy & coverage vs N.
    a = ax[0, 1]
    a.plot(Ns, f1, "o-", label="kept-neuron F1")
    a.plot(Ns, corr, "s--", label="firing-rate corr")
    a.plot(Ns, cov, "^-", color="C2", label="coverage (yield frac)")
    a.set_xlabel("template count N"); a.set_ylabel("score / fraction")
    a.set_title("Accuracy & coverage vs N"); a.legend(); a.grid(alpha=0.3)

    # (1,0) Latency vs N, with deadline.
    a = ax[1, 0]
    a.plot(Ns, lat, "o-", color="C3")
    a.axhline(50, color="r", ls=":", lw=1, label="50 ms deadline")
    a.set_xlabel("template count N"); a.set_ylabel("median batch latency (ms)")
    a.set_title("Latency vs N"); a.legend(); a.grid(alpha=0.3)

    # (1,1) Predicted real-time effective accuracy = fidelity * coverage * (1 - skip).
    a = ax[1, 1]
    eff = []
    for r in rows:
        skip = r["frac_over_50"] if not np.isnan(r["frac_over_50"]) else 0.0
        base = r["median_f1"] if not np.isnan(r["median_f1"]) else 0.0
        eff.append(base * r["coverage"] * (1.0 - skip))
    a.plot(Ns, eff, "o-", color="C4")
    a.set_xlabel("template count N")
    a.set_ylabel("predicted real-time effective accuracy")
    a.set_title("Sweet spot:  F1 x coverage x (1 - skip frac)")
    a.grid(alpha=0.3)
    if any(not np.isnan(r["frac_over_50"]) for r in rows):
        best = Ns[int(np.nanargmax(eff))]
        a.axvline(best, color="g", ls="--", label=f"peak N={best}")
        a.legend()

    fig.tight_layout()
    png = f"{out_prefix}.png"
    fig.savefig(png, dpi=130)
    print(f"\nWrote plot -> {png}")

    csv = f"{out_prefix}.csv"
    keys = list(rows[0].keys())
    with open(csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"Wrote metrics -> {csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="full-template spikeOutput.txt")
    ap.add_argument("--run", action="append", required=True, metavar="N:PATH[:LATENCY]",
                    help="a reduced run; LATENCY is median ms or a processTime log path")
    ap.add_argument("--fs", type=float, default=30000.0, help="sample rate (Hz)")
    ap.add_argument("--tol-ms", type=float, default=0.5, help="spike match tolerance (ms)")
    ap.add_argument("--bin-ms", type=float, default=10.0, help="firing-rate bin (ms)")
    ap.add_argument("--refractory-ms", type=float, default=1.5, help="refractory window (ms)")
    ap.add_argument("--coarse-ms", type=float, default=100.0, help="bin for global-lag alignment (ms)")
    ap.add_argument("--min-spikes", type=int, default=10, help="min spikes to count a neuron active")
    ap.add_argument("--out", default="template_tradeoff", help="output prefix")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        parts = spec.split(":")
        if len(parts) < 2:
            sys.exit(f"--run must be N:PATH[:LATENCY], got {spec!r}")
        N = int(parts[0])
        path = parts[1]
        lat = parts[2] if len(parts) > 2 else "nan"
        runs.append((N, path, lat))

    rows = analyze(args.reference, runs, args.fs, args.tol_ms, args.bin_ms,
                   args.refractory_ms, args.min_spikes, args.coarse_ms)
    if rows:
        plot(rows, args.out)


if __name__ == "__main__":
    main()
