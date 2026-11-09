#!/usr/bin/env python3
"""
subset_templates.py  --  template-subsetting tool
-------
Produces a reduced copy of a Kilosort-derived `oss_input/` directory that
contains only the N most useful *preclustered* templates, so that lower-spec
GPUs can run the real-time sorter faster.

The dominant GPU step (`matchingPursuit`) convolves the live stream against
`Wall3.npy`, whose first dimension is the *preclustered* template count
(`unclu_T` in OnlineSpikesV2). Its cost scales ~linearly with `unclu_T`, so
trimming `unclu_T` is the lever that actually buys real-time headroom.

This tool subsets ONLY the preclustered (`unclu_T`) space. It leaves the
final-cluster (`T`) space (templates.npy, templateMap.npy,
cluster_centroids_pca.npy) and all channel/PC-space tensors untouched. That
keeps the engine's load-time invariants intact and avoids the (geometric,
runtime-only) preclustered->cluster remapping problem entirely. Some final
clusters may simply stop receiving spikes; that is the intended graceful
degradation.

You do NOT need to re-run Kilosort to use this. It consumes an existing,
already-cropped `oss_input/` and writes a new sibling directory.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# File classification
#
#   unclu_T  = Wall3.npy.shape[0]      (preclustered template count)
#   T        = templates.npy.shape[0]  (final cluster count)
#   K        = Wall3.npy.shape[1]      (principal components)
#   M        = templates.npy.shape[1]  (samples per template)
#   C        = templates.npy.shape[2]  (channels)
#
# ---------------------------------------------------------------------------

UNCLU_T_AXES = {
    "Wall3.npy": (0,),                            # [unclu_T, K, C]
    "ctc.npy": (0, 1),                            # [unclu_T, unclu_T, 2M+1]
    "ctc_permuted.npy": (0, 1),                   # [unclu_T, unclu_T, 2M+1]; not engine-loaded, kept consistent
    "iU.npy": (0,),                               # [unclu_T]
    "Ucc.npy": (1,),                              # [numNearestChans, unclu_T, K] -- see d_Ucc indexing OnlineSpikesV2.cu:162-164
    "preclustered_template_waveforms.npy": (0,),  # [unclu_T, M, C]
}


# Array used to RANK templates by activity
RANK_SOURCE = "spike_detection_templates.npy"

# Files to copy verbatim, listed for documentation
COPY_VERBATIM_HINT = (
    "templates.npy", "templateMap.npy", "cluster_centroids_pca.npy",
    "whiteningMat.npy", "channelMap.npy", "drift_matrix.npy", "iCC.npy",
    "wPCA.npy", "wPCA_permuted.npy", "hp_filter.npy", "xc.npy", "yc.npy",
    "misc.txt", "ops.npz",
)


def _load_shapes(in_dir: Path):
    """Reads the dimensions from templates.npy and Wall3.npy."""
    templates = np.load(in_dir / "templates.npy")
    wall3 = np.load(in_dir / "Wall3.npy")

    if templates.ndim != 3:
        sys.exit(f"templates.npy expected 3-D [T,M,C], got shape {templates.shape}")
    if wall3.ndim != 3:
        sys.exit(f"Wall3.npy expected 3-D [unclu_T,K,C], got shape {wall3.shape}")

    T, M, C = templates.shape
    unclu_T, K, C_w = wall3.shape
    if C_w != C:
        sys.exit(f"Channel mismatch: templates C={C} but Wall3 C={C_w}")

    return dict(T=int(T), M=int(M), C=int(C), unclu_T=int(unclu_T), K=int(K))


def _choose_keep(in_dir: Path, dims: dict, n: int, metric: str, candidates=None):
    """Return a sorted list of preclustered template indices to keep.

    ``candidates`` optionally restricts selection to a subset of preclustered
    indices (e.g. those with support on a channel range). When given, the
    activity ranking happens only among the candidates and the top-``n`` are
    kept -- this is how the channel filter composes with the activity filter:
    the channel mask narrows the field first, then ``n`` caps the survivors by
    activity. ``n <= 0`` or ``n >= len(candidates)`` keeps every candidate.
    """
    unclu_T = dims["unclu_T"]
    if candidates is None:
        candidates = range(unclu_T)
    candidates = sorted({int(c) for c in candidates})

    # No activity cap, or cap is bigger than the candidate pool: keep them all.
    if not n or n <= 0 or n >= len(candidates):
        label = "all" if len(candidates) == unclu_T else "channel range"
        return list(candidates), label

    if metric == "spikes":
        sp_path = in_dir / RANK_SOURCE
        if sp_path.exists():
            sp = np.load(sp_path).astype(np.int64).ravel()
            if sp.size and int(sp.max()) < dims["T"]:
                sys.exit(
                    f"[abort] {RANK_SOURCE} max index {int(sp.max())} < T ({dims['T']}); "
                    f"it looks like FINAL-cluster space, not preclustered. Ranking would "
                    f"silently drop templates [T, unclu_T). Check the file."
                )
            sp = sp[(sp >= 0) & (sp < unclu_T)]
            counts = np.bincount(sp, minlength=unclu_T)
            order = np.argsort(counts)[::-1]  # find most active first
            ranked = "spike count"
        else:
            print("  [warn] spike_templates.npy not found; falling back to norm")
            metric = "norm"

    if metric == "norm":
        wall3 = np.load(in_dir / "Wall3.npy").reshape(unclu_T, -1)
        energy = np.linalg.norm(wall3, axis=1)
        order = np.argsort(energy)[::-1]  # highest energy first
        ranked = "template L2 norm"

    # Walk the activity-ranked order but keep only candidates, top-n.
    cand_set = set(candidates)
    ranked_keep = [int(i) for i in order if int(i) in cand_set][:n]
    keep = sorted(ranked_keep)
    print(f"  selected {len(keep)} / {len(candidates)} candidate preclustered "
          f"templates by {ranked}")
    return keep, ranked


def parse_channel_range(s):
    """Parse a ``'<lo>-<hi>'`` inclusive probe-channel range string.

    Returns ``(lo, hi)`` ints, or ``None`` if ``s`` is blank/None. Raises
    ``ValueError`` on malformed input. ``lo``/``hi`` are *probe channel
    numbers* (as labelled on the Neuropixels probe), not template C-axis
    indices -- the mapping happens in ``_channels_in_range``.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if "-" not in s:
        raise ValueError(
            f"channel range '{s}' must look like 'lo-hi', e.g. '100-150'")
    lo_s, hi_s = s.split("-", 1)
    lo, hi = int(lo_s.strip()), int(hi_s.strip())
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def _channels_in_range(in_dir: Path, c_lo: int, c_hi: int) -> np.ndarray:
    """Map an inclusive probe-channel range to template C-axis indices.

    Template tensors (``preclustered_template_waveforms.npy``, ``Wall3.npy``,
    ...) carry a channel axis ``C`` whose position ``j`` corresponds to probe
    channel ``channelMap[j]``. So the C-axis indices for probe channels in
    ``[c_lo, c_hi]`` are the positions where ``channelMap`` falls in range.
    """
    in_dir = Path(in_dir)
    cmap_path = in_dir / "channelMap.npy"
    if not cmap_path.exists():
        cmap_path = in_dir / "channel_map.npy"
    if not cmap_path.exists():
        raise RuntimeError(
            "[subset] need channelMap.npy (or channel_map.npy) to map probe "
            "channels onto the template channel axis.")
    cmap = np.load(cmap_path).ravel().astype(np.int64)
    return np.where((cmap >= c_lo) & (cmap <= c_hi))[0]


def _templates_on_channels(in_dir: Path, dims: dict, allowed_chan_idx: np.ndarray,
                           criterion: str = "peak", overlap_frac: float = 0.1):
    """Return preclustered indices whose support lands on the allowed channels.

    Support is read from ``preclustered_template_waveforms.npy`` ([unclu_T,M,C])
    via per-channel peak-to-peak amplitude, falling back to per-channel L2 norm
    of ``Wall3.npy`` ([unclu_T,K,C]) when the waveforms are absent.

    criterion:
      "peak"    -- keep a template iff its dominant (max-amplitude) channel is
                   inside the range. Robust default: insensitive to the small
                   non-zero leakage that whitening/PCA spreads across channels.
      "overlap" -- keep a template iff any channel with amplitude >=
                   overlap_frac * its peak falls inside the range (the literal
                   "has support on these channels" reading, thresholded).
    """
    in_dir = Path(in_dir)
    pre_path = in_dir / "preclustered_template_waveforms.npy"
    if pre_path.exists():
        pre = np.load(pre_path)                       # [unclu_T, M, C]
        amp = pre.max(axis=1) - pre.min(axis=1)       # peak-to-peak [unclu_T, C]
    else:
        w = np.load(in_dir / "Wall3.npy")             # [unclu_T, K, C]
        amp = np.linalg.norm(w, axis=1)               # per-channel energy [unclu_T, C]

    C = amp.shape[1]
    allowed = np.zeros(C, dtype=bool)
    allowed[np.asarray(allowed_chan_idx, dtype=np.int64)] = True

    if criterion == "peak":
        peak = amp.argmax(axis=1)                     # dominant channel per template
        keep = np.where(allowed[peak])[0]
    elif criterion == "overlap":
        thr = overlap_frac * amp.max(axis=1, keepdims=True)
        on = amp >= thr                               # thresholded support mask
        keep = np.where((on & allowed[None, :]).any(axis=1))[0]
    else:
        sys.exit(f"[abort] unknown channel criterion '{criterion}' (peak|overlap)")
    return [int(t) for t in keep]


def _compute_keep(in_dir: Path, dims: dict, n: int, metric: str,
                  channel_range=None, channel_criterion: str = "peak"):
    """Resolve the final kept preclustered-index list and a manifest fragment.

    Composition: the channel mask narrows the candidate pool first, then the
    activity cap ``n`` keeps the most active survivors.
    """
    candidates = None
    chan_meta = None
    if channel_range:
        c_lo, c_hi = channel_range
        chan_idx = _channels_in_range(in_dir, c_lo, c_hi)
        if chan_idx.size == 0:
            raise RuntimeError(
                f"[subset] no probe channels fall in range [{c_lo}, {c_hi}]; "
                f"check the range against channelMap.npy.")
        candidates = _templates_on_channels(in_dir, dims, chan_idx, channel_criterion)
        print(f"  channel range [{c_lo}, {c_hi}] -> {chan_idx.size} probe channels, "
              f"{len(candidates)} / {dims['unclu_T']} preclustered templates with "
              f"support there (criterion={channel_criterion})")
        if len(candidates) == 0:
            raise RuntimeError(
                f"[subset] no preclustered templates have support on channels "
                f"[{c_lo}, {c_hi}]; nothing would survive.")
        chan_meta = {
            "channel_range": [int(c_lo), int(c_hi)],
            "channel_criterion": channel_criterion,
            "probe_channels_in_range": int(chan_idx.size),
            "templates_on_channels": len(candidates),
        }

    keep, ranked = _choose_keep(in_dir, dims, n, metric, candidates=candidates)
    return keep, ranked, chan_meta


def _plan(in_dir: Path, dims: dict):
    """Return list of (filename, action, detail) for dry-run reporting"""
    rows = []
    for name in sorted(os.listdir(in_dir)):
        p = in_dir / name
        if p.is_dir():
            rows.append((name, "skip (dir)", ""))
            continue
        if name in UNCLU_T_AXES:
            arr = np.load(p, allow_pickle=False)
            rows.append((name, "subset", f"shape {arr.shape}, axes {UNCLU_T_AXES[name]}"))
        elif name.endswith(".npy"):
            arr = np.load(p, allow_pickle=False)
            rows.append((name, "copy verbatim", f"shape {arr.shape}"))
        else:
            rows.append((name, "copy verbatim", ""))
    return rows


def _subset_axes(arr: np.ndarray, axes, keep_idx: np.ndarray, unclu_T: int, name: str):
    for ax in axes:
        real_ax = ax if ax >= 0 else arr.ndim + ax
        if arr.shape[real_ax] != unclu_T:
            sys.exit(
                f"[abort] {name}: axis {ax} length {arr.shape[real_ax]} "
                f"!= unclu_T {unclu_T}. Encoded rule is wrong for this data; "
                f"re-run with --dry-run and verify axis order before trusting."
            )
        arr = np.take(arr, keep_idx, axis=real_ax)
    return arr


def subset_oss_input_inplace(oss_dir, n, metric="spikes", allow_below_t=False,
                             channel_range=None, channel_criterion="peak"):
    """Subset the preclustered (unclu_T) tensors of an oss_input dir IN PLACE.

    Called by run_online_spikes.py right after it regenerates `oss_input`.

    Two independent, composable filters select which preclustered templates to
    keep:
      * channel_range=(lo, hi)  keep only templates with support on probe
                                channels [lo, hi] (see _templates_on_channels).
      * n  (max_templates)      keep the N most active survivors.
    They compose channel-first, then activity-cap. Either may be omitted.
    """
    oss_dir = Path(oss_dir)
    dims = _load_shapes(oss_dir)
    unclu_T, T = dims["unclu_T"], dims["T"]

    keep, ranked, chan_meta = _compute_keep(
        oss_dir, dims, n, metric, channel_range, channel_criterion)

    # Nothing dropped -> nothing to do.
    if len(keep) >= unclu_T:
        print(f"[subset] keeping all {unclu_T} preclustered templates (no-op).")
        return {"new_unclu_T": unclu_T, "applied": False, **dims}

    new_unclu_T = len(keep)
    if new_unclu_T < T:
        if not allow_below_t:
            raise RuntimeError(
                f"[subset] resulting template count {new_unclu_T} < final-cluster "
                f"count T ({T}). closestCluster() scans only the first unclu_T "
                f"centroids, so dropping below T misassigns spikes until the "
                f"one-line engine fix ships (loop bound unclu_T -> xs.size()). "
                f"Pass allow_below_t=True only on a rebuilt engine."
            )
    keep_idx = np.asarray(keep, dtype=np.int64)

    for name, axes in UNCLU_T_AXES.items():
        p = oss_dir / name
        if not p.exists():
            continue
        arr = np.load(p, allow_pickle=False)
        arr = _subset_axes(arr, axes, keep_idx, unclu_T, name)
        np.save(p, arr)

    manifest = {
        "original_unclu_T": unclu_T, "new_unclu_T": new_unclu_T,
        "T_unchanged": T, "metric": ranked, "kept_indices": keep,
    }
    if chan_meta is not None:
        manifest["channel_filter"] = chan_meta
    (oss_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))

    _verify(oss_dir, new_unclu_T, dims)
    print(f"[subset] reduced preclustered templates {unclu_T} -> {new_unclu_T} (by {ranked}).")
    return {"new_unclu_T": new_unclu_T, "applied": True, "kept_indices": keep, **dims}


def run(in_dir: Path, out_dir: Path, n: int, metric: str, dry_run: bool,
        allow_below_t: bool, channel_range=None, channel_criterion="peak"):
    dims = _load_shapes(in_dir)
    print(f"Detected: unclu_T={dims['unclu_T']} (preclustered), T={dims['T']} "
          f"(final clusters), K={dims['K']}, M={dims['M']}, C={dims['C']}")

    if dry_run:
        print("\n-- DRY RUN: planned actions --")
        for name, action, detail in _plan(in_dir, dims):
            print(f"  {name:42s} {action:26s} {detail}")
        return

    keep, ranked, chan_meta = _compute_keep(
        in_dir, dims, n, metric, channel_range, channel_criterion)
    unclu_T = dims["unclu_T"]
    new_unclu_T = len(keep)

    if new_unclu_T >= unclu_T:
        sys.exit("Nothing to do: all templates retained. Set --n_templates "
                 "and/or --channel-range to narrow the selection.")
    if new_unclu_T < dims["T"] and not allow_below_t:
        sys.exit(
            f"resulting template count {new_unclu_T} < T ({dims['T']}). "
            f"closestCluster() scans only the\nfirst unclu_T centroids, so "
            f"dropping below T misassigns spikes until the one-line\nengine fix "
            f"ships (see module docstring). Re-run with --allow-below-T to\n"
            f"override (e.g. for a build that already has the fix)."
        )

    keep_idx = np.asarray(keep, dtype=np.int64)

    if out_dir.exists():
        sys.exit(f"--output {out_dir} already exists; refusing to overwrite.")
    out_dir.mkdir(parents=True)

    actions = {}
    for name in sorted(os.listdir(in_dir)):
        src = in_dir / name
        if src.is_dir():
            continue
        dst = out_dir / name

        if name in UNCLU_T_AXES:
            arr = np.load(src, allow_pickle=False)
            arr = _subset_axes(arr, UNCLU_T_AXES[name], keep_idx, unclu_T, name)
            np.save(dst, arr)
            actions[name] = f"subset axes {UNCLU_T_AXES[name]} -> {arr.shape}"
        else:
            shutil.copy2(src, dst)
            actions[name] = "copy verbatim"

    manifest = {
        "source": str(in_dir),
        "original_unclu_T": unclu_T,
        "new_unclu_T": new_unclu_T,
        "T_unchanged": dims["T"],
        "metric": ranked,
        "kept_indices": keep,
        "actions": actions,
    }
    if chan_meta is not None:
        manifest["channel_filter"] = chan_meta
    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))

    _verify(out_dir, new_unclu_T, dims)
    print(f"\nDone. Wrote {new_unclu_T}-template directory to {out_dir}")
    print("Point LSS (--oss_input / KS Output Dir) at it. No Kilosort re-run needed.")


def _verify(out_dir: Path, new_unclu_T: int, dims: dict):
    """Re-check the engine's load-time invariants on the produced directory."""
    print("\n-- Verifying output invariants --")
    checks = {
        "Wall3.npy": ("shape0", new_unclu_T),
        "ctc.npy": ("shape01", new_unclu_T),
        "ctc_permuted.npy": ("shape01", new_unclu_T),
        "iU.npy": ("len", new_unclu_T),
        "Ucc.npy": ("shape1", new_unclu_T),
        "preclustered_template_waveforms.npy": ("shape0", new_unclu_T),
        "templates.npy": ("shape0", dims["T"]),              # unchanged 
        "templateMap.npy": ("len", dims["T"]),               # unchanged
        "cluster_centroids.npy": ("shape0", dims["T"]),      # unchanged 
        "cluster_centroids_pca.npy": ("shape0", dims["T"]),  # unchanged
    }
    ok = True
    for name, (kind, expected) in checks.items():
        p = out_dir / name
        if not p.exists():
            print(f"  [skip] {name} not present in source")
            continue
        arr = np.load(p, allow_pickle=False)
        if kind == "shape0":
            got = arr.shape[0]
        elif kind == "shape01":
            got = arr.shape[0] if arr.shape[0] == arr.shape[1] else -1
        elif kind == "shape1":
            got = arr.shape[1]
        elif kind == "len":
            got = arr.shape[0]
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  {name:42s} {kind:10s} expected {expected:>6}  got {got:>6}  [{status}]")
    if not ok:
        sys.exit("[abort] verification failed; output directory is NOT safe to use.")


def main():
    ap = argparse.ArgumentParser(description="Subset LSS preclustered templates.")
    ap.add_argument("--input", required=True, type=Path, help="source oss_input dir")
    ap.add_argument("--output", type=Path, help="destination dir (required unless --dry-run)")
    ap.add_argument("--n_templates", type=int, default=0, help="number of preclustered templates to keep")
    ap.add_argument("--metric", choices=("spikes", "norm"), default="spikes",
                    help="selection criterion: 'spikes' (most active) or 'norm' (highest energy)")
    ap.add_argument("--channel-range", type=str, default=None,
                    help="probe channel range 'lo-hi' (inclusive); keep only templates "
                         "with support on those channels. Composes with --n_templates "
                         "(channel mask first, then most-active cap).")
    ap.add_argument("--channel-criterion", choices=("peak", "overlap"), default="peak",
                    help="'peak' (dominant channel in range; default) or 'overlap' "
                         "(any thresholded support channel in range)")
    ap.add_argument("--dry-run", action="store_true", help="inspect shapes/actions without writing")
    ap.add_argument("--allow-below-T", action="store_true",
                    help="permit a resulting count < final-cluster count T (only safe "
                         "with the closestCluster engine fix)")
    args = ap.parse_args()

    if not args.input.is_dir():
        sys.exit(f"--input {args.input} is not a directory")
    if not args.dry_run and args.output is None:
        sys.exit("--output is required unless --dry-run")

    channel_range = parse_channel_range(args.channel_range)
    run(args.input, args.output, args.n_templates, args.metric, args.dry_run,
        args.allow_below_T, channel_range=channel_range,
        channel_criterion=args.channel_criterion)


if __name__ == "__main__":
    main()
