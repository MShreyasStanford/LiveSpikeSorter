"""
Channel groups for common median referencing (CMR).

The median is subtracted per group instead of across the whole probe. On
Neuropixels, grouping channels that the ADCs sample at the same instant cancels
high-frequency noise better than one global median.

Simultaneously sampled channels follow the ADC sample shifts used by
SpikeInterface (get_neuropixels_sample_shifts) and IBL destriping:
    NP1.0: 12 channels per ADC, channel pairs -> group = (ch // 2) % 12  (12 x 32)
    NP2.0: 16 channels per ADC, channel pairs -> group = (ch // 2) % 16  (16 x 24)

The result is one int32 group id per sorted channel, in Kilosort chanMap order.
That is the row order of the online sorter, which reads it from car_groups.npy.
"""
import numpy as np

CMR_MODES = ["Global", "Auto (from .meta)", "NP1.0 ADC (12x32)", "NP2.0 ADC (16x24)", "Contiguous blocks"]

# imDatPrb_type values (SpikeGLX .meta) by ADC layout
NP1_TYPES = {0, 1020, 1030, 1100, 1120, 1121, 1122, 1123, 1200, 1300}
NP2_TYPES = {21, 24, 2003, 2004, 2013, 2014}


def saved_to_hardware_channels(meta, n_saved):
    """Hardware channel id of each channel saved in the .bin (from snsSaveChanSubset)."""
    subset = (meta or {}).get("snsSaveChanSubset", "all").strip()
    if subset == "all" or not subset:
        return np.arange(n_saved)
    chans = []
    for part in subset.split(","):
        if ":" in part:
            lo, hi = part.split(":")
            chans.extend(range(int(lo), int(hi) + 1))
        else:
            chans.append(int(part))
    return np.asarray(chans[:n_saved])


def resolve_mode(mode, meta):
    """Turn 'Auto' into a concrete mode using imDatPrb_type. Returns (mode, note)."""
    if not mode.startswith("Auto"):
        return mode, ""
    try:
        prb_type = int((meta or {}).get("imDatPrb_type", "-1"))
    except ValueError:
        prb_type = -1
    if prb_type in NP1_TYPES:
        return CMR_MODES[2], f"imDatPrb_type={prb_type} -> NP1.0 ADC groups"
    if prb_type in NP2_TYPES:
        return CMR_MODES[3], f"imDatPrb_type={prb_type} -> NP2.0 ADC groups"
    return CMR_MODES[0], f"WARNING: unknown imDatPrb_type={prb_type}; using global median"


def compute_cmr_groups(mode, block_size, meta, chan_map):
    """
    mode       : one of CMR_MODES
    block_size : channels per group for 'Contiguous blocks'
    meta       : parsed SpikeGLX .meta dict (string values), may be None
    chan_map   : Kilosort chanMap (indices into the saved .bin channels)
    Returns (int32 array of group ids, one per chan_map entry; description string)
    """
    chan_map = np.asarray(chan_map).astype(int).ravel()
    mode, note = resolve_mode(mode, meta)
    n_saved = int((meta or {}).get("nSavedChans", chan_map.max() + 1))
    hw = saved_to_hardware_channels(meta, n_saved)[chan_map]

    if mode == CMR_MODES[2]:
        groups = (hw // 2) % 12
    elif mode == CMR_MODES[3]:
        groups = (hw // 2) % 16
    elif mode == CMR_MODES[4]:
        n = int(block_size)
        if n < 2:
            raise ValueError(f"CMR block size must be >= 2 (got {block_size})")
        groups = hw // n
    else:
        groups = np.zeros_like(hw)

    groups = groups.astype(np.int32)
    sizes = np.bincount(np.unique(groups, return_inverse=True)[1])
    desc = f"{mode}: {sizes.size} group(s), {sizes.min()}-{sizes.max()} channels each"
    if sizes.max() > 1024:
        raise ValueError("CMR: a group has more than 1024 channels (GPU kernel limit)")
    return groups, (note + "; " if note else "") + desc
