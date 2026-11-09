"""SDM processors the launcher knows about -- the one Python file to edit when adding one.

Each Processor below lists the settings an experimenter sees in the launcher's SDM
panel. The launcher builds the widgets from this list, saves the values in
multi_gui_state.json, and hands them to OnlineSpikes.exe itself (as
`--sdm_param key=value ...`); nobody types them. On the C++ side the processor reads
each value by the same key:

    SdmParams p(params.mapSdmParams);          // src/Decoder/SdmParams.h
    m_triggerZ = p.getFloat("trigger_z", 1.0f);

So adding a processor is: the C++ class (ending in REGISTER_SDM_PROCESSOR("name", ...)),
its two lines in OnlineSpikes.vcxproj, and one Processor(...) entry here with the same
name and keys. run_online_spikes.py, main.cpp, Decoder.cpp and inputParameters.h need
no edits.

Settings left blank are not passed, so the C++ default applies.
"""
import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class Param:
    key: str                          # key the C++ processor reads with SdmParams
    label: str                        # text shown in the launcher
    default: str = ""
    choices: Optional[List[str]] = None   # radio buttons instead of a text box
    row: Optional[str] = None         # params with the same row label sit side by side
    when: Optional[Tuple[str, str]] = None  # (key, value): editable only while key == value
    browse: Optional[str] = None      # "file" or "folder": adds a Browse button
    width: int = 10


@dataclass
class Context:
    """What a processor's prepare/button hook gets to work with, for one sorter."""
    oss_dir: Path                     # this sorter's oss_input folder
    bin_ms: str                       # the SDM trigger bin (ms), shared by all processors
    values: Dict[str, str]            # this processor's current settings, key -> value


@dataclass
class Prepared:
    """What a hook returns."""
    ok: bool = True
    message: str = ""
    params: Dict[str, str] = field(default_factory=dict)   # extra settings to pass to the exe
    subset: Optional[List[int]] = None                     # replaces the SDM subset if not None


@dataclass
class Processor:
    name: str                         # must match REGISTER_SDM_PROCESSOR("name", ...) in C++
    help: str
    params: List[Param] = field(default_factory=list)
    # Runs automatically for this sorter right before OnlineSpikes.exe starts.
    prepare: Optional[Callable[[Context], Prepared]] = None
    # Optional extra launcher button: (label, hook). Its subset (if any) fills the Subset box.
    button: Optional[Tuple[str, Callable[[Context], Prepared]]] = None


# ---------------------------------------------------------------------------
# closedloop: FS/RS baseline built from Kilosort's training sort
# ---------------------------------------------------------------------------

def fs_rs_ids_from_labels(oss_dir):
    """Sorted FS+RS template indices from oss_input/rs_fs_labels.csv ([] if absent)."""
    path = Path(oss_dir) / "rs_fs_labels.csv"
    if not path.exists():
        return []
    ids = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("label", "").strip().upper() in ("FS", "RS"):
                try:
                    ids.append(int(row["template_index"]))
                except (ValueError, KeyError):
                    pass
    return sorted(ids)


def run_baseline_build(oss_dir, boundary="200", bin_ms="100"):
    """FS/RS labels + baseline stats for one oss_input folder, straight from Kilosort's
    TRAINING sort (spike_times.npy + spike_templates.npy). Fast (<1 s). Returns (ok, message)."""
    oss_dir = Path(oss_dir)
    missing = [f for f in ("templates.npy", "spike_times.npy", "spike_templates.npy")
               if not (oss_dir / f).exists()]
    if missing:
        return False, (f"Missing {', '.join(missing)} in\n{oss_dir}\n"
                       "Run Kilosort4 (training) first.")
    here = Path(__file__).parent.resolve()
    labels = str(oss_dir / "rs_fs_labels.csv")
    try:
        out1 = subprocess.run([sys.executable, str(here / "waveform_classification.py"), str(oss_dir),
                               "--boundary-us", str(boundary)],
                              capture_output=True, text=True, check=True)
        out2 = subprocess.run([sys.executable, str(here / "closedloop_baseline.py"), str(oss_dir),
                               "--labels", labels, "--bin-ms", str(bin_ms)],
                              capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or e.stdout or str(e))[-1500:]
    return True, (out1.stdout + "\n" + out2.stdout).strip()[-1500:]


def closedloop_prepare(ctx):
    ok, msg = run_baseline_build(ctx.oss_dir, ctx.values.get("boundary_us") or "200", ctx.bin_ms or "100")
    ids = fs_rs_ids_from_labels(ctx.oss_dir)
    return Prepared(ok=ok, message=msg,
                    params={"rs_fs_path": str(ctx.oss_dir / "rs_fs_labels.csv"),
                            "stats_path": str(ctx.oss_dir / "closedloop_stats.txt")},
                    subset=ids or None)


def _fs_rs_grid(mode, prefix, low_label, high_label, default):
    """Four thresholds: FS/RS population x low/high side, live only in `mode`."""
    tag = "median" if mode == "median" else "z-score"
    return [Param(f"{pop}_{prefix}_{side}", lbl, default, row=f"{pop.upper()} ({tag})", when=("mode", mode), width=7)
            for pop in ("fs", "rs") for side, lbl in (("low", low_label), ("high", high_label))]


# ---------------------------------------------------------------------------
# The processors. Order = order in the launcher's dropdown.
# ---------------------------------------------------------------------------

PROCESSORS: Dict[str, Processor] = {p.name: p for p in [
    Processor(
        name="zscore",
        help=("Firing-threshold trigger. Sends a 13-byte UDP packet per bin "
              "(int8 direction, float32 z, uint64 sampleCt). The baseline mean/sd is "
              "learned from the first seconds of the run, or typed in the console."),
        params=[
            Param("trigger_z", "Trigger Z", "1.0"),
            Param("baseline_min_seconds", "Baseline min seconds", "10.0"),
        ],
    ),
    Processor(
        name="logreg",
        help=("Logistic-regression decoder trained on a prior session's spikes and "
              "event labels. Sends the 13-byte UDP packet."),
        params=[
            Param("spikes_file", "Training spikes file", browse="file", width=40),
            Param("event_file", "Training event file", browse="file", width=40),
            Param("work_folder", "Work folder (blank = decoder folder)", browse="folder", width=40),
            Param("window_ms", "Decoder window (ms)", "300"),
        ],
    ),
    Processor(
        name="bincounts",
        help=("Raw per-neuron spike counts over the trigger bin, sent over TCP. "
              "Uses the Subset box (blank = all neurons)."),
    ),
    Processor(
        name="closedloop",
        help=("FS/RS early-release trigger. Splits spikes into FS and RS populations and "
              "sends a 16-byte UDP packet per bin (int32 FS, int32 RS, uint64 sampleCt). "
              "Pick ONE stat; only its grid is editable. Each grid has four independent "
              "knobs, FS and RS x low (-) and high (+), so the populations need not match "
              "and the neutral zone need not be symmetric.\n\n"
              "median: a population reads low when its per-bin value < training median - "
              "low offset, high when > median + high offset (0 = split at the median).\n"
              "z-score: low when z < -low, high when z > +high, z standardized by the "
              "training mean/sd.\n\n"
              "Waveform boundary (us) is the FS/RS trough-to-peak cutoff (200 matches "
              "AnalysisGUI). The baseline is rebuilt automatically at every launch from "
              "Kilosort's training sort, and the Subset is set to the FS+RS neurons; the "
              "button just rebuilds it now, e.g. after changing the boundary or bin size."),
        params=[
            Param("mode", "Stat", "median", choices=["median", "zscore"]),
            *_fs_rs_grid("median", "offset", "low −", "high +", "0"),
            *_fs_rs_grid("zscore", "z", "low −", "high +", "1.0"),
            Param("boundary_us", "Waveform boundary (us)", "200"),   # used by prepare; C++ ignores it
        ],
        prepare=closedloop_prepare,
        button=("Build closed-loop baseline", closedloop_prepare),
    ),
]}


def default_values():
    """{processor: {key: default}} for every processor -- one sorter's fresh settings."""
    return {p.name: {q.key: q.default for q in p.params} for p in PROCESSORS.values()}


def to_exe_args(values, extra=None):
    """key=value strings for `--sdm_param`, skipping blanks (C++ default applies)."""
    merged = dict(values)
    merged.update(extra or {})
    return [f"{k}={v}" for k, v in merged.items() if str(v).strip() != ""]
