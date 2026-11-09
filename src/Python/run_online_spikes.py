import sys
import pathlib
from pathlib import Path
from crop_methods import crop_kilosort_output, parse_bin_meta_file
from subset_templates import subset_oss_input_inplace, parse_channel_range
from cmr_groups import CMR_MODES, compute_cmr_groups
from sdm_processors import PROCESSORS, Context, default_values, to_exe_args
import numpy as np
import subprocess
import time
import os
import shlex
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import json

# -------------------------------
# Persistence config
# -------------------------------
STATE_FILE = Path(__file__).parent / "multi_gui_state.json"
root = tk.Tk()
root.title("Live Spike Sorter \u2014 Configuration")

# -------------------------------
# GUI: Global Config
# -------------------------------

# Number of sorters
num_sorters_var = tk.IntVar(value=1)
spin = tk.Spinbox(root, from_=1, to=16, textvariable=num_sorters_var, width=5, command=lambda: update_tabs())
tk.Label(root, text="Number of sorters:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
spin.grid(row=0, column=1, padx=5, pady=5, sticky="w")

# Global Data Acquisition parameters (shared across all sorters: one SpikeGLX
# stream connection for the whole process)
sglx_host_var = tk.StringVar(value="127.0.0.1")
sglx_port_var = tk.StringVar(value="4142")

# -------------------------------
# GUI: Per-Sorter Level Config
# -------------------------------

daq_frame = tk.Frame(root, borderwidth=1, relief="groove")
daq_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="w")

tk.Label(daq_frame, text="Data Acquisition (shared)", font=("TkDefaultFont", 9, "bold")).grid(
    row=0, column=0, columnspan=4, padx=5, pady=(5, 2), sticky="w"
)

tk.Label(daq_frame, text="SGLX Host:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
tk.Entry(daq_frame, textvariable=sglx_host_var, width=15).grid(row=1, column=1, sticky="w")
tk.Button(daq_frame, text="?", command=lambda: show_hint("SGLX_HOST"), width=3).grid(row=1, column=3)

tk.Label(daq_frame, text="SGLX Port:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
tk.Entry(daq_frame, textvariable=sglx_port_var, width=15).grid(row=2, column=1, sticky="w")
tk.Button(daq_frame, text="?", command=lambda: show_hint("SGLX_PORT"), width=3).grid(row=2, column=3)

# Drift-correction parameters (shared across all sorters)
drift_enabled_var = tk.BooleanVar(value=True)
drift_window_var = tk.StringVar(value="10")
drift_max_shift_var = tk.StringVar(value="50")
drift_retrain_threshold_var = tk.StringVar(value="50")

drift_frame = tk.Frame(root, borderwidth=1, relief="groove")
drift_frame.grid(row=1, column=4, columnspan=4, padx=5, pady=5, sticky="w")

tk.Label(drift_frame, text="Drift Correction (shared)", font=("TkDefaultFont", 9, "bold")).grid(
    row=0, column=0, columnspan=4, padx=5, pady=(5, 2), sticky="w"
)

tk.Checkbutton(drift_frame, text="Real-time drift correction", variable=drift_enabled_var).grid(
    row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w"
)
tk.Button(drift_frame, text="?", command=lambda: show_hint("DRIFT_ENABLED"), width=3).grid(row=1, column=3)

tk.Label(drift_frame, text="Window (s):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
tk.Entry(drift_frame, textvariable=drift_window_var, width=10).grid(row=2, column=1, sticky="w")
tk.Button(drift_frame, text="?", command=lambda: show_hint("DRIFT_WINDOW"), width=3).grid(row=2, column=3)

tk.Label(drift_frame, text="Max shift (um):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
tk.Entry(drift_frame, textvariable=drift_max_shift_var, width=10).grid(row=3, column=1, sticky="w")
tk.Button(drift_frame, text="?", command=lambda: show_hint("DRIFT_MAX_SHIFT"), width=3).grid(row=3, column=3)

tk.Label(drift_frame, text="Retrain threshold (um):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
tk.Entry(drift_frame, textvariable=drift_retrain_threshold_var, width=10).grid(row=4, column=1, sticky="w")
tk.Button(drift_frame, text="?", command=lambda: show_hint("DRIFT_RETRAIN_THRESHOLD"), width=3).grid(row=4, column=3)

# Public spike stream to external processes (shared across all sorters)
spike_stream_enabled_var = tk.BooleanVar(value=False)
spike_stream_addr_var = tk.StringVar(value="127.0.0.1:9100")

stream_frame = tk.Frame(root, borderwidth=1, relief="groove")
stream_frame.grid(row=1, column=8, columnspan=4, padx=5, pady=5, sticky="nw")

tk.Label(stream_frame, text="Spike Stream (shared)", font=("TkDefaultFont", 9, "bold")).grid(
    row=0, column=0, columnspan=4, padx=5, pady=(5, 2), sticky="w"
)
tk.Checkbutton(stream_frame, text="Stream spikes over UDP", variable=spike_stream_enabled_var).grid(
    row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w"
)
tk.Button(stream_frame, text="?", command=lambda: show_hint("SPIKE_STREAM"), width=3).grid(row=1, column=3)
tk.Label(stream_frame, text="host:port:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
tk.Entry(stream_frame, textvariable=spike_stream_addr_var, width=18).grid(row=2, column=1, sticky="w")

# Backlog / skip handling (shared across all sorters)
max_lag_var = tk.StringVar(value="0")
small_skip_var = tk.BooleanVar(value=False)

latency_frame = tk.Frame(root, borderwidth=1, relief="groove")
latency_frame.grid(row=1, column=12, columnspan=4, padx=5, pady=5, sticky="nw")

tk.Label(latency_frame, text="Latency (shared)", font=("TkDefaultFont", 9, "bold")).grid(
    row=0, column=0, columnspan=4, padx=5, pady=(5, 2), sticky="w"
)
tk.Label(latency_frame, text="Max lag (ms, 0 = skip):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
tk.Entry(latency_frame, textvariable=max_lag_var, width=10).grid(row=1, column=1, sticky="w")
tk.Button(latency_frame, text="?", command=lambda: show_hint("MAX_LAG"), width=3).grid(row=1, column=3)
tk.Checkbutton(latency_frame, text="Small skip", variable=small_skip_var).grid(
    row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w"
)
tk.Button(latency_frame, text="?", command=lambda: show_hint("SMALL_SKIP"), width=3).grid(row=2, column=3)

# Notebook for sorter tabs
toolkit = ttk.Notebook(root)
toolkit.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

# Per-sorter storage
base_path_vars, ks_output_dir_vars = [], []
bin_file_vars, meta_file_vars, chanmap_file_vars = [], [], []
rerun_ks_vars, sdm_vars = [], []
sdm_ip_vars, sdm_port_vars = [], []
sdm_subset_vars, sdm_trigger_bin_ms_vars = [], []
sdm_processor_vars = []
sdm_param_vars = []
sdm_param_frames = []       # per sorter: the frame holding the selected processor's widgets
sdm_when_widgets = {}       # sorter idx -> [(Param, Entry)] for enabling fields by `when`
template_toggle_vars = []
probe_vars = []            # per sorter: imec probe (SpikeGLX substream) it sorts
max_templates_vars = []
channel_range_vars = []
cmr_mode_vars, cmr_block_vars = [], []   # per sorter: median-reference grouping used by Kilosort + LSS
template_frames, file_frames, sdm_frames = [], [], []

# Hints dictionary
HINTS = {
    "BASE": "This directory should contain 'imec_raw' folder for your recording.",
    "KS_OUTPUT": "Select where the Kilosort output is or will be stored.",
    "BIN": "The location of your recording (.bin file).",
    "META": "The location of your recording's metadata (.meta file).",
    "CHANMAP": "The location of your probe's channel map (.mat file).",
    "SDM": "Send decoder output to stimulus display machine?",
    "SDM_PROCESSOR": ("How spike activity is turned into packets for the stimulus "
                      "display machine. Settings below change with the processor; "
                      "each processor's own help is under its settings."),
    "MAX_TEMPLATES": ("Limit how many preclustered templates the sorter uses "
                      "(0 = use all). Fewer templates = less GPU work in "
                      "matchingPursuit = better real-time performance on low-spec "
                      "GPUs, at some loss of detection. Applied to oss_input every "
                      "launch."),
    "CMR": ("Common median reference grouping. Global = one median across all "
            "channels (Kilosort default). The ADC presets take the median over "
            "channels the probe samples at the same instant (NP1.0: 12 groups of 32, "
            "NP2.0: 16 groups of 24), which cancels high-frequency noise better. "
            "Auto picks the preset from imDatPrb_type in the .meta file. Contiguous "
            "blocks = groups of N neighbouring channels. Used when Kilosort4 is "
            "re-run, and the online sorter applies the same groups. Reusing an "
            "existing Kilosort output keeps the grouping it was trained with."),
    "CHANNEL_RANGE": ("Keep only templates with support on a probe channel range, "
                      "e.g. '100-150' (inclusive probe channel numbers; blank = all "
                      "channels). Use this to focus the sorter on one region of "
                      "cortical space. Composes with Max templates: the channel "
                      "range narrows the field first, then Max templates keeps the "
                      "most active survivors. Applied to oss_input every launch."),
    "SGLX_HOST": ("The IP address of the machine running SpikeGLX. If SpikeGLX is "
                  "running on this same machine, leave as 127.0.0.1. Shared by all "
                  "sorters (they use one SpikeGLX connection)."),
    "SGLX_PORT": ("The port SpikeGLX is streaming on. Shared by all sorters."),
    "DRIFT_ENABLED": ("Controls BOTH the live drift estimate and the correction "
                      "applied to the data. ON: each batch is scanned with "
                      "Kilosort's universal templates (on whitened, NOT yet "
                      "drift-corrected data), the depth and amplitude of those "
                      "detections are binned into an activity fingerprint over a "
                      "window, and that fingerprint is registered against the "
                      "training reference to give an absolute vertical shift (um). "
                      "The correction matrix is rebuilt from that shift every "
                      "window, replacing the static one, and a drift trace is shown "
                      "in the output GUI. OFF: no estimate and no trace; the sorter "
                      "falls back to the STATIC drift_matrix.npy that Kilosort "
                      "produced at training time, which never updates -- this is the "
                      "control condition for drift experiments, not a "
                      "correction-free one. Shared by all sorters. Requires the "
                      "universal-template exports in oss_input/ -- re-run Kilosort4 "
                      "once if they are missing."),
    "DRIFT_WINDOW": ("Length in seconds of the window used to estimate drift. "
                     "Longer = more spikes = more robust estimate but slower to "
                     "react. Targets slow drift; 10 s is a good starting point and "
                     "matches the diagnostic's default."),
    "DRIFT_MAX_SHIFT": ("Safety clamp (microns) on the estimated shift. This is a "
                        "rail, not a tuning knob: if it engages, the estimate was "
                        "not trustworthy in the first place. The output GUI reports "
                        "how often it fires."),
    "DRIFT_RETRAIN_THRESHOLD": ("Microns of estimated drift above which the output "
                                "GUI suggests retraining templates. This only shows a "
                                "banner -- it never retrains on its own. Press "
                                "'Retrain templates' in the drift plot to actually "
                                "stop sorting and re-run Kilosort4 on the data "
                                "recorded so far, then relaunch. Distinct from Max "
                                "shift, which clamps the estimator itself."),
    "SPIKE_STREAM": ("Stream every sorted spike over UDP to another process "
                     "(Python, MATLAB, Bonsai, ...) so closed-loop logic can run "
                     "outside LSS without writing C++. host:port of the receiver; "
                     "127.0.0.1:9100 = a script on this machine. Fire-and-forget: "
                     "a missing receiver is harmless. Receiver + packet format: "
                     "src/Python/lss_stream.py. Shared by all sorters (packets "
                     "carry a sorter id and imec probe number); "
                     "src/Python/multi_probe_receiver.py combines probes."),
    "PROBE": ("Which imec probe this sorter reads from SpikeGLX: N = imecN "
              "(the number in the probe's .imecN.ap.bin file name)."
              "Every sorter needs its own GPU. Each sorter must use "
              "templates trained on its own probe's recording. Spike stream packets "
              "carry this number, so one receiver can combine probes."),
    "MAX_LAG": ("What to do when a batch takes longer than real time and "
                "unprocessed data (> 50 ms) piles up in SpikeGLX's buffer. "
                "0 (default): drop the backlog and skip to latest batch"
                "(lowest latency, but data loss) "
                "N > 0: keep sorting from the oldest unprocessed sample and catch "
                "up over later batches; only skip once the sorter is more than N ms "
                "behind. "
                "Shared by all sorters."),
    "SMALL_SKIP": ("Only needed when Max lag > 0. When the lag limit is exceeded: "
                   "checked = skip just far enough to be back at the limit (drops "
                   "less data, stays ~Max lag behind); unchecked = skip all the way "
                   "to the newest data."),
}

def show_hint(key):
    messagebox.showinfo("Hint", HINTS[key])


def processor_values(idx, name):
    """{key: value} of processor `name`'s settings for sorter idx."""
    return {k: v.get().strip() for k, v in sdm_param_vars[idx].get(name, {}).items()}


def apply_when(idx):
    """Enable only the fields whose `when` condition holds (e.g. the median grid
    while Stat == median). Cosmetic: every value is still saved and passed."""
    proc = PROCESSORS.get(sdm_processor_vars[idx].get())
    if proc is None:
        return
    values = processor_values(idx, proc.name)
    try:
        for prm, entry in sdm_when_widgets.get(idx, []):
            key, want = prm.when
            entry.config(state=("normal" if values.get(key) == want else "disabled"))
    except tk.TclError:
        pass   # widget destroyed during a tab rebuild


def browse_param(var, kind):
    if kind == "folder":
        d = filedialog.askdirectory(initialdir=var.get() or str(Path.home()))
        if d: var.set(d + "\\")
    else:
        f = filedialog.askopenfilename(initialdir=str(Path(var.get()).parent) if var.get() else str(Path.home()))
        if f: var.set(f)


def run_processor_button(idx):
    """The selected processor's extra button (e.g. closedloop's baseline build)."""
    proc = PROCESSORS[sdm_processor_vars[idx].get()]
    label, hook = proc.button
    oss_in = Path(base_path_vars[idx].get().strip()) / "oss_input"
    res = hook(Context(oss_in, sdm_trigger_bin_ms_vars[idx].get().strip() or "100",
                       processor_values(idx, proc.name)))
    msg = res.message
    if res.ok and res.subset is not None:
        sdm_subset_vars[idx].set(",".join(str(x) for x in res.subset))
        msg += f"\n\nSDM subset auto-set to {len(res.subset)} neurons."
    (messagebox.showinfo if res.ok else messagebox.showerror)(label + ("" if res.ok else " failed"), msg)


def render_processor_settings(idx):
    """(Re)build sorter idx's settings panel for the processor chosen in its dropdown."""
    frame = sdm_param_frames[idx]
    if frame is None:
        return
    for w in frame.winfo_children():
        w.destroy()
    sdm_when_widgets[idx] = []
    proc = PROCESSORS.get(sdm_processor_vars[idx].get())
    if proc is None:
        return
    pvars = sdm_param_vars[idx][proc.name]
    grid_row, row_slots = 0, {}      # row label -> [grid row, next free column]
    for prm in proc.params:
        var = pvars[prm.key]
        if prm.row:                  # side-by-side group, e.g. "FS (median)   low - [ ]  high + [ ]"
            if prm.row not in row_slots:
                tk.Label(frame, text=prm.row + ":").grid(row=grid_row, column=0, sticky="e", padx=5)
                row_slots[prm.row] = [grid_row, 1]
                grid_row += 1
            r, c = row_slots[prm.row]
            tk.Label(frame, text=prm.label).grid(row=r, column=c, sticky="e")
            widget = tk.Entry(frame, textvariable=var, width=prm.width)
            widget.grid(row=r, column=c + 1, sticky="w", padx=(0, 8))
            row_slots[prm.row][1] += 2
        else:
            tk.Label(frame, text=prm.label + ":").grid(row=grid_row, column=0, sticky="w", padx=5, pady=2)
            if prm.choices:
                widget = tk.Frame(frame)
                for choice in prm.choices:
                    tk.Radiobutton(widget, text=choice, variable=var, value=choice,
                                   command=lambda i=idx: apply_when(i)).pack(side="left")
            else:
                widget = tk.Entry(frame, textvariable=var, width=prm.width)
            widget.grid(row=grid_row, column=1, columnspan=4, sticky="w")
            if prm.browse:
                tk.Button(frame, text="Browse", command=lambda v=var, k=prm.browse: browse_param(v, k)).grid(
                    row=grid_row, column=5, padx=5)
            grid_row += 1
        if prm.when and isinstance(widget, tk.Entry):
            sdm_when_widgets[idx].append((prm, widget))
    if proc.button:
        tk.Button(frame, text=proc.button[0], command=lambda i=idx: run_processor_button(i)).grid(
            row=grid_row, column=0, columnspan=3, padx=5, pady=5, sticky="w")
    apply_when(idx)


def show_processor_help(idx):
    proc = PROCESSORS.get(sdm_processor_vars[idx].get())
    messagebox.showinfo("SDM processor", (proc.help if proc else "Unknown processor.") +
                        "\n\nSettings are declared in src/Python/sdm_processors.py.")

def browse_directory(idx):
    d = filedialog.askdirectory(initialdir=base_path_vars[idx].get() or str(Path.home()), title="Select BASE_PATH")
    if d: base_path_vars[idx].set(d)

def browse_ks_output_dir(idx):
    d = filedialog.askdirectory(initialdir=ks_output_dir_vars[idx].get() or str(Path.home()),
                               title="Select KS Output Directory")
    if d:
        ks_output_dir_vars[idx].set(d)
        # autopopulate BASE_PATH to parent directory of selected Kilosort directory
        base_path_vars[idx].set(str(Path(d).parent))

def browse_bin_file(idx):
    init = os.path.join(base_path_vars[idx].get().strip(), "imec_raw")
    f = filedialog.askopenfilename(initialdir=init, title="Select Recording Binary File", filetypes=[("Binary files","*.bin"), ("All files","*.*")])
    if f: bin_file_vars[idx].set(f)

def browse_meta_file(idx):
    init = os.path.join(base_path_vars[idx].get().strip(), "imec_raw")
    f = filedialog.askopenfilename(initialdir=init, title="Select Metadata File", filetypes=[("Metadata files","*.meta"),("All files","*.*")])
    if f: meta_file_vars[idx].set(f)

def browse_chanmap_file(idx):
    init = os.path.join(base_path_vars[idx].get().strip(), "imec_raw")
    f = filedialog.askopenfilename(initialdir=init, title="Select Channel Map File", filetypes=[("MAT files","*.mat"),("All files","*.*")])
    if f: chanmap_file_vars[idx].set(f)

def toggle_rerun(idx):
    if rerun_ks_vars[idx].get(): file_frames[idx].grid()
    else: file_frames[idx].grid_remove()

def refresh_tab_labels(*_):
    for i, tab in enumerate(toolkit.tabs()):
        if i < len(probe_vars):
            toolkit.tab(tab, text=f"Sorter {i+1} \u00b7 imec{probe_vars[i].get().strip()} \u00b7 GPU {i}")

def detected_gpu_count():
    """Number of NVIDIA GPUs per nvidia-smi, or None if it can't be run."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))

def toggle_template(idx):
    if template_toggle_vars[idx].get(): template_frames[idx].grid()
    else: template_frames[idx].grid_remove()

def toggle_sdm(idx):
    if sdm_vars[idx].get(): sdm_frames[idx].grid()
    else: sdm_frames[idx].grid_remove()

def update_tabs():
    n = num_sorters_var.get()
    old = len(base_path_vars)

    # Truncate lists if reducing
    if n < old:
        for lst in (base_path_vars, ks_output_dir_vars, bin_file_vars,
                    meta_file_vars, chanmap_file_vars, rerun_ks_vars,
                    sdm_vars, sdm_ip_vars, sdm_port_vars,
                    sdm_subset_vars, sdm_trigger_bin_ms_vars,
                    sdm_processor_vars, sdm_param_vars, sdm_param_frames,
                    template_toggle_vars, max_templates_vars, channel_range_vars,
                    cmr_mode_vars, cmr_block_vars, template_frames, file_frames, sdm_frames, probe_vars):
            del lst[n:]

    # Append new entries if increasing
    for i in range(old, n):
        rerun_ks_vars.append(tk.BooleanVar(value=True))
        base_path_vars.append(tk.StringVar(value=str(Path.home())))
        ks_output_dir_vars.append(tk.StringVar(value=""))
        bin_file_vars.append(tk.StringVar(value=""))
        meta_file_vars.append(tk.StringVar(value=""))
        chanmap_file_vars.append(tk.StringVar(value=""))
        sdm_vars.append(tk.BooleanVar(value=False))
        sdm_ip_vars.append(tk.StringVar(value=""))
        sdm_port_vars.append(tk.StringVar(value=""))
        sdm_subset_vars.append(tk.StringVar(value=""))
        sdm_trigger_bin_ms_vars.append(tk.StringVar(value="100"))
        sdm_processor_vars.append(tk.StringVar(value="zscore"))
        sdm_param_vars.append({proc: {k: tk.StringVar(value=v) for k, v in vals.items()}
                               for proc, vals in default_values().items()})
        sdm_param_frames.append(None)
        template_toggle_vars.append(tk.BooleanVar(value=False))
        probe_vars.append(tk.StringVar(value=str(i)))  
        probe_vars[-1].trace_add("write", refresh_tab_labels)
        max_templates_vars.append(tk.StringVar(value="0"))
        channel_range_vars.append(tk.StringVar(value=""))
        cmr_mode_vars.append(tk.StringVar(value=CMR_MODES[0]))
        cmr_block_vars.append(tk.StringVar(value="32"))
        template_frames.append(None)
        file_frames.append(None)
        sdm_frames.append(None)

    # Rebuild all tabs
    for tab in toolkit.tabs():
        toolkit.forget(tab)
    for i in range(n):
        frame = ttk.Frame(toolkit)
        toolkit.add(frame, text=f"Sorter {i+1}")
        build_tab(frame, i)
    refresh_tab_labels()

# Construct UI for a single sorter tab
def build_tab(frame, idx):
    row = 0
    probe_group = tk.Frame(frame)
    probe_group.grid(row=row, column=0, columnspan=4, padx=5, pady=5, sticky="w")
    tk.Label(probe_group, text="Probe (imecN):").grid(row=0, column=0, padx=5, sticky="w")
    probe = probe_vars[idx].get() 
    tk.Spinbox(probe_group, from_=0, to=31, textvariable=probe_vars[idx], width=4).grid(
        row=0, column=1, sticky="w"
    )
    probe_vars[idx].set(probe)
    tk.Button(probe_group, text="?", command=lambda: show_hint("PROBE"), width=3).grid(row=0, column=2, padx=5)
    row += 1

    # Rerun group: checkbox plus the file paths it reveals
    rerun_group = tk.Frame(frame)
    rerun_group.grid(row=row, column=0, columnspan=4, padx=5, pady=5, sticky="w")
    tk.Checkbutton(
        rerun_group, text="Rerun Kilosort4", variable=rerun_ks_vars[idx],
        command=lambda i=idx: toggle_rerun(i)
    ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
    row += 1

    # Kilosort output directory
    tk.Label(frame, text="Kilosort Output Directory:").grid(
        row=row, column=0, padx=5, pady=5, sticky="w"
    )
    tk.Entry(frame, textvariable=ks_output_dir_vars[idx], width=50).grid(
        row=row, column=1
    )
    tk.Button(
        frame, text="Browse", command=lambda i=idx: browse_ks_output_dir(i)
    ).grid(row=row, column=2)
    tk.Button(
        frame, text="?", command=lambda i=idx: show_hint("KS_OUTPUT"), width=3
    ).grid(row=row, column=3)
    row += 1

    # BASE_PATH
    tk.Label(frame, text="BASE_PATH:").grid(
        row=row, column=0, padx=5, pady=5, sticky="w"
    )
    tk.Entry(frame, textvariable=base_path_vars[idx], width=50).grid(
        row=row, column=1
    )
    tk.Button(
        frame, text="Browse", command=lambda i=idx: browse_directory(i)
    ).grid(row=row, column=2)
    tk.Button(
        frame, text="?", command=lambda i=idx: show_hint("BASE"), width=3
    ).grid(row=row, column=3)
    row += 1

    # Template filtering group
    template_group = tk.Frame(frame)
    template_group.grid(row=row, column=0, columnspan=4, padx=5, pady=5, sticky="w")
    tk.Checkbutton(
        template_group, text="Template filtering", variable=template_toggle_vars[idx],
        command=lambda i=idx: toggle_template(i)
    ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
    row += 1

    template_frame = tk.Frame(template_group, borderwidth=1, relief="sunken")
    template_frames[idx] = template_frame

    # Max templates (0 = use all)
    tk.Label(template_frame, text="Max templates (0 = all):").grid(
        row=0, column=0, padx=5, pady=5, sticky="w"
    )
    tk.Entry(template_frame, textvariable=max_templates_vars[idx], width=10).grid(
        row=0, column=1, sticky="w"
    )
    tk.Button(
        template_frame, text="?", command=lambda i=idx: show_hint("MAX_TEMPLATES"), width=3
    ).grid(row=0, column=3)

    # Channel range (blank = all channels)
    tk.Label(template_frame, text="Channel range (e.g. 100-150):").grid(
        row=1, column=0, padx=5, pady=5, sticky="w"
    )
    tk.Entry(template_frame, textvariable=channel_range_vars[idx], width=10).grid(
        row=1, column=1, sticky="w"
    )
    tk.Button(
        template_frame, text="?", command=lambda i=idx: show_hint("CHANNEL_RANGE"), width=3
    ).grid(row=1, column=3)

    template_frame.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    toggle_template(idx)

    # SDM toggle
    sdm_group = tk.Frame(frame)
    sdm_group.grid(row=row, column=0, columnspan=4, padx=5, pady=5, sticky="w")
    tk.Checkbutton(
        sdm_group, text="SDM?", variable=sdm_vars[idx],
        command=lambda i=idx: toggle_sdm(i)
    ).grid(row=0, column=0, sticky="w", padx=5, pady=5)
    tk.Button(
        sdm_group, text="?", command=lambda i=idx: show_hint("SDM"), width=3
    ).grid(row=0, column=1)
    row += 1

    # SDM subframe
    sdm_frame = tk.Frame(sdm_group, borderwidth=1, relief="sunken")
    sdm_frames[idx] = sdm_frame
    tk.Label(sdm_frame, text="SDM IP Address:").grid(
        row=0, column=0, padx=5, pady=5
    )
    tk.Entry(sdm_frame, textvariable=sdm_ip_vars[idx], width=25).grid(
        row=0, column=1
    )
    tk.Label(sdm_frame, text="Port Number:").grid(
        row=1, column=0
    )
    tk.Entry(sdm_frame, textvariable=sdm_port_vars[idx], width=10).grid(
        row=1, column=1
    )

    tk.Label(sdm_frame, text="Subset (comma-separated templates):").grid(
        row=2, column=0, padx=5, pady=5
    )
    tk.Entry(sdm_frame, textvariable=sdm_subset_vars[idx], width=25).grid(
        row=2, column=1
    )

    tk.Label(sdm_frame, text="Trigger bin (ms):").grid(
        row=3, column=0, padx=5, pady=5
    )
    tk.Entry(sdm_frame, textvariable=sdm_trigger_bin_ms_vars[idx], width=10).grid(
        row=3, column=1
    )

    tk.Label(sdm_frame, text="Processor:").grid(
        row=4, column=0, padx=5, pady=5
    )
    tk.OptionMenu(sdm_frame, sdm_processor_vars[idx], *PROCESSORS,
                  command=lambda _choice, i=idx: render_processor_settings(i)).grid(
        row=4, column=1, sticky="w"
    )
    tk.Button(
        sdm_frame, text="?", command=lambda i=idx: show_processor_help(i), width=3
    ).grid(row=4, column=2)

    param_frame = tk.Frame(sdm_frame)
    param_frame.grid(row=5, column=0, columnspan=4, padx=5, pady=(0, 5), sticky="w")
    sdm_param_frames[idx] = param_frame
    render_processor_settings(idx)

    sdm_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    toggle_sdm(idx)

    # File selection subframe
    f_frame = tk.Frame(rerun_group, borderwidth=1, relief="sunken")
    file_frames[idx] = f_frame
    tk.Label(f_frame, text="Recording binary file:").grid(
        row=0, column=0, padx=5, pady=5, sticky="w"
    )
    tk.Entry(f_frame, textvariable=bin_file_vars[idx], width=50).grid(
        row=0, column=1
    )
    tk.Button(
        f_frame, text="Browse", command=lambda i=idx: browse_bin_file(i)
    ).grid(row=0, column=2)
    tk.Button(
        f_frame, text="?", command=lambda i=idx: show_hint("BIN"), width=3
    ).grid(row=0, column=3)

    tk.Label(f_frame, text="Recording metadata file:").grid(
        row=1, column=0, padx=5, pady=5
    )
    tk.Entry(f_frame, textvariable=meta_file_vars[idx], width=50).grid(
        row=1, column=1
    )
    tk.Button(
        f_frame, text="Browse", command=lambda i=idx: browse_meta_file(i)
    ).grid(row=1, column=2)
    tk.Button(
        f_frame, text="?", command=lambda i=idx: show_hint("META"), width=3
    ).grid(row=1, column=3)

    tk.Label(f_frame, text="Channel map file:").grid(
        row=2, column=0, padx=5, pady=5
    )
    tk.Entry(f_frame, textvariable=chanmap_file_vars[idx], width=50).grid(
        row=2, column=1
    )
    tk.Button(
        f_frame, text="Browse", command=lambda i=idx: browse_chanmap_file(i)
    ).grid(row=2, column=2)
    tk.Button(
        f_frame, text="?", command=lambda i=idx: show_hint("CHANMAP"), width=3
    ).grid(row=2, column=3)

    # Median reference grouping (applied in Kilosort4 and in LSS)
    tk.Label(f_frame, text="Median reference:").grid(
        row=3, column=0, padx=5, pady=5, sticky="w"
    )
    cmr_row = tk.Frame(f_frame)
    cmr_row.grid(row=3, column=1, sticky="w")
    block_entry = tk.Entry(cmr_row, textvariable=cmr_block_vars[idx], width=6)
    def cmr_changed(*_):
        block_entry.configure(state="normal" if cmr_mode_vars[idx].get() == CMR_MODES[4] else "disabled")
    tk.OptionMenu(cmr_row, cmr_mode_vars[idx], *CMR_MODES, command=cmr_changed).grid(
        row=0, column=0, sticky="w"
    )
    tk.Label(cmr_row, text="Block size:").grid(row=0, column=1, padx=(10, 2))
    block_entry.grid(row=0, column=2)
    cmr_changed()
    tk.Button(
        f_frame, text="?", command=lambda: show_hint("CMR"), width=3
    ).grid(row=3, column=3)
    f_frame.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    toggle_rerun(idx)

# Finish and error widgets
def create_finish_widgets():
    global finish_button, error_text
    finish_button = tk.Button(root, text="Finish", command=finish_and_quit)
    finish_button.grid(row=3, column=0, columnspan=4, padx=5, pady=5)
    error_text = tk.Text(root, height=4, width=60, fg="red")
    error_text.grid(row=4, column=0, columnspan=4, padx=5, pady=5)
    error_text.configure(state="disabled")

# Destroy GUI on finish
def finish_and_quit():
    lag = max_lag_var.get().strip() or "0"
    if not lag.isdigit():
        messagebox.showerror("Invalid setting", f"Max lag must be a whole number of ms >= 0 (got '{lag}').")
        return
    if not check_probes():
        return
    root.destroy()

def check_probes():
    """Per-sorter probe sanity checks before launch. False = stay in the GUI."""
    n = num_sorters_var.get()
    probes = [v.get().strip() for v in probe_vars[:n]]
    bad = [f"Sorter {i+1}: '{p}'" for i, p in enumerate(probes) if not p.isdigit()]
    if bad:
        messagebox.showerror("Invalid probe", "Probe must be a whole number >= 0.\n" + "\n".join(bad))
        return False

    gpus = detected_gpu_count()
    if gpus is not None and n > gpus:
        messagebox.showerror("Not enough GPUs",
                             f"{n} sorters need {n} GPUs (one each), but nvidia-smi lists {gpus}.")
        return False

    warnings = []
    for p in sorted(set(probes)):
        users = [i + 1 for i, q in enumerate(probes) if q == p]
        if len(users) > 1:
            warnings.append(f"Sorters {', '.join(map(str, users))} all read imec{p}.")
    for i in range(n):
        name = Path(bin_file_vars[i].get().strip()).name
        if rerun_ks_vars[i].get() and ".imec" in name and f".imec{probes[i]}." not in name:
            warnings.append(f"Sorter {i+1} reads imec{probes[i]} but trains on {name}.")
    if warnings:
        return messagebox.askyesno("Check probe selection", "\n".join(warnings) + "\n\nLaunch anyway?")
    return True

def legacy_sdm_params(state, i):
    """Processor settings from a multi_gui_state.json written before sdm_processors.py
    existed (flat per-field keys), so saved FS/RS thresholds survive the upgrade."""
    def get(key, default):
        vals = state.get(key)
        return vals[i] if isinstance(vals, list) and i < len(vals) else default
    off, z = get("sdm_offsets", "0"), get("sdm_trigger_zs", "1.0")
    return {
        "zscore": {"trigger_z": z, "baseline_min_seconds": get("sdm_baseline_min_seconds", "10.0")},
        "closedloop": {
            "mode": get("sdm_modes", "median"), "boundary_us": get("sdm_boundary_us", "200"),
            **{f"{pop}_offset_{side}": get(f"sdm_offset_{pop}_{side}", off)
               for pop in ("fs", "rs") for side in ("low", "high")},
            **{f"{pop}_z_{side}": get(f"sdm_z_{pop}_{side}", z)
               for pop in ("fs", "rs") for side in ("low", "high")},
        },
    }

# Initialize GUI (with state loading and applying)
def main():
    # Load previous state
    state = None
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            num_sorters_var.set(state.get("num_sorters", num_sorters_var.get()))
            sglx_host_var.set(state.get("sglx_host", sglx_host_var.get()))
            sglx_port_var.set(state.get("sglx_port", sglx_port_var.get()))
            drift_enabled_var.set(state.get("drift_enabled", drift_enabled_var.get()))
            drift_window_var.set(state.get("drift_window_s", drift_window_var.get()))
            drift_max_shift_var.set(state.get("drift_max_shift_um", drift_max_shift_var.get()))
            drift_retrain_threshold_var.set(state.get("drift_retrain_threshold_um", drift_retrain_threshold_var.get()))
            spike_stream_enabled_var.set(state.get("spike_stream_enabled", spike_stream_enabled_var.get()))
            spike_stream_addr_var.set(state.get("spike_stream_addr", spike_stream_addr_var.get()))
            max_lag_var.set(state.get("max_lag_ms", max_lag_var.get()))
            small_skip_var.set(state.get("small_skip", small_skip_var.get()))
        except Exception as e:
            print(f"Could not load GUI state: {e}")

    # Build tabs
    update_tabs()

    # Apply loaded state to each tab
    if state:
        for i in range(num_sorters_var.get()):
            base_path_vars[i].set(state["base_paths"][i])
            ks_output_dir_vars[i].set(state["ks_output_dirs"][i])
            bin_file_vars[i].set(state["bin_files"][i])
            meta_file_vars[i].set(state["meta_files"][i])
            chanmap_file_vars[i].set(state["chanmap_files"][i])
            rerun_ks_vars[i].set(state["rerun_flags"][i])
            toggle_rerun(i)
            sdm_vars[i].set(state["sdm_flags"][i])
            toggle_sdm(i)
            sdm_ip_vars[i].set(state["sdm_ips"][i])
            sdm_port_vars[i].set(state["sdm_ports"][i])
            sdm_subset_vars[i].set(state.get("sdm_subsets", [""] * num_sorters_var.get())[i])
            sdm_trigger_bin_ms_vars[i].set(state.get("sdm_trigger_bin_ms", ["50"] * num_sorters_var.get())[i])
            sdm_processor_vars[i].set(state.get("sdm_processors", ["zscore"] * num_sorters_var.get())[i])
            saved = state.get("sdm_params") or []
            saved = saved[i] if i < len(saved) else legacy_sdm_params(state, i)
            for proc, vals in saved.items():
                for key, val in vals.items():
                    if key in sdm_param_vars[i].get(proc, {}):
                        sdm_param_vars[i][proc][key].set(val)
            render_processor_settings(i)
            max_templates_vars[i].set(state.get("max_templates", ["0"] * num_sorters_var.get())[i])
            channel_range_vars[i].set(state.get("channel_ranges", [""] * num_sorters_var.get())[i])
            modes = state.get("cmr_modes", [])
            cmr_mode_vars[i].set(modes[i] if i < len(modes) and modes[i] in CMR_MODES else CMR_MODES[0])
            blocks = state.get("cmr_block_sizes", [])
            cmr_block_vars[i].set(blocks[i] if i < len(blocks) else "32")
            flags = state.get("template_filter_flags", [])
            template_toggle_vars[i].set(flags[i] if i < len(flags) else False)
            toggle_template(i)
            probes = state.get("substreams", [])
            probe_vars[i].set(probes[i] if i < len(probes) else str(i))

    create_finish_widgets()
    root.mainloop()

# Entry point for full run
def run_online_multi():
    main()

    # Gather inputs
    n = num_sorters_var.get()
    BASE_PATHS = [Path(v.get().strip()) for v in base_path_vars]
    KS_OUTPUT_DIRS = [Path(v.get().strip()) for v in ks_output_dir_vars]
    BIN_FILES = [Path(v.get().strip()) for v in bin_file_vars]
    META_FILES = [Path(v.get().strip()) for v in meta_file_vars]
    CHANMAP_FILES = [Path(v.get().strip()) for v in chanmap_file_vars]
    RERUN_FLAGS = [v.get() for v in rerun_ks_vars]
    PROBES = [v.get().strip() for v in probe_vars]
    SDM_FLAGS = [v.get() for v in sdm_vars]
    SDM_IPS = [v.get().strip() for v in sdm_ip_vars]
    SDM_PORTS = [v.get().strip() for v in sdm_port_vars]
    SDM_SUBSETS = [v.get().strip() for v in sdm_subset_vars]
    SDM_TRIGGER_BIN_MS = [v.get().strip() for v in sdm_trigger_bin_ms_vars]
    SDM_PROCESSORS = [v.get().strip() for v in sdm_processor_vars]
    SDM_PARAMS = [{proc: {k: v.get().strip() for k, v in keys.items()} for proc, keys in per.items()}
                  for per in sdm_param_vars]
    TEMPLATE_FILTER_FLAGS = [v.get() for v in template_toggle_vars]
    RAW_MAX_TEMPLATES = [v.get().strip() for v in max_templates_vars]
    RAW_CHANNEL_RANGES = [v.get().strip() for v in channel_range_vars]
    MAX_TEMPLATES = [m if on else "0" for m, on in zip(RAW_MAX_TEMPLATES, TEMPLATE_FILTER_FLAGS)]
    CHANNEL_RANGES = [c if on else "" for c, on in zip(RAW_CHANNEL_RANGES, TEMPLATE_FILTER_FLAGS)]
    CMR_CONFIGS = [(m.get(), b.get().strip()) for m, b in zip(cmr_mode_vars, cmr_block_vars)]
    SGLX_HOST = sglx_host_var.get().strip()
    SGLX_PORT = sglx_port_var.get().strip()
    DRIFT_ENABLED = drift_enabled_var.get()
    DRIFT_WINDOW_S = drift_window_var.get().strip()
    DRIFT_MAX_SHIFT_UM = drift_max_shift_var.get().strip()
    DRIFT_RETRAIN_THRESHOLD_UM = drift_retrain_threshold_var.get().strip()
    SPIKE_STREAM_ENABLED = spike_stream_enabled_var.get()
    SPIKE_STREAM_ADDR = spike_stream_addr_var.get().strip()
    MAX_LAG_MS = max_lag_var.get().strip() or "0"
    SMALL_SKIP = small_skip_var.get()

    # Save current state
    state = {
        "num_sorters": n,
        "sglx_host": SGLX_HOST,
        "sglx_port": SGLX_PORT,
        "drift_enabled": DRIFT_ENABLED,
        "drift_window_s": DRIFT_WINDOW_S,
        "drift_max_shift_um": DRIFT_MAX_SHIFT_UM,
        "drift_retrain_threshold_um": DRIFT_RETRAIN_THRESHOLD_UM,
        "spike_stream_enabled": SPIKE_STREAM_ENABLED,
        "spike_stream_addr": SPIKE_STREAM_ADDR,
        "max_lag_ms": MAX_LAG_MS,
        "small_skip": SMALL_SKIP,
        "base_paths": [str(p) for p in BASE_PATHS],
        "ks_output_dirs": [str(d) for d in KS_OUTPUT_DIRS],
        "bin_files": [str(p) for p in BIN_FILES],
        "meta_files": [str(p) for p in META_FILES],
        "chanmap_files": [str(p) for p in CHANMAP_FILES],
        "substreams": PROBES,
        "rerun_flags": RERUN_FLAGS,
        "sdm_flags": SDM_FLAGS,
        "sdm_ips": SDM_IPS,
        "sdm_ports": SDM_PORTS,
        "sdm_subsets": SDM_SUBSETS,
        "sdm_trigger_bin_ms": SDM_TRIGGER_BIN_MS,
        "sdm_processors": SDM_PROCESSORS,
        "sdm_params": SDM_PARAMS,
        "template_filter_flags": TEMPLATE_FILTER_FLAGS,
        "max_templates": RAW_MAX_TEMPLATES,
        "channel_ranges": RAW_CHANNEL_RANGES,
        "cmr_modes": [m for m, _ in CMR_CONFIGS],
        "cmr_block_sizes": [b for _, b in CMR_CONFIGS]
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Could not save GUI state: {e}")

    # Curate OSS input dirs
    OSS_DIRS = curate_oss_input_dir(BASE_PATHS, KS_OUTPUT_DIRS, BIN_FILES,
                                     META_FILES, CHANMAP_FILES, RERUN_FLAGS,
                                     MAX_TEMPLATES, CHANNEL_RANGES, CMR_CONFIGS)

    PREPARED = {}
    for i, proc_name in enumerate(SDM_PROCESSORS):
        proc = PROCESSORS.get(proc_name)
        if SDM_FLAGS[i] and proc is not None and proc.prepare is not None:
            res = proc.prepare(Context(Path(OSS_DIRS[i]), SDM_TRIGGER_BIN_MS[i] or "100",
                                       SDM_PARAMS[i].get(proc_name, {})))
            print(f"[{proc_name}] sorter {i+1} prepare{'' if res.ok else ' FAILED'}:\n{res.message}")
            PREPARED[i] = res

    # Build and run C++ command
    decoder_input_dirs = [bp / 'decoder_input' for bp in BASE_PATHS]
    cuda_output_dirs = [bp / 'cuda_output' for bp in BASE_PATHS]
    for d in decoder_input_dirs + cuda_output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    arguments = {
        '--n_gpus': str(n),
        '--substream': PROBES,
        '--oss_input': OSS_DIRS,
        '--decoder_input': [str(d) + '\\' for d in decoder_input_dirs],
        '--spikes_output': [str(d / 'spikeOutput.txt') for d in decoder_input_dirs],
        '--cuda_output_dir': [str(d) + '\\' for d in cuda_output_dirs],
        '--sglx_host': SGLX_HOST,
        '--sglx_port': SGLX_PORT
    }

    if any(SDM_FLAGS):
        sdm_idxs = [i for i, enabled in enumerate(SDM_FLAGS) if enabled]
        sdm_idx = sdm_idxs[0]

        sdm_port = SDM_PORTS[sdm_idx]
        if not sdm_port.isdigit():
            print(f"Error: SDM enabled but port is not a valid integer: '{sdm_port}'")
            sys.exit(1)

        arguments['--sdm_ip'] = SDM_IPS[sdm_idx]
        arguments['--sdm_port'] = sdm_port
        proc_name = SDM_PROCESSORS[sdm_idx] or 'zscore'
        arguments['--sdm_processor'] = proc_name
        if SDM_TRIGGER_BIN_MS[sdm_idx]:
            arguments['--sdm_trigger_bin_ms'] = SDM_TRIGGER_BIN_MS[sdm_idx]

        prepared = PREPARED.get(sdm_idx)
        subset = SDM_SUBSETS[sdm_idx]
        if prepared is not None and prepared.subset is not None:
            subset = ",".join(str(x) for x in prepared.subset)
        if subset:
            arguments['--sdm_subset'] = subset

        # The processor's own settings, as --sdm_param key=value ... (see sdm_processors.py)
        kv = to_exe_args(SDM_PARAMS[sdm_idx].get(proc_name, {}), prepared.params if prepared else None)
        if kv:
            arguments['--sdm_param'] = kv

    if DRIFT_ENABLED:
        if DRIFT_WINDOW_S:
            arguments['--drift_window_s'] = DRIFT_WINDOW_S
        if DRIFT_MAX_SHIFT_UM:
            arguments['--drift_max_shift_um'] = DRIFT_MAX_SHIFT_UM
        if DRIFT_RETRAIN_THRESHOLD_UM:
            arguments['--drift_retrain_threshold_um'] = DRIFT_RETRAIN_THRESHOLD_UM

    if SPIKE_STREAM_ENABLED and SPIKE_STREAM_ADDR:
        arguments['--spike_stream'] = SPIKE_STREAM_ADDR

    arguments['--max_lag_ms'] = MAX_LAG_MS

    script_dir = pathlib.Path(__file__).parent.resolve()
    # Go back exactly two levels to reach the project root and build path to executable 
    exe_path = script_dir.parent.parent / "x64" / "RELEASE" / "OnlineSpikes.exe"
    # Ex: cmd = [r"C:\Users\moorelab\source\repos\LiveSpikeSorter_Remote\x64\RELEASE\OnlineSpikes.exe"]
    cmd = [str(exe_path)]
    for k, v in arguments.items():
        if isinstance(v, list):
            cmd.append(k)
            cmd.extend(v)
        else:
            cmd.extend([k, str(v)])
    cmd.append('--no_input_gui')
    if DRIFT_ENABLED:
        cmd.append('--drift_estimation')
    if SMALL_SKIP:
        cmd.append('--small_skip')

    # RETRAIN loop
    while True:
        print("Running LSS with:", shlex.join(cmd))
        proc = subprocess.Popen(cmd, shell=True)
        rc = proc.wait()

        retrain_sources = {}   # sorter index -> (bin_path, meta_path)
        for i, oss in enumerate(OSS_DIRS):
            sentinel = Path(oss.rstrip('\\/')) / 'retrain_request.json'
            if not sentinel.exists():
                continue
            try:
                info = json.loads(sentinel.read_text())
                bin_path, meta_path = resolve_retrain_recording(info)
            except Exception as e:
                print(f"[Retrain] Could not resolve saved recording for sorter "
                      f"{i+1}: {e}. Skipping retrain for this sorter.")
                sentinel.unlink(missing_ok=True)
                continue
            retrain_sources[i] = (bin_path, meta_path)
            sentinel.unlink(missing_ok=True)

        if not retrain_sources:
            break

        print(f"[Retrain] Re-running Kilosort4 for sorter(s) "
              f"{[i+1 for i in retrain_sources]} on the data recorded so far. "
              f"Sorting is stopped until this finishes.")
        rerun = [(i in retrain_sources) for i in range(len(BASE_PATHS))]
        OSS_DIRS = curate_oss_input_dir(
            BASE_PATHS, KS_OUTPUT_DIRS, BIN_FILES, META_FILES, CHANMAP_FILES,
            rerun, MAX_TEMPLATES, CHANNEL_RANGES, CMR_CONFIGS, retrain_sources=retrain_sources)
        print("[Retrain] Templates rebuilt. Relaunching sorter...")


def resolve_retrain_recording(info):
    """Given a parsed retrain_request.json, return (bin_path, meta_path) for the
    SpikeGLX file the sorter just finalized.

    The sorter records SpikeGLX's data dir, run name and probe substream, but
    the exact gate/trigger indices are awkward to reconstruct reliably. Instead
    we glob the data dir for this probe's AP files and pick the most recently
    modified one -- that is the file finalized when the button was pressed.
    """
    data_dir = info.get("data_dir", "")
    substream = int(info.get("substream", 0))
    if not data_dir or not Path(data_dir).is_dir():
        raise FileNotFoundError(f"SpikeGLX data_dir not found: {data_dir!r}")

    pattern = f"*.imec{substream}.ap.bin"
    candidates = list(Path(data_dir).rglob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No {pattern} under {data_dir} (is SpikeGLX Save enabled?)")

    bin_path = max(candidates, key=lambda p: p.stat().st_mtime)
    meta_path = bin_path.with_suffix(".meta")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta beside {bin_path.name}: {meta_path}")

    print(f"[Retrain] Using recording {bin_path} "
          f"(run_name={info.get('run_name')!r}, "
          f"file_sample_count={info.get('file_sample_count')}).")
    return str(bin_path), str(meta_path)


def cluster_centroids_pca_compute(templates, Wall, pc_feature_ind):
    """
    Compute 1D PCA‐space centroids for each template by weighting
    the per‐channel PC features in `Wall` by that channel’s
    peak-to-peak amplitude in `templates`.

    Parameters
    ----------
    templates : np.ndarray, shape (T, nt, C)
        Raw waveforms for each of T templates,
        nt time‐samples × C channels.

    Wall : np.ndarray, shape (T, C, P)
        PC‐feature “templates”: for each of the T templates,
        C channels × P PCA‐dimensions.

    pc_feature_ind : np.ndarray of ints, shape (T, K)
        For each template t, the indices of the K channels
        to use in the weighting.

    Returns
    -------
    centroids : np.ndarray, shape (T, P)
        For each template t and PCA‐dimension p,
        the weighted average of Wall[t, channel_j, p].
    """
    T, nt, C = templates.shape
    T2, C2, P = Wall.shape
    assert T == T2 and C == C2, "templates and Wall must agree on T and C"
    
    # 1) Compute peak-to-peak amplitude per template/channel: shape (T, C)
    p2p = templates.max(axis=1) - templates.min(axis=1)
    
    # 2) Allocate output
    centroids = np.zeros((T, P), dtype=Wall.dtype)
    
    # 3) Loop over templates
    for t in range(T):
        idx = pc_feature_ind[t]             # shape (K,)
        weights = p2p[t, idx].astype(float) # shape (K,)
        wsum = weights.sum()
        if wsum == 0:
            # fallback to uniform if all amplitudes are zero
            weights = np.ones_like(weights)
            wsum = float(weights.sum())
        
        # select the K×P block of PC‐features
        W_sub = Wall[t, idx, :]             # shape (K, P)
        
        # weighted average across the K channels:
        #   sum_j weights[j] * W_sub[j, :]  /  sum(weights)
        centroids[t, :] = (weights[:, None] * W_sub).sum(axis=0) / wsum
    
    return centroids

def save_kilosort_drift_plots(dshift, st0, settings):
    """
    Saves drift_amount.png into our results directory (using Kilosort4's own plotting code).
    Used for targetting training only atm

    Resulting PNGs should be identical to what Kilosort would have produced
    """
    if dshift is None or st0 is None:
        print("Drift correction disabled (nblocks=0); skipping drift plots.")
        return
    from qtpy import QtWidgets
    from kilosort.gui.sanity_plots import PlotWindow, plot_drift_amount, plot_drift_scatter
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    drift_amount_window = PlotWindow(width=400, height=400, title='Drift Amount')
    drift_scatter_window = PlotWindow(width=1500, height=700, title='Drift Scatter',
                                      background='w')
    plot_drift_amount(drift_amount_window, dshift, settings)
    plot_drift_scatter(drift_scatter_window, st0, settings)
    app.processEvents()


def run_kilosort_with_drift_plots(settings, probe_name=None, results_dir=None,
                                  filename=None, data_dtype=None, do_CAR=True,
                                  invert_sign=False, device=None, cmr=None):
    """
    Runs kilosort.run_kilosort() with the drift-correction outputs
    """
    import torch
    from kilosort.parameters import DEFAULT_SETTINGS
    from kilosort.run_kilosort import (
        set_files, setup_logger as ks_setup_logger, logger as ks_logger,
        initialize_ops, compute_preprocessing, compute_drift_correction,
        detect_spikes, cluster_spikes, save_sorting,
    )

    settings = {**DEFAULT_SETTINGS, **settings}
    filename, data_dir, results_dir, probe = set_files(
        settings, filename, None, probe_name, None, results_dir)
    ks_setup_logger(results_dir)

    if data_dtype is None:
        data_dtype = 'int16'
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if probe['chanMap'].max() >= settings['n_chan_bin']:
        raise ValueError(
            'Largest value of chanMap exceeds channel count of data, '
            'make sure chanMap is 0-indexed.')

    tic0 = time.time()
    ops = initialize_ops(settings, probe, data_dtype, do_CAR, invert_sign,
                         device, False)

    # Median reference groups 
    car_groups = np.zeros(len(probe['chanMap']), dtype=np.int32)
    if cmr is not None:
        car_groups, desc = compute_cmr_groups(cmr[0], cmr[1], cmr[2], probe['chanMap'])
        print(f"Median reference: {desc}")
        if np.unique(car_groups).size > 1:
            import inspect
            from kilosort.io import BinaryFiltered
            if 'car_groups' not in inspect.signature(BinaryFiltered.__init__).parameters:
                raise RuntimeError(
                    "Grouped median reference needs the patched Kilosort in External/Kilosort. "
                    "Install it with: pip install ./External/Kilosort")
            ops['car_groups'] = car_groups
    np.save(Path(results_dir) / 'car_groups.npy', car_groups)

    ops = compute_preprocessing(ops, device, tic0=tic0)

    np.random.seed(1)
    torch.cuda.manual_seed_all(1)
    torch.random.manual_seed(1)
    ops, bfile, st0 = compute_drift_correction(ops, device, tic0=tic0)

    # Snapshot the drift-stage detector before detect_spikes clobbers it.
    drift_snapshot = snapshot_drift_stage(ops, st0)

    ops['settings']['results_dir'] = str(results_dir)
    save_kilosort_drift_plots(ops.get('dshift'), st0, ops['settings'])

    st, tF, _, _ = detect_spikes(ops, results_dir, device, bfile, tic0=tic0)
    clu, Wall = cluster_spikes(results_dir, st, tF, ops, device, bfile, ks_logger,
                               tic0=tic0)
    save_sorting(ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0)

    save_match_baseline(ops, results_dir, st, clu)

    return drift_snapshot


def save_match_baseline(ops, results_dir, st, clu):
    """
    Per-final-cluster match strength measured on the TRAINING recording, so the
    live sorter can report each unit as a percentage of how well Kilosort itself
    matched that template when it built it.

    st comes from detect_spikes, i.e. from template_matching.extract():
      st[:,1] = preclustered template index (indexes Wall3)
      st[:,2] = matching-pursuit amplitude, amp = B/nm with nm = ||U||^2

    Kilosort gates detection on Cf = relu(B)^2/nm > Th_learned^2, so
      sqrt(Cf) = B/||U|| = amp * sqrt(nm)
    and dividing by Th_learned puts it on the same >= 1.0 scale the live sorter
    reports. Same formula, same Th, same templates -> directly comparable.
    """
    from pathlib import Path
    results_dir = Path(results_dir)
    try:
        Wall3 = np.load(results_dir / 'Wall3.npy')          # [unclu_T, K, C]
    except Exception as e:
        print(f"[MatchBaseline] Wall3.npy unavailable ({e}); skipping baseline export.")
        return

    Th = float(ops['Th_learned'])
    if Th <= 0:
        print("[MatchBaseline] Th_learned is not positive; skipping baseline export.")
        return

    st  = np.asarray(st)
    clu = np.asarray(clu).astype(np.int64)
    if st.shape[0] != clu.shape[0]:
        print(f"[MatchBaseline] st ({st.shape[0]}) and clu ({clu.shape[0]}) "
              f"lengths disagree; skipping baseline export.")
        return

    nm = (Wall3.astype(np.float64) ** 2).reshape(Wall3.shape[0], -1).sum(axis=1)

    tmpl = st[:, 1].astype(np.int64)
    amp  = st[:, 2].astype(np.float64)
    ok   = (tmpl >= 0) & (tmpl < nm.shape[0])
    scores = np.zeros(st.shape[0], dtype=np.float64)
    scores[ok] = amp[ok] * np.sqrt(nm[tmpl[ok]]) / Th

    nClusters = int(clu.max()) + 1 if clu.size else 0
    baseline = np.zeros(nClusters, dtype=np.float32)
    counts   = np.zeros(nClusters, dtype=np.int64)
    for c in range(nClusters):
        sel = (clu == c) & ok
        n = int(sel.sum())
        counts[c] = n
        if n:
            baseline[c] = float(np.median(scores[sel]))

    np.save(results_dir / 'match_baseline.npy', baseline)
    nz = counts > 0
    print(f"[MatchBaseline] {int(nz.sum())}/{nClusters} clusters, "
          f"median baseline {np.median(baseline[nz]) if nz.any() else float('nan'):.3f} "
          f"(x detection threshold) -> match_baseline.npy")


def _to_numpy(x, dtype=None):
    """torch tensor / list / array -> contiguous numpy array."""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    x = np.ascontiguousarray(np.asarray(x))
    return x.astype(dtype) if dtype is not None else x


def snapshot_drift_stage(ops, st0, device=None):
    """
    Copy everything the live estimator needs
    """
    import torch
    from kilosort.datashift import bin_spikes, align_block2

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    F, ysamp = bin_spikes(ops, st0)
    _, _, F0, _ = align_block2(F, ysamp, ops, device=device)

    # dshift is (Nbatches, 2*nblocks-1). We estimate rigidly, so collapse the
    # blocks to one trace; with the default nblocks=1 this is a no-op.
    dshift = np.asarray(ops['dshift']).mean(axis=1)
    yc = _to_numpy(ops['yc'], np.float32)

    return {
        # universal-template detector (drift stage)
        'ud_wTEMP': _to_numpy(ops['wTEMP'], np.float32),
        'ud_iC': _to_numpy(ops['iC'], np.int32),
        'ud_iC2': _to_numpy(ops['iC2'], np.int32),
        'ud_weigh': _to_numpy(ops['weigh'], np.float32),
        'ud_ycup': _to_numpy(ops['ycup'], np.float32),
        'ud_xcup': _to_numpy(ops['xcup'], np.float32),
        # reference fingerprint + Kilosort's own training drift trace
        'ks_F0': np.ascontiguousarray(_to_numpy(F0, np.float32)),
        'ks_dshift': np.ascontiguousarray(dshift.astype(np.float32)),
        # scalars for misc.txt
        'scalars': {
            'Th_universal': float(ops['settings']['Th_universal']),
            'n_templates': int(ops['settings']['n_templates']),
            'template_sizes': int(ops['settings']['template_sizes']),
            'nearest_templates': int(ops['settings']['nearest_templates']),
            'nt': int(ops['nt']),
            'n_filters': int(_to_numpy(ops['iC']).shape[1]),
            'batch_size': int(ops['batch_size']),
            'n_amp_bins': 20,
            # yc_min/yc_max are written by the main misc.txt block; do not
            # duplicate them here (the C++ parser is a map -- last line wins).
        },
    }


def curate_oss_input_dir(BASE_PATHS, KS_OUTPUT_DIRS, BIN_FILES,
                         META_FILES, CHANMAP_FILES, RERUN_FLAGS,
                         MAX_TEMPLATES=None, CHANNEL_RANGES=None,
                         CMR_CONFIGS=None, retrain_sources=None):
    retrain_sources = retrain_sources or {}
    OSS_DIRS = []
    for i, base in enumerate(BASE_PATHS):
        ks_out = KS_OUTPUT_DIRS[i]
        oss_in = base / 'oss_input'
        # Kilosort4 is checked, or on the very first curate for this sorter.
        if not RERUN_FLAGS[i] and (oss_in / 'templates.npy').exists():
            print(f"Reusing existing oss_input for sorter {i+1} "
                  "(Rerun Kilosort4 unchecked); skipping Kilosort4/torch-dependent steps.")
            OSS_DIRS.append(str(oss_in) + '\\')
            continue

        try:
            import torch
            from kilosort.preprocessing import get_drift_matrix
            from kilosort.template_matching import prepare_extract
        except ImportError as e:
            print("Error: curating oss_input/ requires the Kilosort4 training "
                  "dependencies (torch, kilosort, ...).\n"
                  "Install them with:\n"
                  "    pip install -r requirements-training.txt\n"
                  f"({e})")
            sys.exit(1)

        if RERUN_FLAGS[i]:
            if not torch.cuda.is_available():
                print(f"Error: No GPU for sorter {i+1}")
                sys.exit(1)
            if i in retrain_sources:
                bin_path, meta_path = retrain_sources[i]
                data_dir = str(Path(bin_path).parent)
                print(f"[Retrain] Starting kilosort for sorter {i+1} on {bin_path}...")
            else:
                bin_path = BIN_FILES[i]
                meta_path = META_FILES[i]
                data_dir = str(base / 'imec_raw')
                print(f"Starting kilosort for sorter {i+1}...")
            meta = parse_bin_meta_file(meta_path)
            start = time.time()
            settings = {'data_dir': data_dir, 'n_chan_bin': meta['nSavedChans']}
            cmr_mode, cmr_block = CMR_CONFIGS[i] if CMR_CONFIGS else (CMR_MODES[0], "32")
            drift_snapshot = run_kilosort_with_drift_plots(
                settings=settings,
                probe_name=CHANMAP_FILES[i],
                results_dir=str(ks_out),
                filename=bin_path,
                cmr=(cmr_mode, cmr_block, meta))
            print(f"Kilosort sorter {i+1} took {time.time() - start:.2f} s")
        else:
            drift_snapshot = None

        print(f"Loading kilosort output sorter {i+1} from {ks_out}")
        # load all files
        amplitudes = np.load(ks_out / 'amplitudes.npy')
        spike_times = np.load(ks_out / 'spike_times.npy')
        spike_templates = np.load(ks_out / 'spike_templates.npy')
        spike_detection_templates = np.load(ks_out / 'spike_detection_templates.npy')
        templates      = np.load(ks_out / 'templates.npy')
        whitening_mat  = np.load(ks_out / 'whitening_mat.npy')
        channel_map    = np.load(ks_out / 'channel_map.npy')
        spike_positions = np.load(ks_out / 'spike_positions.npy')
        ops             = np.load(ks_out / 'ops.npy', allow_pickle=True).item()
        Wall3           = np.ascontiguousarray(np.load(ks_out / 'Wall3.npy'))
        ctc             = np.ascontiguousarray(np.load(ks_out / 'ctc.npy'))
        ctc_p           = torch.tensor(ctc.copy()).permute(1,0,2).contiguous().numpy()
        wPCA            = np.ascontiguousarray(np.load(ks_out / 'wPCA.npy'))
        wPCA_p          = torch.from_numpy(np.copy(wPCA)).permute(1,0).contiguous().numpy()
        centroids      = np.load(ks_out / 'cluster_centroids.npy', allow_pickle=True).item()
        centroids      = [v for v in centroids.values()]
        hpf            = np.array(ops['preprocessing']['hp_filter'])
        dshift         = np.array(ops['dshift'])
        drift_slope, _ = np.polyfit(np.arange(0, 30), dshift[-30:], 1)
        pc_feature_ind = np.load(ks_out / 'pc_feature_ind.npy')
        Wall           = np.load(ks_out / 'Wall.npy')
        cluster_centroids_pca = cluster_centroids_pca_compute(templates, Wall, pc_feature_ind)
        print(cluster_centroids_pca.shape)

        if abs(2 * drift_slope) >= 0.01: # If more than 0.01 micron drift per second in the last minute, it is unstable
            print("WARNING: Probe not yet stable. Spike sorter quality may be impacted by drift.")

        print(f"Wall3.shape = {Wall3.shape}")
        drift_matrix   = np.array(get_drift_matrix(ops, dshift[-1], device='cpu')).T
        iCC, iU, Ucc  = prepare_extract(ops, torch.tensor(Wall3), ops['settings']['nearest_chans'], device='cpu')
        iCC = np.ascontiguousarray(iCC)
        iU  = np.ascontiguousarray(iU)
        Ucc = np.ascontiguousarray(Ucc)
        xc  = np.ascontiguousarray(np.array(ops['xc']))
        yc  = np.ascontiguousarray(np.array(ops['yc']))
        pre_wf = torch.einsum('ijk,jl->kil', torch.tensor(Wall3), torch.tensor(wPCA)).permute(1,2,0).contiguous().numpy()
        print(f"Detected {channel_map.shape[0]} channels, {templates.shape[0]} templates")

        oss_in.mkdir(parents=True, exist_ok=True)
        # crop core tensors
        crop_kilosort_output(templates, whitening_mat, channel_map,
                              range(channel_map.shape[0]), range(templates.shape[0]), oss_in)
        # save all other outputs
        np.save(oss_in / 'amplitudes.npy', amplitudes)
        np.save(oss_in / 'spike_times.npy', spike_times)
        np.save(oss_in / 'spike_templates.npy',    spike_templates)
        np.save(oss_in / 'spike_detection_templates.npy', spike_detection_templates)
        np.save(oss_in / 'templates.npy',          templates)
        np.save(oss_in / 'whitening_mat.npy',      whitening_mat)
        np.save(oss_in / 'channel_map.npy',        channel_map)
        np.save(oss_in / 'spike_positions.npy',    spike_positions)
        np.savez(oss_in / 'ops.npz', **ops)
        np.save(oss_in / 'Wall3.npy', Wall3)
        np.save(oss_in / 'ctc.npy', ctc)
        np.save(oss_in / 'ctc_permuted.npy', ctc_p)
        np.save(oss_in / 'wPCA.npy', wPCA)
        np.save(oss_in / 'wPCA_permuted.npy', wPCA_p)
        np.save(oss_in / 'cluster_centroids.npy', centroids)
        np.save(oss_in / 'hp_filter.npy', hpf)
        np.save(oss_in / 'drift_matrix.npy', drift_matrix)
        # Median reference groups Kilosort trained with (absent = stock global median)
        cg_src = ks_out / 'car_groups.npy'
        car_groups = np.load(cg_src) if cg_src.exists() else np.zeros(channel_map.shape[0], dtype=np.int32)
        np.save(oss_in / 'car_groups.npy', car_groups.astype(np.int32))
        np.save(oss_in / 'iCC.npy', iCC)
        np.save(oss_in / 'iU.npy', iU)
        np.save(oss_in / 'Ucc.npy', Ucc)
        np.save(oss_in / 'xc.npy', xc)
        np.save(oss_in / 'yc.npy', yc)
        np.save(oss_in / 'preclustered_template_waveforms.npy', pre_wf)
        np.save(oss_in / 'cluster_centroids_pca.npy', cluster_centroids_pca)
        # Training-recording match baseline (absent on oss_input dirs built by an
        # older launcher -- the GUI falls back to showing raw match strength).
        mb_src = ks_out / 'match_baseline.npy'
        if mb_src.exists():
            np.save(oss_in / 'match_baseline.npy', np.load(mb_src))
        else:
            print("Note: match_baseline.npy not in kilosort output; "
                  "re-run Kilosort4 to enable % of training match.")
        # --- Real-time drift estimation exports ---
        iKxx = np.ascontiguousarray(np.array(ops['iKxx'], dtype=np.float32))
        np.save(oss_in / 'iKxx.npy', iKxx)

        binning_depth = float(ops['settings']['binning_depth'])
        sig_interp    = float(ops['settings']['sig_interp'])
        yc_min = float(np.min(yc)); yc_max = float(np.max(yc))

        # --- Universal-template detector for the LIVE drift estimator ---
        # These come from the snapshot taken mid-Kilosort, not from ops.npy /
        # ops.npz -- see snapshot_drift_stage() for why those are the wrong
        # detector. Without them the C++ estimator refuses to enable itself.
        if drift_snapshot is not None:
            for key in ('ud_wTEMP', 'ud_iC', 'ud_iC2', 'ud_weigh',
                        'ud_ycup', 'ud_xcup', 'ks_F0', 'ks_dshift'):
                np.save(oss_in / f'{key}.npy', drift_snapshot[key])
            print(f"  drift detector: {drift_snapshot['scalars']['n_filters']} "
                  f"template positions, ks_F0 {drift_snapshot['ks_F0'].shape}")
        else:
            print("  NOTE: universal-template drift exports not written "
                  "(Kilosort4 was not re-run this launch). Live drift "
                  "estimation will stay off until you re-run Kilosort4 once.")

        with open(oss_in / 'misc.txt', 'w') as f:
            f.write(f"nt0min:{ops['nt0min']}\n")
            f.write(f"numNearestChans:{ops['settings']['nearest_chans']}\n")
            f.write(f"Th_learned:{ops['Th_learned']}\n")
            f.write(f"duplicate_spike_bins:{ops['duplicate_spike_bins']}\n")
            f.write(f"sig_interp:{sig_interp}\n")
            f.write(f"binning_depth:{binning_depth}\n")
            f.write(f"yc_min:{yc_min}\n")
            f.write(f"yc_max:{yc_max}\n")
            f.write(f"dshift_last:{float(dshift[-1])}\n")
            if drift_snapshot is not None:
                for k, v in drift_snapshot['scalars'].items():
                    f.write(f"{k}:{v}\n")

        # Optionally subset preclustered templates for lower-spec GPUs and/or
        # to focus on a probe channel range. The two filters compose: the
        # channel range narrows the field first, then max_templates caps the
        # most active survivors.
        try:
            n_keep = int(MAX_TEMPLATES[i]) if MAX_TEMPLATES else 0
        except (ValueError, TypeError):
            n_keep = 0
        chan_range = parse_channel_range(CHANNEL_RANGES[i]) if CHANNEL_RANGES else None
        if n_keep > 0 or chan_range is not None:
            subset_oss_input_inplace(oss_in, n_keep, metric="spikes",
                                     allow_below_t=True, channel_range=chan_range)

        OSS_DIRS.append(str(oss_in) + '\\')
    return OSS_DIRS

if __name__ == '__main__':
    run_online_multi()
