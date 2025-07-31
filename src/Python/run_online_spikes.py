import sys
from pathlib import Path
from crop_methods import crop_kilosort_output, parse_bin_meta_file
import kilosort
import numpy as np
import torch
import subprocess
import time
import os
import shlex
from kilosort.preprocessing import get_drift_matrix
from kilosort.template_matching import prepare_extract
import copy
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import json

# -------------------------------
# Persistence config
# -------------------------------
STATE_FILE = Path(__file__).parent / "multi_gui_state.json"

# -------------------------------
# GUI for Multiple Sorters
# -------------------------------
root = tk.Tk()
root.title("Select Directories and Files for Multiple Sorters")

# Number of sorters
num_sorters_var = tk.IntVar(value=1)
spin = tk.Spinbox(root, from_=1, to=16, textvariable=num_sorters_var, width=5, command=lambda: update_tabs())
tk.Label(root, text="Number of sorters:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
spin.grid(row=0, column=1, padx=5, pady=5, sticky="w")

# Notebook for sorter tabs
toolkit = ttk.Notebook(root)
toolkit.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

# Per-sorter storage
base_path_vars, ks_output_dir_vars = [], []
bin_file_vars, meta_file_vars, chanmap_file_vars = [], [], []
rerun_ks_vars, sdm_vars = [], []
sdm_ip_vars, sdm_port_vars = [], []
file_frames, sdm_frames = [], []

# Hints dictionary
HINTS = {
    "BASE": "This directory should contain 'imec_raw' folder for your recording.",
    "KS_OUTPUT": "Select where the Kilosort output is or will be stored.",
    "BIN": "The location of your recording (.bin file).",
    "META": "The location of your recording's metadata (.meta file).",
    "CHANMAP": "The location of your probe's channel map (.mat file).",
    "SDM": "Send decoder output to stimulus display machine?"
}

def show_hint(key):
    messagebox.showinfo("Hint", HINTS[key])

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
                    file_frames, sdm_frames):
            del lst[n:]

    # Append new entries if increasing
    for i in range(old, n):
        rerun_ks_vars.append(tk.BooleanVar(value=False))
        base_path_vars.append(tk.StringVar(value=str(Path.home())))
        ks_output_dir_vars.append(tk.StringVar(value=""))
        bin_file_vars.append(tk.StringVar(value=""))
        meta_file_vars.append(tk.StringVar(value=""))
        chanmap_file_vars.append(tk.StringVar(value=""))
        sdm_vars.append(tk.BooleanVar(value=False))
        sdm_ip_vars.append(tk.StringVar(value=""))
        sdm_port_vars.append(tk.StringVar(value=""))
        file_frames.append(None)
        sdm_frames.append(None)

    # Rebuild all tabs
    for tab in toolkit.tabs():
        toolkit.forget(tab)
    for i in range(n):
        frame = ttk.Frame(toolkit)
        toolkit.add(frame, text=f"Sorter {i+1}")
        build_tab(frame, i)

# Construct UI for a single sorter tab
def build_tab(frame, idx):
    row = 0
    # Rerun checkbox
    tk.Checkbutton(
        frame, text="Rerun Kilosort4", variable=rerun_ks_vars[idx],
        command=lambda i=idx: toggle_rerun(i)
    ).grid(row=row, column=0, columnspan=4,
           padx=5, pady=5, sticky="w")
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

    # SDM toggle
    tk.Checkbutton(
        frame, text="SDM?", variable=sdm_vars[idx],
        command=lambda i=idx: toggle_sdm(i)
    ).grid(row=row, column=0, columnspan=4,
           sticky="w", padx=5, pady=5)
    tk.Button(
        frame, text="?", command=lambda i=idx: show_hint("SDM"), width=3
    ).grid(row=row, column=3)
    row += 1

    # SDM subframe
    sdm_frame = tk.Frame(frame, borderwidth=1, relief="sunken")
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
    if sdm_vars[idx].get():
        sdm_frame.grid(row=row, column=0, columnspan=4,
                       padx=5, pady=5)
    row += 1

    # File selection subframe
    f_frame = tk.Frame(frame, borderwidth=1, relief="sunken")
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
    if rerun_ks_vars[idx].get():
        f_frame.grid(row=row, column=0, columnspan=4,
                     padx=5, pady=5)

# Finish and error widgets
def create_finish_widgets():
    global finish_button, error_text
    finish_button = tk.Button(root, text="Finish", command=finish_and_quit)
    finish_button.grid(row=2, column=0, columnspan=4, padx=5, pady=5)
    error_text = tk.Text(root, height=4, width=60, fg="red")
    error_text.grid(row=3, column=0, columnspan=4, padx=5, pady=5)
    error_text.configure(state="disabled")

# Destroy GUI on finish
def finish_and_quit():
    root.destroy()

# Initialize GUI (with state loading and applying)
def main():
    # Load previous state
    state = None
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            num_sorters_var.set(state.get("num_sorters", num_sorters_var.get()))
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
            if state["rerun_flags"][i]:
                toggle_rerun(i)
            sdm_vars[i].set(state["sdm_flags"][i])
            if state["sdm_flags"][i]:
                toggle_sdm(i)
            sdm_ip_vars[i].set(state["sdm_ips"][i])
            sdm_port_vars[i].set(state["sdm_ports"][i])

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
    SDM_FLAGS = [v.get() for v in sdm_vars]
    SDM_IPS = [v.get().strip() for v in sdm_ip_vars]
    SDM_PORTS = [v.get().strip() for v in sdm_port_vars]

    # Save current state
    state = {
        "num_sorters": n,
        "base_paths": [str(p) for p in BASE_PATHS],
        "ks_output_dirs": [str(d) for d in KS_OUTPUT_DIRS],
        "bin_files": [str(p) for p in BIN_FILES],
        "meta_files": [str(p) for p in META_FILES],
        "chanmap_files": [str(p) for p in CHANMAP_FILES],
        "rerun_flags": RERUN_FLAGS,
        "sdm_flags": SDM_FLAGS,
        "sdm_ips": SDM_IPS,
        "sdm_ports": SDM_PORTS
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Could not save GUI state: {e}")

    # Curate OSS input dirs
    OSS_DIRS = curate_oss_input_dir(BASE_PATHS, KS_OUTPUT_DIRS, BIN_FILES,
                                     META_FILES, CHANMAP_FILES, RERUN_FLAGS)

    # Build and run C++ command
    arguments = {
        '--n_gpus': str(n),
        '--oss_input': OSS_DIRS,
        '--decoder_input': [str(bp / 'decoder_input') + '\\' for bp in BASE_PATHS],
        '--spikes_output': [str(bp / 'decoder_input' / 'spikeOutput.txt') for bp in BASE_PATHS],
        '--cuda_output_dir': [str(bp / 'cuda_output') + '\\' for bp in BASE_PATHS]
    }
    if any(SDM_FLAGS):
        arguments['--sdm_ip'] = SDM_IPS
        arguments['--sdm_port'] = SDM_PORTS

    cmd = [r"C:\Users\Spike Sorter\source\repos\OnlineSpikes_v2\x64\RELEASE\OnlineSpikes.exe"]
    for k, v in arguments.items():
        if isinstance(v, list):
            cmd.append(k)
            cmd.extend(v)
        else:
            cmd.extend([k, str(v)])

    print("Running OSS with:", shlex.join(cmd))
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()

def curate_oss_input_dir(BASE_PATHS, KS_OUTPUT_DIRS, BIN_FILES,
                         META_FILES, CHANMAP_FILES, RERUN_FLAGS):
    OSS_DIRS = []
    for i, base in enumerate(BASE_PATHS):
        ks_out = KS_OUTPUT_DIRS[i]
        if RERUN_FLAGS[i]:
            if not torch.cuda.is_available():
                print(f"Error: No GPU for sorter {i+1}")
                sys.exit(1)
            print(f"Starting kilosort for sorter {i+1}...")
            meta = parse_bin_meta_file(META_FILES[i])
            start = time.time()
            settings = {'data_dir': str(base / 'imec_raw'), 'n_chan_bin': meta['nSavedChans']}
            kilosort.run_kilosort(settings=settings,
                                  probe_name=CHANMAP_FILES[i],
                                  results_dir=str(ks_out),
                                  filename=BIN_FILES[i])
            print(f"Kilosort sorter {i+1} took {time.time() - start:.2f} s")

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
        centroids = [v for v in centroids.values()]
        hpf            = np.array(ops['preprocessing']['hp_filter'])
        dshift         = np.array(ops['dshift'])
        drift_slope, _ = np.polyfit(np.arange(0, 30), dshift[-30:], 1)

        if abs(2 * drift_slope) >= 0.01: # If more than 0.01 micron drift per second in the last minute, it is unstable
            print("WARNING: Probe not yet stable. Spike sorter quality may be impacted by drift.")

        drift_matrix   = np.array(get_drift_matrix(ops, dshift[-1], device='cpu')).T
        iCC, iU, Ucc  = prepare_extract(ops, torch.tensor(Wall3), ops['settings']['nearest_chans'], device='cpu')
        iCC = np.ascontiguousarray(iCC)
        iU  = np.ascontiguousarray(iU)
        Ucc = np.ascontiguousarray(Ucc)
        xc  = np.ascontiguousarray(np.array(ops['xc']))
        yc  = np.ascontiguousarray(np.array(ops['yc']))
        pre_wf = torch.einsum('ijk,jl->kil', torch.tensor(Wall3), torch.tensor(wPCA)).permute(1,2,0).contiguous().numpy()
        print(f"Detected {channel_map.shape[0]} channels, {templates.shape[0]} templates")

        oss_in = base / 'oss_input'
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
        np.save(oss_in / 'iCC.npy', iCC)
        np.save(oss_in / 'iU.npy', iU)
        np.save(oss_in / 'Ucc.npy', Ucc)
        np.save(oss_in / 'xc.npy', xc)
        np.save(oss_in / 'yc.npy', yc)
        np.save(oss_in / 'preclustered_template_waveforms.npy', pre_wf)
        with open(oss_in / 'misc.txt', 'w') as f:
            f.write(f"nt0min:{ops['nt0min']}\n")
            f.write(f"numNearestChans:{ops['settings']['nearest_chans']}\n")
            f.write(f"Th_learned:{ops['Th_learned']}\n")
            f.write(f"duplicate_spike_bins:{ops['duplicate_spike_bins']}\n")
        OSS_DIRS.append(str(oss_in) + '\\')
    return OSS_DIRS

if __name__ == '__main__':
    run_online_multi()
