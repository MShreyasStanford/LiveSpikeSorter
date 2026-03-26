import numpy as np
from pathlib import Path
import os
import h5py

hdf_path = Path("C:/", "SGL_DATA", "tg_20231206_MGS_RF.h5")

def load_selected_tensors(hdf_path, names):
    if not os.path.exists(hdf_path):
        print(f"Error: '{hdf_path}' not found.", file=sys.stderr)
        sys.exit(1)

    data = {}
    print(f"→ attempting to open HDF5 file: {hdf_path!r}")
    print("  exists:", os.path.exists(hdf_path))
    print("  size  :", os.path.getsize(hdf_path), "bytes")

    with h5py.File(hdf_path, 'r') as f:
        for full_name in names:
            if full_name not in f:
                print(f"Warning: '{full_name}' missing.", file=sys.stderr)
                continue

            arr = f[full_name][()]
            parts = full_name.split('/')
            # descend into nested dict, creating sub‑dicts as needed
            d = data
            for key in parts[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            # assign the leaf dataset
            d[parts[-1]] = arr

    return data

names = [
    # time‑vector
    "ts",

    # spikes & unit infos
    "imec0/spk",
    "imec1/spk",
    "imec0/unit_info/y",
    "imec1/unit_info/y",

    # trial timing
    "trial_info/edata_st1_acquired",
    "trial_info/edata_fixation_stable",
    "trial_info/esetup_probe_on",
    "trial_info/edata_st1_on",
    "trial_info/esetup_st1_rad",
    "trial_info/edata_st1_maintained",

    # probe metadata (you were missing these)
    "probe_info/edata_probe_on",
    "probe_info/esetup_probe_coord",
    "probe_info/esetup_probe_trial_index",
]

data = load_selected_tensors(str(hdf_path), names)

selected_units = np.flatnonzero(np.ones(data['imec0']['spk'].shape[2]) > 0)
selected_probes = np.flatnonzero(~np.isnan(data['probe_info']['edata_probe_on']))
data_to_plot = data['imec0']['spk']
ts = data['ts']
unique_x = np.unique(data['probe_info']['esetup_probe_coord'][selected_trials, 0])
unique_y = np.unique(data['probe_info']['esetup_probe_coord'][selected_trials, 1])

# Initial fixation
time_align_trial = 'edata_st1_on'
time_win_trial = [-0.5, 0]

# # Presaccadic
# time_align_trial = 'st1_sac_ini_time'
# time_win_trial = [-0.2, 0]


time_win_stim = [0.05, 0.3] # time window for calculating RFs after each probe (stimulus) onset

RF = np.zeros((len(unique_y), len(unique_x), data_to_plot.shape[2]))
RF_reps = np.zeros((len(unique_y), len(unique_x)))
n=0
for p in selected_probes:
    trial = int(data['probe_info']['esetup_probe_trial_index'][p]-1)
    probe_on_time = data['probe_info']['edata_probe_on'][p]
    time_align_trial_p = (data['trial_info'][time_align_trial] - data['trial_info']['edata_fixation_stable'])[int(data['probe_info']['esetup_probe_trial_index'][p]-1)]
    if (probe_on_time >= time_align_trial_p + time_win_trial[0]) and (probe_on_time < time_align_trial_p + time_win_trial[1]):
        n = n+1
        print(probe_on_time)
        data_in_time_win = data_to_plot[trial][(ts>=probe_on_time+time_win_stim[0])&(ts<probe_on_time+time_win_stim[1])]
        which_x = np.flatnonzero(np.abs(unique_x - data['probe_info']['esetup_probe_coord'][p, 0]) < 1e-4)[0]
        which_y = np.flatnonzero(np.abs(unique_y - data['probe_info']['esetup_probe_coord'][p, 1]) < 1e-4)[0]
        RF[which_y, which_x] = RF[which_y, which_x] + np.nanmean(data_in_time_win, axis=0)
        RF_reps[which_y, which_x] = RF_reps[which_y, which_x] + 1

RF = RF / RF_reps[:, :, None]
