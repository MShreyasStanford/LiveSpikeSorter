# Offline Spike Sorting - Setup & Run Guide

This guide covers running the Live Spike Sorter (LSS) against a previously recorded `.ap.bin` file. It assumes you already have the recording you want to sort and a Windows machine with a CUDA-capable GPU to run the sorter on.

---

## A - One-Time Machine Setup

You only need to do these steps once per machine. Skip to [A6](#a6-install-python-310--environment-setup) if all your prerequisites are installed. Skip to [PART B](#part-b----session-workflow) if your machine is already set up.

### A1. INSTALL VISUAL STUDIO 2022

LSS builds with VS2022 Build Tools and the v143 platform toolset. Other VS versions are not supported by the `.vcxproj`.

- Install Visual Studio 2022 Build Tools (Community is fine too).
- Under "Workloads", check "Desktop development with C++".
- Under "Individual components", make sure these are selected:
  - MSVC v143 - VS 2022 C++ x64/x86 build tools
  - Windows 10 SDK 10.0.17763.0

### A2. INSTALL CUDA 11.8

LSS targets CUDA 11.8 specifically.

- Download the CUDA 11.8 installer from NVIDIA's archive.
- Install with default options *after* Visual Studio 2022 is already installed (or rerun the CUDA installer's "Visual Studio Integration" component afterward) so its MSBuild integration files land in VS2022's BuildCustomizations folder automatically. Default path:

  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\
  ```

- Verify the system environment variable `CUDA_PATH` points there with:

  ```
  echo %CUDA_PATH%
  ```

  - Alternatively, search for "edit the system environment variables" using Windows search, open it, click "Environment Variables" under the Advanced tab and ensure `CUDA_PATH` is listed with the correct path under System Variables:

    | Variable | Value |
    |----------|-------|
    | `CUDA_PATH` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8` |

### A3. INSTALL cuDNN 8.x

Use a cuDNN 8.x release compatible with CUDA 11.8.

After downloading the cuDNN zip from NVIDIA:

- Copy `bin\*.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\`
- Copy `include\*.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\`
- Copy `lib\x64\*.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\`

### A4. CLONE THE LSS REPO

Clone the LSS repo somewhere convenient. Conventional location:

```
C:\Users\<you>\source\repos\LiveSpikeSorter\
```

### A5. INSTALL SpikeGLX

SpikeGLX is an external dependency, downloaded separately.

- Download the latest release from <http://billkarsh.github.io/SpikeGLX/>, e.g. `App/Release_v20251218-api4.zip`
- The extracted files contain two executables: `SpikeGLX_STD.exe` (needs NI-DAQmx drivers) and `SpikeGLX_STD_NISIM.exe` (no NI dependency). For offline replay with no NI hardware, you can use `SpikeGLX_STD_NISIM.exe`.

### A6. INSTALL PYTHON 3.10 + ENVIRONMENT SETUP

Install Python 3.10 from python.org. Our dependencies do not currently support newer versions.

- At install time, leave "Add Python to PATH" checked.

Then create a virtual environment and install the dependencies. Run these from the repo root you cloned in A4:

```
py -3.10 -m venv venv

venv\Scripts\activate
pip install -r requirements.txt
```

### A7. BUILD LSS (Release | x64)

Open `OnlineSpikes.vcxproj` in VS2022. Set the configuration dropdown to:

```
Release  |  x64
```

Then Build -> Rebuild Solution.

Output: `<repo>\x64\RELEASE\OnlineSpikes.exe`

---

## PART B -- Session Workflow

Follow these steps every time you want to sort a recording.

### B1. CREATE A SESSION DIRECTORY

Pick a session folder. Convention:

```
C:\Users\<you>\OSSExperiments\MM_DD_YY\
```

This will be your `BASE_PATH`.

### B2. COPY OR LOCATE THE TRAINING RECORDING

Put (or note the path to) the `.ap.bin` and `.ap.meta` files that LSS will use as both the Kilosort training set and the offline replay source.

```
BASE_PATH/
└── [RecordingName]_g0/
    └── [RecordingName]_g0_imec0/
        ├── ...t0.imec0.ap.bin     <- Raw AP-band binary
        └── ...t0.imec0.ap.meta    <- Metadata
```

### B3. RUN KILOSORT 4 TO PRODUCE TEMPLATES

LSS does not learn templates online. You must run Kilosort4 on the recording first to produce templates, the whitening matrix, drift matrices, etc.

(Ensure your environment is activated)

```
cd <repo>\src\Python
python run_online_spikes.py
```

A GUI appears. For each sorter/probe, fill in:

- **Base Path**: your session directory (e.g. `C:\Users\<you>\Spike Sorter\OSSExperiments\03_04_26`)
- **KS Output Dir**: typically `oss_training` inside your base path
- **Bin File**: the `.ap.bin` from your training recording
- **Meta File**: the matching `.ap.meta`
- **Channel Map**: if using a custom probe layout
- **Max templates**: caps the amount of templates LSS can use for low-spec GPUs. Discarding starts with least active templates (by spike count). Set to 0 by default to use all templates.
- **SDM settings**: IP/port for stimulus display machine, if sending decoder feedback

Check "Re-run Kilosort4" and let it run. Monitor progress in the terminal. (Duration should be similar to length of training recording.)

When finished, the launcher automatically crops Kilosort's output into `BASE_PATH/oss_input/` for LSS to load at startup.

Resulting layout:

```
BASE_PATH/
+-- [RecordingName]_g0/        <- Unchanged, your original recording
+-- oss_training/              <- Full Kilosort 4 output
+-- oss_input/                 <- Cropped subset that LSS reads:
    +-- templates.npy
    +-- whiteningMat.npy
    +-- channelMap.npy
    +-- templateMap.npy
    +-- Wall3.npy
    +-- ctc.npy
    +-- wPCA.npy
    +-- wPCA_permuted.npy
    +-- drift_matrix.npy
    +-- iCC.npy
    +-- iU.npy
    +-- Ucc.npy
    +-- preclustered_template_waveforms.npy
    +-- cluster_centroids_pca.npy
    +-- hp_filter.npy
    +-- xc.npy / yc.npy
    +-- misc.txt
    +-- ops.npz
```

### B4. (OPTIONAL) UNCHECK "Re-run Kilosort4" FOR SUBSEQUENT RUNS

Once `oss_input/` is populated, subsequent LSS runs on the same recording can skip the Kilosort step. The launcher GUI has a checkbox; you can toggle it off after the first successful Kilosort run.

### B5. START SpikeGLX WITH A SIMULATED PROBE REPLAYING YOUR .ap.bin

1. Launch `SpikeGLX_STD_NISIM.exe` (or `SpikeGLX_STD.exe` if you have NI drivers installed).

2. Open the Configure Acquisition dialog (File -> New Acquisition).

3. In the probe table, add a row using the "Sim Probes" option. Point it at your `.ap.bin`. Example configuration:

   ```
   Slot: 40   Port: 1   Dock: 1   PN: NP:1030   Cal: Y
   ```

4. Click detect, then verify and save the configuration. Press "Run" and SpikeGLX should transition into a streaming/acquisition state.

5. Open Options -> Command Server Settings. Set:

   ```
   IP Address: 127.0.0.1
   Port:       4142
   ```

   Enable the server. Leave SpikeGLX running.

### B6. LAUNCH LSS AND POINT IT AT THE SAME 127.0.0.1:4142

Ensure your IP Address and Port are matching (if needed, relaunch):

```
cd <repo>\src\Python
py -3.10 run_online_spikes.py
```

LSS launches automatically after Kilosort completes.

### B7. VERIFY OUTPUT

Each line of `spikeOutput.txt` is one spike:

```
sample_time, template_id, amplitude, y_position
```

E.g.:

```
30028732,14,1.0635,1040.57
30028735,787,4.63187,1040.6
```

### B8. EXITING

The sorter does not auto-exit when the `.ap.bin` runs out -- SpikeGLX keeps the stream "running" until you stop it. To end the session:

1. In SpikeGLX, press Stop to end the run.
2. In LSS, close the LSS windows.

---

## What Gets Written During Sorting

```
BASE_PATH/
+-- [RecordingName]_g0/        <- Unchanged (your original recording)
+-- oss_training/              <- Unchanged (Kilosort output)
+-- oss_input/                 <- Unchanged (LSS reads at startup)
+-- cuda_output/               <- WRITTEN CONTINUOUSLY by the GPU sorter
|   +-- spikeOutput.txt        Each line: sample_time, template_id,
|                              amplitude, y_position
+-- decoder_input/             <- Used by decoder if decoding is enabled
    +-- spikeOutput.txt
```

If decoding is enabled (`bIsDecoding=true`), the decoder ALSO writes:

| File | Description |
|------|-------------|
| `binnedSpikes.txt` | Binned spike counts for training (once) |
| `binnedSpikesScaled.txt` | Normalized version of above (once) |
| `predictions.txt` | Decoder predictions, continuous |
| `log.txt` | Diagnostic log |

---

## Complete Directory Structure (After a Full Session)

```
BASE_PATH/
│
├── [RecordingName]_g0/              <- SpikeGLX training recording
│   ├── [RecordingName]_g0_imec0/
│   │   ├── ...t0.imec0.ap.bin          Raw AP-band binary
│   │   └── ...t0.imec0.ap.meta         Metadata
│   ├── [RecordingName]_g0_imec1/       (if multi-probe)
│   └── ...t0.nidq.bin / .meta          Auxiliary channels
│
├── oss_training/                    <- Kilosort 4 output
│   ├── spike_times.npy
│   ├── templates.npy
│   ├── whitening_mat.npy
│   └── ...
│
├── oss_input/                       <- Cropped for real-time sorter
│   ├── templates.npy
│   ├── whitening_mat.npy
│   ├── ops.npz
│   └── ...
│
├── cuda_output/                     <- Real-time spike detections
│   └── spikeOutput.txt
│
└── decoder_input/                   <- Decoder I/O
    └── spikeOutput.txt
```
