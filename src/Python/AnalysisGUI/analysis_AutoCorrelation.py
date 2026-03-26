import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

__analysis_metadata__ = {
    "name": "Autocorrelation",
    "description": "Compute autocorrelation for a specific neuron and fit exponential decay.",
    "parameters": [
        ("Spike File (.txt)", "FilePath", r"^(\d+),(\d+),(\d+\.\d+)(?:,[^,]+)*$"),
        ("Template Index", "int")
    ]
}

def run(base_path, spike_output_file, neuron_index):
    # Try using CuPy for acceleration
    use_cupy = False
    try:
        import cupy as cp
        xp = cp
        use_cupy = True
        print("Using CuPy for acceleration.")
    except ImportError:
        xp = np
        print("CuPy not available; using NumPy.")

    # Load spike data
    spike_times, spike_templates = [], []
    with open(spike_output_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                t = int(row[0]); templ = int(row[1])
            except (ValueError, IndexError):
                continue
            spike_times.append(t)
            spike_templates.append(templ)

    spike_times = np.array(spike_times)
    spike_templates = np.array(spike_templates)

    # Filter and sort times for the target neuron
    times = xp.asarray(spike_times)
    templates = xp.asarray(spike_templates)
    times_j = xp.sort(times[templates == neuron_index])

    # Sliding-window binning
    window_size = 500    # samples per bin
    step = 10            # samples between window starts
    max_time = int(times_j.max()) if times_j.size > 0 else 0
    starts = xp.arange(0, max_time - window_size + 1, step)

    # Spike counts per window
    idx_start = xp.searchsorted(times_j, starts, side='left')
    idx_end = xp.searchsorted(times_j, starts + window_size, side='left')
    counts = (idx_end - idx_start).astype(float)

    # Autocorrelation
    n = counts.size
    max_lag = min(1000, n - 1)
    mu = counts.mean()
    var = counts.var()
    full_corr = xp.correlate(counts - mu, counts - mu, mode='full')
    autocov = full_corr[n - 1 : n - 1 + max_lag + 1] / n
    autocorr = autocov / var

    # Convert to NumPy for peak detection and fitting
    if use_cupy:
        autocorr = cp.asnumpy(autocorr)
    lags = np.arange(max_lag + 1)
    sample_lags = lags * step

    # Detect first peak after initial drop (lag > 0)
    from scipy.signal import find_peaks
    # ignore lag 0 when finding peaks
    peaks, _ = find_peaks(autocorr[1:])
    if peaks.size == 0:
        print("No peak detected after initial drop-off.")
        peak_idx = 0
    else:
        peak_idx = peaks[0] + 1
        
    peak_lag = sample_lags[peak_idx]
    peak_val = autocorr[peak_idx]

    # Fit exponential decay to tail after the peak
    from scipy.optimize import curve_fit
    # Define decay model
    def exp_decay(x, A, tau, C):
        return A * np.exp(-x / tau) + C

    tail_lags = sample_lags[peak_idx:]
    tail_corr = autocorr[peak_idx:]
    # Initial parameter guesses: A=peak_val, tau=mean tail lag, C=0
    p0 = [peak_val, tail_lags.mean(), 0]
    try:
        params, _ = curve_fit(exp_decay, tail_lags, tail_corr, p0=p0)
        A_fit, tau_fit, C_fit = params
    except Exception as e:
        print(f"Exponential fit failed: {e}")
        A_fit, tau_fit, C_fit = (np.nan, np.nan, np.nan)

    # Plotting
    plt.figure()
    plt.plot(sample_lags, autocorr, label='Autocorrelation')
    # Mark the detected peak
    plt.plot(peak_lag, peak_val, 'ro', label='First Peak')
    # Overlay exponential fit
    if not np.isnan(tau_fit):
        fit_curve = exp_decay(tail_lags, A_fit, tau_fit, C_fit)
        plt.plot(tail_lags, fit_curve, '--',
                 label=f'Exp decay (tau={tau_fit:.1f} samples)')

    plt.xlabel('Lag (samples)')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorr & Exp Decay Fit (neuron {neuron_index})')
    plt.legend()
    plt.tight_layout()

    # Save figure
    out_file = Path(base_path) / f"autocorr_decay_neuron_{neuron_index}.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")
