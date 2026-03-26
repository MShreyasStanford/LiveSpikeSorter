import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from kilosort.io import load_ops

def plot_drift_for_dirs(base_dir, prefix):
    base_dir = Path(base_dir)
    # Find subdirectories that start with the given prefix
    dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not dirs:
        print("No directories found with the given prefix.")
        return

    n = len(dirs)
    # Calculate grid size (roughly square)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
    axes = axes.flatten()

    for i, d in enumerate(dirs):
        kilosort_dir = d / "Kilosort4"
        print(f"Plotting drift for {kilosort_dir}.")
        try:
            ops = load_ops(kilosort_dir / 'ops.npy')
        except Exception as e:
            print(f"Could not load ops.npy from {kilosort_dir}: {e}")
            continue

        dshift = ops['dshift']
        # Use Nbatches if present; otherwise, infer from dshift length
        Nbatches = ops.get('Nbatches', len(dshift))
        time = np.arange(Nbatches) * 2  # Assuming a batch represents 2 sec.

        ax = axes[i]
        ax.plot(time, dshift)
        ax.set_xlabel('Time (sec.)')
        ax.set_ylabel('Drift (um)')
        ax.set_title(f"Drift for {kilosort_dir}")

    # Remove any extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Example usage:
base_directory = Path('C:/', 'SGL_DATA')
prefix_string = "jo"
plot_drift_for_dirs(base_directory, prefix_string)