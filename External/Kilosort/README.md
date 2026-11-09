# Kilosort 4.0.13 (Live Spike Sorter fork)

Upstream [Kilosort](https://github.com/MouseLand/Kilosort) 4.0.13 with a small
patch so that LSS can consume its intermediate tensors.

The patch adds five `np.save` calls, writing into the Kilosort results
directory:

| Tensor | Written by |
| :--- | :--- |
| `Wall3.npy` | `run_kilosort.py` |
| `wPCA.npy` | `run_kilosort.py` |
| `ctc.npy` | `template_matching.py` |
| `imap.npy` | `template_matching.py` |
| `cluster_centroids.npy` | `io.py` |

`OnlineSpikesV2.cu` loads `Wall3.npy`, `ctc.npy` and `cluster_centroids.npy`
directly. Stock Kilosort 4.0.13 writes none of them, so a stock install fails
at run time with `FileNotFoundError: Wall3.npy`.

Patched files: `io.py`, `run_kilosort.py`, `template_matching.py`,
`clustering_qr.py`, `gui/sorter.py`. Everything else is untouched upstream.
The patch also prints `[CALVIN DEBUG] ...` lines, which double as a runtime
check that the fork is the copy actually being imported.

This directory is a **read-only reference** -- treat it as vendored upstream
code. It is installed into your environment by `pip install -r requirements.txt`
from the repository root.
