import os
from pathlib import Path
import warnings

def crop_bin_file(
    input_file: Path | str,
    output_file: Path | str,
    start_sample: int,
    end_sample: int,
    num_channels: int,
    buffer_samples: int = 64 * 256
) -> None:
    """
    Crop a SpikeGLX-style .bin file to only include samples
    in [start_sample, end_sample).

    Parameters
    ----------
    input_file : Path or str
        Path to the original .bin file.
    output_file : Path or str
        Path where the cropped .bin will be written.
    start_sample : int
        First sample index to include (0-based).
    end_sample : int
        One-past-the-last sample index to include.
    num_channels : int
        Number of interleaved channels in the data.
    buffer_samples : int, optional
        Number of timepoints to read per chunk (per channel).
        Defaults to 64 * 256.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_per_sample = 2  # int16
    frame_size = num_channels * bytes_per_sample

    # Calculate byte offsets
    start_offset = start_sample * frame_size
    total_frames = end_sample - start_sample

    with open(input_path, "rb") as fid_in, open(output_path, "wb") as fid_out:
        # Jump to the first sample we want
        fid_in.seek(start_offset, os.SEEK_SET)

        frames_remaining = total_frames
        while frames_remaining > 0:
            # How many frames to read in this chunk?
            chunk_frames = min(buffer_samples, frames_remaining)
            bytes_to_read = chunk_frames * frame_size

            data = fid_in.read(bytes_to_read)
            actual_bytes = len(data)
            if actual_bytes == 0:
                warnings.warn(
                    f"Reached EOF after writing "
                    f"{total_frames - frames_remaining} samples."
                )
                break

            # If we got less than requested, still write what we have
            fid_out.write(data)
            frames_remaining -= (actual_bytes // frame_size)

    # Optionally, you could return the number of frames written
