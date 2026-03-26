import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def l1_difference(mat1, mat2):
    # Ensure both matrices have the same shape
    if mat1.shape != mat2.shape:
        raise ValueError("The two matrices must have the same shape.")
    # Compute the L1 difference (sum of absolute differences)
    return np.sum(np.abs(mat1 - mat2))

def read_cuda_batch(directory, filename, num_chans):
    with open(directory / filename, "r") as file:
        line = file.readline().strip()

    # Split the line into individual string values and convert to float
    values = [float(val) for val in line.split()]

    # Create a flat NumPy array from the float values
    data_flat = np.array(values)
    print(data_flat[0:10])
    # Reshape the flat array to the desired shape
    assert data_flat.size % num_chans == 0
    data_reshaped = data_flat.reshape((data_flat.size // num_chans, num_chans))
    print(f"Read {data_flat.size // num_chans} samples on {num_chans} channels.")
    return data_reshaped

def read_raw_batch(directory, filename, num_chans, start, num_samps):
	print(f"Reading raw batch from {directory / filename} from sample {start} to {end} on {num_chans} channels")
	with open(directory / filename, 'rb') as fidInput:
		batch = np.fromfile(fidInput, dtype=np.int16, offset=start * (num_chans + 1) * 2, count=(num_chans + 1) * num_samps)

		if batch.size == 0 or batch.size != (num_chans + 1) * num_samps:
			print(f"Read {batch.size} entries from {directory / filename}, expected {num_chans * num_samps}")

		print(batch[0:10])
		batch = np.reshape(batch, (num_chans + 1, num_samps), order='F')[ :num_chans, :].T
		batch = batch.astype(np.float32)
		return batch

num_chans = 384
cuda_output_dir = Path("C:/", "SGL_DATA", "05_31", "cuda_output")
bin_dir = Path("C:/", "SGL_DATA", "05_31", "imec_raw")
bin_file = "240531_g0_t0.imec0.ap.bin"
pattern = re.compile(r'^batch_(\d+)_(\d+)\.txt$')

for filename in os.listdir(cuda_output_dir):
	# Check if the filename matches the pattern
	match = pattern.match(filename)
	if match:
		# Extract integers from the filename
		start = int(match.group(1))
		end = int(match.group(2))
		print(f"Verifying CUDA batch from filename: {filename}, start sample: {start}, end sample: {end}")
		cuda_batch = read_cuda_batch(cuda_output_dir, filename, num_chans)
		print(f"cuda_batch read with shape {cuda_batch.shape}")
		num_samps = cuda_batch.shape[0]
		raw_batch = read_raw_batch(bin_dir, bin_file, num_chans, start, num_samps)
		print(f"raw_batch read with shape {raw_batch.shape}")
		print(f"L1 difference between batches: {l1_difference(cuda_batch, raw_batch)}")
		plt.subplot(211)
		plt.plot(cuda_batch)
		plt.title("CUDA batch")
		plt.subplot(212)
		plt.plot(raw_batch)
		plt.title("Raw batch")
		plt.show()


'''
self.logger.debug(f"Fetching data from {self.bin_file} from sample {start} to {start + self.W} on {self.C} channels")
with open(self.bin_file, 'rb') as fidInput:
	self.batch = np.fromfile(fidInput, dtype=np.int16, offset=start * self.raw_C * 2, count=self.raw_C * self.W)

	if self.batch.size == 0 or self.batch.size != self.raw_C * self.W:
		self.logger.error(f"Read {self.batch.size} entries from {self.bin_file}, expected {self.C * self.W}")

	self.last_processed_sample = start + len(self.batch) // self.raw_C
	self.batch = np.reshape(self.batch, (self.raw_C, self.W), order='F')[self.channel_mask, :]
	self.batch = self.batch.astype(self.dtype)
	self.ks_batch = copy.deepcopy(torch.from_numpy(self.batch))
	self.batch = self.batch.flatten(order='F')
	self.d_batch = cp.array(self.batch)
	samples_read = self.batch.size // self.C
	return samples_read
'''