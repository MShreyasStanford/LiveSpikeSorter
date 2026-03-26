import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
import subprocess
import cupy as cp
import torch
import torch.nn as nn
import random
import math
import scipy

DRIVE_PATH = Path("C:/")
BASE_PATH = Path(DRIVE_PATH, "SGL_DATA", "05_31")
BIN_DIR = Path(BASE_PATH, "imec_raw")
KS_OUTPUT_DIR = Path(BIN_DIR, "kilosort4_output_test")
DECODER_INPUT_DIR = Path(BASE_PATH, "decoder_input")
CROPPED_OUTPUT_DIR = Path(BASE_PATH, "oss_training")
CROPPED_OUTPUT_DIR_1 = Path(BASE_PATH, "oss_training_1")
CROPPED_OUTPUT_DIR_2 = Path(BASE_PATH, "oss_training_2")

BIN_FILE = "240531_g0_t0.imec0.ap.bin"
BIN_META_FILE = "240531_g0_t0.imec0.ap.meta"
CHANNEL_MAP_FILE = "neuropixels_NHP_channel_map_dev_staggered_v1.mat"
CPP_MAIN_FILE = Path(r"C:\Users\Spike Sorter\source\repos\OnlineSpikes_v2\x64\RELEASE", "OnlineSpikes.exe")
SPIKES_OUTPUT_FILE = Path(BASE_PATH, "sorter_output", "spikeOutput.txt")


def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfilt(sos, data)
    return filtered_data

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfilt(sos, data)
    return filtered_data

def plot_recording(data):
    samples, channels = data.shape
    plt.figure(figsize=(15, 10))  # Set the figure size

    # Create a time axis for the data
    time = np.arange(samples)
    # Plot each channel
    for i in range(channels):
        offset = 2 * i
        plt.plot(time, (data[:, i] + offset) * 50)

    plt.title('Recording with Multiple Channels')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (arbitrary units)')
    plt.legend(loc='upper right')
    plt.grid(True)  # Add a grid for better readability
    plt.show()

def extract_submatrix(A, indices, M, W, C, T):
    num_cols = len(indices)
    num_rows = W * C
    
    g_A = np.zeros(num_rows * num_cols, dtype=np.float32)

    count = 0
    for i, index in enumerate(indices):
        sample_index = index // T
        template_index = index % T

        print(f"Adding template {template_index} at sample {sample_index} to submatrix")
        print(f"Memcopying to indices of g_A starting at {i * W * C + sample_index * C} to index {i * W* C + sample_index * C + min(M, W - sample_index) * C}")
        for chansamp_index in range(min(M, W - sample_index) * C):
            if A[template_index * M * C + chansamp_index] != 0.0:
                count += 1
            g_A[i * W * C + sample_index * C + chansamp_index] = A[template_index * M * C + chansamp_index]

    print(f"Copied over {count} non-zero entries")
    return np.reshape(g_A, (num_rows, num_cols), order='F')
    
def construct_A(templates):
    T = templates.shape[0]
    M = templates.shape[1]
    C = templates.shape[2]
    templates_buff = templates.flatten()
    print(templates_buff[0:10])
    D  = np.zeros(C * M * T, dtype=np.float32)
    D2 = np.zeros(C * M * T, dtype=np.float32)
    D3 = np.zeros(C * M * T, dtype=np.float32)

    for sampleInd in range(M):
        for templateInd in range(T):
            for chanInd in range(C):
                ind = chanInd + sampleInd * C + templateInd * C * M
                temp = templates_buff[ind]
                D2[chanInd + sampleInd * C + templateInd * C * M] = temp
                D[sampleInd + templateInd * M + chanInd * T * M] = temp
                D3[chanInd + templateInd * C + sampleInd * C * T] = temp

    return D, D2, D3
    

def heatmap(matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')  # 'viridis' is a perceptually-uniform colormap
    plt.colorbar()  # Show color scale
    plt.title(f'Heatmap {matrix.shape}')
    plt.show()
    
def print_row_major(flat_A, num_rows, num_cols):
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            row.append(flat_A[i * num_cols + j])
        print(row)

def conjugate_gradients_gpu(A, y):
    print("================== CONJUGATE GRADIENTS ==================")
    print(f"Condition number = {LA.cond(A)}")
    print(f"Is A nonzero? {np.any(A != 0)}")
    print(f"Norm of y: {np.linalg.norm(y)}")
    y_support = np.where(y != 0, True, False)
    support_intersect_counts = []
    for col in range(A.shape[1]):
        support = np.where(A[:, col] != 0, True, False)
        support_intersect_count = 0
        for i in range(y.shape[0]):
            if support[i] and y_support[i]:
                support_intersect_count += 1
        support_intersect_counts.append(support_intersect_count)
            
    print(f"Size of support overlap between y and each column of A: {support_intersect_counts}")
    A = cp.asarray(A)
    y = cp.asarray(y)
    n = A.shape[1]
    x = cp.zeros(n)
    r = cp.dot(A.T, y)
    d = r.copy()
    delta_new = cp.dot(r, r)
    delta_0 = delta_new.copy()
    print(f"Initial residual L2 squared = {delta_new}")
    i = 0
    while i < 100 and delta_new > 0.0001 * delta_0:
        q = cp.dot(A.T, cp.dot(A, d))
        alpha = delta_new / cp.dot(d, q)
        x += alpha *d
        r -= alpha * q
        delta_old = delta_new.copy()
        delta_new = cp.dot(r, r)
        beta = delta_new / delta_old
        d = r + beta * d
        print(f"Iteration {i}: alpha = {alpha}, beta = {beta}, residual = {delta_new}")
        i += 1

    row_space_residual = y - cp.dot(A, x)
    residual_norm = pow(cp.dot(row_space_residual, row_space_residual), 0.5)
    print(f"Final residual norm = {residual_norm}")
    print(f"Conjugate gradients final x = {x}")
    print("================== CONJUGATE GRADIENTS END ==================")
    return x.get()  # Convert back to NumPy array if needed

def conjugate_gradients(A, y):
    eps = 0.01
    n = A.shape[1]
    x = np.zeros(n)
    r = np.dot(A.T, y)
    d = np.copy(r)
    delta_new = np.dot(r, r)
    print(f"Condition number = {LA.cond(A)}")
    print(f"Initial l2r squared = {delta_new}")
    delta_0 = delta_new
    
    i = 0
    while i < 100 and delta_new > eps * eps * delta_0:
        q = np.dot(A.T, np.dot(A, d))
        alpha = delta_new / np.dot(d, q)
        x = x + alpha * d
        r = r - alpha * q
        delta_old = delta_new
        delta_new = np.dot(r, r)
        beta = delta_new / delta_old
        d = r + beta * d
        print(f"Iteration {i}: alpha = {alpha}, beta = {beta}, residual = {pow(delta_new, 0.5)}")
        i += 1

    row_space_residual = y - np.dot(A, x)
    residual_norm = pow(np.dot(row_space_residual, row_space_residual), 0.5)
    print(f"Row space residual norm = {residual_norm}")
    return x

def convolve(batch, template):
    # Ensure inputs are CuPy arrays for GPU acceleration
    g_batch = cp.asarray(batch)
    g_template = cp.asarray(template)

    # Compute the convolution for each channel and sum across channels
    # We assume 'batch' and 'template' are both formatted correctly (W x C and M x C respectively)
    result = cp.zeros(g_batch.shape[0])

    for channel in range(g_batch.shape[1]):
        # Convolve each channel of the batch with the corresponding channel of the template
        # 'full' mode is used initially to get all possible overlaps
        channel_result = cp.convolve(g_batch[:, channel], g_template[::-1, channel], mode='full')

        # We need to trim the result to match the input batch size
        # Start index for slicing the convolution result to align with the 'same' mode concept
        start_idx = g_template.shape[0] - 1
        result += channel_result[start_idx:start_idx + g_batch.shape[0]]
    
    return result.get()

def find_max_projection(projections):
    # Initialize variables to keep track of the maximum value and its indices
    max_value = -np.inf
    max_template_index = -1
    max_sample_index = -1
    
    # Iterate over each projection to find the maximum value and its indices
    for i, projection in enumerate(projections):
        # Find the index of the maximum value in the current projection
        current_max_value = np.abs(np.max(projection))
        if current_max_value > max_value:
            max_value = current_max_value
            max_template_index = i
            max_sample_index = np.argmax(projection)
    
    return max_template_index, max_sample_index

import numpy as np

def detect_memory_order(flattened_array, rows, cols):
    A_row_major = flattened_array.reshape((rows, cols))
    A_col_major = flattened_array.reshape((cols, rows)).T
    
    row_major_diff_sum = np.sum(np.abs(A_row_major[:, :-1] - A_row_major[:, 1:]))
    col_major_diff_sum = np.sum(np.abs(A_col_major[:, :-1] - A_col_major[:, 1:]))
    
    print(f"Row-major difference sum: {row_major_diff_sum}")
    print(f"Column-major difference sum: {col_major_diff_sum}")
    
    if row_major_diff_sum < col_major_diff_sum:
        print("The array is most likely row-major.")
    else:
        print("The array is most likely column-major.")


def orthogonal_matching_pursuit(A_flattened, templates, y, M, W, C, T, order=None):
    if order == 'F':
        y = np.reshape(y, (W, C), order='F').flatten()

    TAU = 0.1
    THRESH = 0.3
    MAX_ITER = 30
    residual = y.copy()
    #residual = y.astype(np.int32).copy()
    print(f"Check for NaNs or Infs in r: {np.isnan(residual).any()}, {np.isinf(residual).any()}")
    indices = []
    x_approx = np.zeros(T * W)
    rdot_init = np.dot(residual, residual)
    rdot_old = rdot_init
    rdot_new = rdot_init
    print(f"Initial residual norm squared: {rdot_init}")
    for _ in range(MAX_ITER):
        # Project the residual on the columns of A and find the best match
        assert templates[0].shape == (M, C)
        residual_reshaped = np.reshape(residual, (W, C))
        #plot_recording(residual_reshaped)
        projections = [convolve(residual_reshaped, template) for template in templates]
        #heatmap(np.reshape(projections, (T, W)))
        best_template, best_sample = find_max_projection(projections)
        indices.append(best_sample * T + best_template)
        if projections[best_template][best_sample] < THRESH:
            break
        print(f"indices = {indices}")
        print(f"The largest uncertainty of {projections[best_template][best_sample]} was found at template {best_template} and sample {best_sample}, corresponding to index {best_template + best_sample * T}.")

        # Update the approximation and residual
        print(f"indices = {indices}")
        A_selected = extract_submatrix(A_flattened, indices, M, W, C, T)
        print(f"A_selected is non-zero = {np.any(A_selected != 0)}")
        x_sparse = conjugate_gradients_gpu(A_selected, y)
        print(f"x_sparse = {x_sparse}")
        x_approx[indices] = x_sparse
        residual = y - np.dot(A_selected, x_sparse)
        rdot_old = rdot_new
        rdot_new = np.dot(residual, residual)
        print(f"Residual norm squared [Iteration {_}]: {np.dot(residual, residual)}")
        print(f"Indices = {indices}\nx_sparse = {x_sparse}\nx_approx = {x_approx}")
        
        if len(indices) > 1:
            print(f"LHS = {(rdot_init - rdot_old) / (len(indices) - 1) * TAU}")
            print(f"RHS = {rdot_old - rdot_new}")
            
        if len(indices) > 1 and (rdot_init - rdot_old) / (len(indices) - 1) * TAU > (rdot_old - rdot_new):
            break

    return x_approx, indices

def fetch_batch_from_file(W, C, filename):
    with open(filename, 'rb') as fidInput:
        data_batch = np.fromfile(fidInput, dtype=np.int16, offset=10 * C * W, count=C * W)

        if data_batch.size == 0 or data_batch.size != C * W:
            print("Error: read {data_batch.size} entries from {filename}, expected {C * W}")
            sys.exit()

        print(data_batch[0])
        return data_batch

def mean_subtract(arr, start, end, inc):
    sublist = arr[start:end:inc]
    mean_value = sum(sublist) / len(sublist)
    
    for i in range(start, len(arr), inc):
        arr[i] -= mean_value
    
    return arr

def whiten(arr, mat, start, end, inc):
    sublist = arr[start:end:inc]
    whitened = np.dot(mat, sublist)
    arr[start:end:inc] = whitened

def compute_mean_std(arr, start, end, inc):
    sublist = arr[start:end:inc]
    mean = sum(sublist) / len(sublist)
    variance = sum((x - mean) ** 2 for x in sublist) / len(sublist)
    std = math.sqrt(variance)
    return mean, std

def covariance_distance_from_identity(matrix):
    # Compute the covariance matrix
    cov_matrix = np.cov(matrix, rowvar=False)

    # Create the identity matrix of the same size
    identity_matrix = np.eye(cov_matrix.shape[0])

    # Compute the Frobenius norm of the difference between the covariance matrix and the identity matrix
    distance = np.linalg.norm(cov_matrix - identity_matrix, ord='fro')

    # Print the distance
    print(f"Distance from the identity matrix (Frobenius norm): {distance}")
    
# Set random seed for reproducibility
np.random.seed(420)

# Construct the matrix A using the templates
OSS_TRAINING_PATH = Path("C:/SGL_DATA/05_31/oss_training_1/")
templates = np.load(OSS_TRAINING_PATH / "templates.npy")
whitening = np.load(OSS_TRAINING_PATH / "whiteningMat.npy")
print(whitening.shape)
T = templates.shape[0]
M = templates.shape[1]
C = templates.shape[2]
print(f"Read in {T} templates, each defined on {M} samples, of {C} channels")

W = 10 * M

# Templates formatted in the way the CUDA code has it
D_flattened, A_flattened, D3_flattened = construct_A(templates)


# Create a sparse signal
num_spikes = 20
nonzero_indices = np.random.choice(T * W, num_spikes, replace=False)
for index in nonzero_indices:
    print(f"Spike inserted at sample {index // T} with template {index % T}, corresponding to index {index}")
    
print(f"nonzero_indices = {nonzero_indices}")
x_true = np.zeros(T * W)
#x_true_induced = [1 for _ in range(num_spikes)]
x_true_induced = np.random.randn(num_spikes) * 0.1 + 1
x_true[nonzero_indices] = x_true_induced

A_submatrix = extract_submatrix(A_flattened, nonzero_indices, M, W, C, T)
print(x_true.shape)
print(A_submatrix.shape)



# Compute y = Ax + noise
#y = fetch_batch_from_file(W, C, Path("C:\\SGL_DATA\\05_31\\imec_raw") / '240531_g0_t0.imec0.ap.bin')
#mean, std = compute_mean_std(y, 0, len(y), 1)
#print(f"mean = {mean}, std = {std}")
y = np.dot(A_submatrix, x_true_induced) + np.random.normal(0, 0.1, W * C)

# Plot pre-processed data
#plot_recording(np.reshape(y, (W, C), order='F'))
#plt.plot(y, 'r-')
#plt.title(f"Initial y, shape={y.shape}")
#plt.show()
#covariance_distance_from_identity(np.reshape(y, (W, C), order='F'))

# Whiten data
#for samp_index in range(W):
#    whiten(y, whitening, samp_index * C, samp_index * C + C, 1)

#plot_recording(np.reshape(y, (W, C), order='F'))
#plt.plot(y, 'r-')
#plt.title(f"Initial y, shape={y.shape}")
#plt.show()

# Zero-mean
#for chan_index in range(C):
#    mean_subtract(y, chan_index, len(y), C)

# Apply high-pass filter
#for chan_index in range(C):
#    filtered = bandpass(y[chan_index:len(y):C], [100, 400], 30000) # 300hz-high pass filter, recording at 30khz
#    y[chan_index:len(y):C] = filtered
    
#plot_recording(np.reshape(y, (W, C), order='F'))
#plt.plot(y, 'r-')
#plt.title(f"Initial y, shape = {y.shape}")
#plt.show()

# Print the distance of the cov matrix to the identity matrix to verify that whitening works
#covariance_distance_from_identity(np.reshape(y, (W, C), order='F'))

# Plot the recording
#plot_recording(np.reshape(y, (W, C), order='F'))
plt.plot(y)
plt.show()
#plt.plot(y[15:115:1], 'r-')
#plt.title(f'Initial y, shape = {y.shape}')
#plt.show()

# Use OMP to recover the sparse signal
x_omp, support = orthogonal_matching_pursuit(A_flattened, templates, y, M, W, C, T)

# Construct the corresponding y
A_submatrix = extract_submatrix(A_flattened, support, M, W, C, T)
y_omp = np.dot(A_submatrix, x_omp[support])
plot_recording(np.reshape(y_omp, (W, C)))

# Visualize the matrix and vectors
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.stem(x_true, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title('True Sparse x')

plt.subplot(222)
plt.stem(x_omp, linefmt='g-', markerfmt='go', basefmt='r-')
plt.title('Recovered Sparse x')

plt.subplot(223)
plt.plot(y, 'r-')
plt.title('Vector y')

plt.subplot(224)
plt.plot(y_omp, 'r-')
plt.title('Reconstructed y')

plt.tight_layout()
plt.show()

#print("x L2 error:", np.linalg.norm(x_true - x_omp))
#print(f"True spikes:")
#for index in nonzero_indices:
#    print(f"True spike of template {index % T} at sample {index // T}")

for index in support:
    print(f"Spike detected of template {index % T} at sample {index // T}")

print("y L2 error:", np.linalg.norm(y - y_omp))










# CODE GRAVEYARD

# Write A, y to file to test on CUDA
#np.save("C:/Users/Spike Sorter/source/repos/OnlineSpikes_v2/src/Python/cg_learning_Y.npy", y)
#print(f"y.shape = {y.shape}")
#print(f"y = {y}")
#print(f"y.dtype = {y.dtype}")
#y_reshaped = np.reshape(y, (W, C))
#projections = [convolve(y_reshaped, template) for template in templates]
#print(projections[0][0:10])


#plt.plot(y, 'r-')
#plt.title(f'Initial y, shape = {y.shape}')
#plt.show()

#y_linear_combination = np.zeros(W * C)

#for i, index in enumerate(nonzero_indices):
    #template_index = index % T
    #sample_index = index // T
    #atom = np.zeros(W * C)
    #atom[sample_index * C : (sample_index + M) * C] = A_submatrix[sample_index * C : (sample_index + M) * C, i]
    #y_linear_combination += atom * x_true_induced[i]
   # plt.plot(atom, 'r-')
  #  plt.title(f"{i}th atom, template {template_index} at sample {sample_index}")
 #   plt.show()

