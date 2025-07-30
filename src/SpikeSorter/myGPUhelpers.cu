
#include "myGPUhelpers.h"
#include <iostream>
#include "../Helpers/Utils.h"
#include <device_functions.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define BLOCK_SIZE      16
#define DEFAULT_TPB     256
#define MAX_CHANNELS    1024

__global__ void transpose_kernel(const float* d_in, float* d_out,
	int numRows, int numCols)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < numRows && col < numCols) {
		d_out[col * numRows + row] = d_in[row * numCols + col];
	}
}

__global__ void real_to_complex(const float* input, cufftComplex* output, int total) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total) {
		output[idx].x = input[idx];
		output[idx].y = 0.0f;
	}
}

__global__ void complex_to_real(const cufftComplex* input, float* output, int total) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total) {
		output[idx] = input[idx].x;
	}
}

__global__ void mult_conj(cufftComplex* data, const cufftComplex* fwav, int total, int width) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total) {
		int k = idx % width;
		cufftComplex a = data[idx];
		cufftComplex b = fwav[k];
		// Compute the conjugate of b: (b.x, -b.y)
		cufftComplex b_conj;
		b_conj.x = b.x;
		b_conj.y = -b.y;
		// Multiply: a * b_conj = (a.x*b_conj.x - a.y*b_conj.y, a.x*b_conj.y + a.y*b_conj.x)
		cufftComplex prod;
		prod.x = a.x * b_conj.x - a.y * b_conj.y;
		prod.y = a.x * b_conj.y + a.y * b_conj.x;
		data[idx] = prod;
	}
}

__global__ void scale_and_extract_real(const cufftComplex* data, float* output, int total, int width) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total) {
		output[idx] = data[idx].x / width;
	}
}

__global__ void scale_kernel(float* data, int len, float coeff) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
		data[idx] = data[idx] * coeff;
	}
}

__global__ void shift_kernel(float* d_data, float* d_out, int currBatchNumSamples, int C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int len = currBatchNumSamples * C;
	if (idx < len) {
		int freq = idx % currBatchNumSamples;
		int c = idx / currBatchNumSamples;
		int shift = currBatchNumSamples / 2;
		int src_freq = (freq + shift) % currBatchNumSamples;
		d_out[idx] = d_data[src_freq + c * currBatchNumSamples];
	}
}


__global__ void fftshift_kernel(const float* input, float* output, int channels, int width) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = channels * width;
	if (idx < total) {
		int channel = idx / width;
		int pos = idx % width;
		int shift = width / 2;  // floor division works for both even and odd widths
		int newPos = (pos + shift) % width;
		int outIdx = channel * width + newPos;
		output[outIdx] = input[idx];
	}
}

__global__ void compute_channel_means(const float* d_batch, float* d_means, int W, int C) {
	int channel = blockIdx.x * blockDim.x + threadIdx.x;
	if (channel < C) {
		d_means[channel] = 0.0;
		float sum = 0;
		for (int samp = 0; samp < W; samp++) {
			d_means[channel] += d_batch[samp * C + channel];
		}
		d_means[channel] /= (float)W;
	}
}

__global__ void subtract_channel_mean(float* d_batch, const float* d_means, int W, int C) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = W * C;

	if (idx < total) {
		int channel = idx % C;
		d_batch[idx] -= d_means[channel];
	}
}

__global__ void subtractSpikeContributions(
	const long *d_spikeIndices, // spike indices array (length: numSpikes)
	const float *d_amps,        // amplitudes (length: numSpikes)
	const float *d_templateWaveforms, // [numTemplates x M x C]
	long M, long C,             // Template length and number of channels
	long currBatchNumSamples,  // Number of samples in the current batch
	const float *d_ctc, // [numTemplates x unclu_T x (2*M+1)]
	long unclu_T,              // Number of templates (or convolution “time bins”)
	int numSpikes,            // Number of spikes in d_spikeIndices
	float *d_residual,        // [currBatchNumSamples x C]
	float *d_convResult)      // [unclu_T x currBatchNumSamples]
{
	// Each spike produces two sets of contributions:
	//   (a) the residual: M * C values (from the template waveform)
	//   (b) the convolution: unclu_T * (2*M+1) values (from d_ctc_permuted)
	const int resSize = M * C;
	const int convSize = unclu_T * (2 * M + 1);
	const int totalPerSpike = resSize + convSize;

	// Each thread handles one contribution element for one spike.
	int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int totalThreads = numSpikes * totalPerSpike;
	if (globalIdx >= totalThreads) return;

	// Determine which spike and which element (local index) this thread processes.
	int spikeId = globalIdx / totalPerSpike;
	int localIdx = globalIdx % totalPerSpike;

	// Decode the spike's index: 
	//   - temp: the template index
	//   - sample: the sample location.
	long spikeVal = d_spikeIndices[spikeId];
	int sample = spikeVal % currBatchNumSamples;
	int temp = spikeVal / currBatchNumSamples;

	// Load the amplitude for this spike.
	float amp = d_amps[spikeId];

	// Update residual
	if (localIdx < resSize) {
		// Compute (m, c) coordinates into the MxC template.
		int m = localIdx / C;
		int c = localIdx % C;
		int sampleLeft_res = sample - (M / 2);
		int targetSample = sampleLeft_res + m;
		// Check that the target index is in bounds.
		if (targetSample >= 0 && targetSample < currBatchNumSamples) {
			int residIndex = targetSample * C + c;
			// Fetch the corresponding template value.
			float val = d_templateWaveforms[temp * (M * C) + m * C + c];
			// Subtract the (scaled) template value from the residual.
			atomicAdd(&(d_residual[residIndex]), -amp * val);
		}
	}
	//  Update convolution result
	else {
		int r2 = localIdx - resSize;
		int t = r2 / (2 * M + 1);  // which "row" in the conv submatrix (0<=t<unclu_T)
		int k = r2 % (2 * M + 1);  // which "column" (0 <= k < 2*M+1)
		int sampleLeft_conv = sample - M;
		int targetSample = sampleLeft_conv + k;
		if (targetSample >= 0 && targetSample < currBatchNumSamples) {
			int convIndex = t * currBatchNumSamples + targetSample;
			float val = d_ctc[t * (unclu_T * (2 * M + 1)) + temp * (2 * M + 1) + k];
			atomicAdd(&(d_convResult[convIndex]), -amp * val);
		}
	}
}

__global__ void subtractSpikeContributions_convOnly(
	const long *d_spikeIndices, // spike indices array (length: numSpikes)
	const float *d_amps,        // amplitudes (length: numSpikes)
	long M,                   // Template length (used to determine offset)
	long currBatchNumSamples, // Number of samples in the current batch
	const float *d_ctc,       // [numTemplates x unclu_T x (2*M+1)]
	long unclu_T,             // Number of convolution time bins
	int numSpikes,            // Number of spikes in d_spikeIndices
	float *d_convResult)      // [unclu_T x currBatchNumSamples]
{
	// Each spike now produces convSize contributions.
	const int convSize = unclu_T * (2 * M + 1);
	int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int totalThreads = numSpikes * convSize;
	if (globalIdx >= totalThreads) return;

	// Determine which spike and which element this thread processes.
	int spikeId = globalIdx / convSize;
	int localIdx = globalIdx % convSize;

	// Decode spike information.
	long spikeVal = d_spikeIndices[spikeId];
	int spikeTime = spikeVal % currBatchNumSamples;
	int spikeTemplate = spikeVal / currBatchNumSamples;
	float amp = d_amps[spikeId];

	// Compute indices within the convolution submatrix.
	int t = localIdx / (2 * M + 1);  // convolution "row" (0 <= t < unclu_T)
	int k = localIdx % (2 * M + 1);    // convolution "column" (0 <= k < 2*M+1)

	// Align the submatrix so that its column 0 goes to sample - M.
	int sampleLeft_conv = spikeTime - M;
	int targetSample = sampleLeft_conv + k;

	if (targetSample >= 0 && targetSample < currBatchNumSamples) {
		int convIndex = t * currBatchNumSamples + targetSample;
		float val = d_ctc[t * (unclu_T * (2 * M + 1)) + spikeTemplate * (2 * M + 1) + k];

		atomicAdd(&(d_convResult[convIndex]), -amp * val);
	}
}

__global__ void multiplyByConjugateKernel(cufftComplex* d_batchFreq,
	const cufftComplex* d_filterFreq,
	int numFreq,
	int numChannels)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = numFreq * numChannels;
	if (idx < total) {
		// Compute frequency bin index k and channel index c.
		int c = idx / numFreq;
		int k = idx % numFreq;

		// Read the batch frequency-domain value.
		cufftComplex X = d_batchFreq[idx];
		// Read the filter frequency-domain value at bin k.
		cufftComplex F = d_filterFreq[k];

		// Take the complex conjugate of the filter: (a, b) -> (a, -b)
		F.y = -F.y;

		// Compute the product: (X * conj(F))
		cufftComplex prod;
		prod.x = X.x * F.x - X.y * F.y;
		prod.y = X.x * F.y + X.y * F.x;

		// Write the result back.
		d_batchFreq[idx] = prod;
	}

}

__global__ void cross_correlation_kernel(const float* __restrict__ Wall3,
	const float* __restrict__ B_in,
	float* __restrict__ B_out,
	int T, int K, int C, int W)
{
	// Our inner dimension: sum_{k,c} becomes p from 0 to P-1.
	const int P = K * C;

	// Each block computes a BLOCK_SIZE x BLOCK_SIZE tile of the output.
	// Compute the global row (t) and column (w) indices.
	int t = blockIdx.x * BLOCK_SIZE + threadIdx.y;  // row index in output (T)
	int w = blockIdx.y * BLOCK_SIZE + threadIdx.x;    // column index in output (W)

	float sum = 0.0f;

	// Declare shared memory tiles for A (from Wall3) and B (from B_in).
	// A_tile holds a BLOCK_SIZE (output tile rows) x BLOCK_SIZE (inner tile) chunk.
	// B_tile holds a BLOCK_SIZE (inner tile) x BLOCK_SIZE (output tile cols) chunk.
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	// Loop over the inner dimension in chunks of BLOCK_SIZE.
	// Each iteration loads one tile of the inner dimension.
	for (int tile_idx = 0; tile_idx < (P + BLOCK_SIZE - 1) / BLOCK_SIZE; tile_idx++) {

		// -----------------------------
		// Load one tile from Wall3 into shared memory.
		// For our GEMM, we want to load A[t, p] = Wall3[t, k, c]
		// where p = tile_idx * BLOCK_SIZE + threadIdx.x,
		// and k = p / C, c = p % C.
		int pA = tile_idx * BLOCK_SIZE + threadIdx.x;
		if (t < T && pA < P) {
			int k_val = pA / C;    // integer division
			int c_val = pA % C;
			// Note: Wall3 is stored with layout: t*(K*C) + k*C + c.
			As[threadIdx.y][threadIdx.x] = Wall3[t * (K * C) + k_val * C + c_val];
		}
		else {
			As[threadIdx.y][threadIdx.x] = 0.0f;
		}

		// -----------------------------
		// Load one tile from B_in into shared memory.
		// We define B[p, w] = B_in[c, k, w] where p = tile_idx * BLOCK_SIZE + threadIdx.y,
		// with k = p / C and c = p % C.
		int pB = tile_idx * BLOCK_SIZE + threadIdx.y;
		if (pB < P && w < W) {
			int k_val = pB / C;
			int c_val = pB % C;
			// B_in is stored as: c*(K*W) + k*W + w.
			Bs[threadIdx.y][threadIdx.x] = B_in[c_val * (K * W) + k_val * W + w];
		}
		else {
			Bs[threadIdx.y][threadIdx.x] = 0.0f;
		}

		// Make sure the tiles are fully loaded before computing.
		__syncthreads();

		// -----------------------------
		// Compute partial sum over this tile.
#pragma unroll
		for (int i = 0; i < BLOCK_SIZE; i++) {
			sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
		}

		// Synchronize before loading the next tile.
		__syncthreads();
	}

	// Write the computed output if within bounds.
	if (t < T && w < W) {
		B_out[t * W + w] = sum;
	}
}

__global__ void normalize_conv_kernel(const float* __restrict__ d_TW, const float* __restrict__ d_nm, float* __restrict__ d_Cf, int T, int W, int M) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = T * W;
	if (idx < total) {
		int temp = idx / W;
		int samp = idx % W;
		float relu = fmaxf(d_TW[idx], 0.0f);

		if (samp < M || samp >= W - M) {
			d_Cf[idx] = 0.0;
		}
		else {
			d_Cf[idx] = (relu * relu) * d_nm[temp];
		}
	}
}

__global__ void conv1d(
	const float* __restrict__ d_batch,      // [C, currBatchNumSamples]
	const float* __restrict__ d_wPCA,       // [K, M]
	float*       __restrict__ d_batchPCA,   // [C, K, currBatchNumSamples]
	int C,
	int K,
	int M,
	int currBatchNumSamples
)
{
	// Each thread will compute exactly one output element: B[n, k, w]
	//  totalOutputs = C*K*currBatchNumSamples
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int totalOutputs = C * K * currBatchNumSamples;
	if (globalIndex >= totalOutputs) return;

	// Decode (n, k, w) from the linear index
	int chanIndex = globalIndex / (K * currBatchNumSamples);                 // which "batch" (C dimension)
	int r = globalIndex % (K * currBatchNumSamples);
	int pcaIndex = r / currBatchNumSamples;                                 // which output filter (K dimension)
	int sampIndex = r % currBatchNumSamples;                                 // which spatial position

	int halfM = M / 2;

	float val = 0.0f;

	// Sum over the kernel width M
	for (int sampOffset = 0; sampOffset < M; sampOffset++) {
		// Compute the corresponding input index with "same" padding
		int w_in = sampIndex + (sampOffset - halfM);

		if (w_in >= 0 && w_in < currBatchNumSamples) {
			//   batch[n, w_in] in memory => index = n*currBatchNumSamples + w_in
			float batch_val = d_batch[chanIndex * currBatchNumSamples + w_in];
			//   wPCA[k, kw] in memory => index = k*M + kw
			float filter_val = d_wPCA[pcaIndex * M + sampOffset];
			val += batch_val * filter_val;
		}
	}

	// Store to B[n, k, w]
	// B is [C, K, currBatchNumSamples]
	// linear index in B => n*(K*currBatchNumSamples) + k*currBatchNumSamples + w
	d_batchPCA[globalIndex] = val;
}

__global__ void reduce_to_time_dim_by_max_kernel(const float* __restrict__ d_Cf, float* __restrict__ d_Cfmax, long* __restrict__ d_imax, int T, int W) {
	int sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (sampleIndex < W) {
		float max_val = -FLT_MAX;
		int max_idx = -1;
		for (int templateIndex = 0; templateIndex < T; ++templateIndex) {
			float val = d_Cf[templateIndex * W + sampleIndex];
			if (val > max_val) {
				max_val = val;
				max_idx = templateIndex;
			}
		}
		d_Cfmax[sampleIndex] = max_val;
		d_imax[sampleIndex] = max_idx;
	}
}

#define UNROLL 4
__global__ void find_matching_indices_kernel(const float* __restrict__ d_Cfmaxpool,
	const float* __restrict__ d_Cf,
	float Th,
	int T, int W,
	long* __restrict__ out,
	int N,
	int* __restrict__ globalCount)
{
	// Optionally load d_Cfmaxpool into shared memory if it fits.
	extern __shared__ float s_Cfmaxpool[];
	if (threadIdx.x < W)
	{
		s_Cfmaxpool[threadIdx.x] = d_Cfmaxpool[threadIdx.x];
	}
	__syncthreads();

	// Compute thread’s global index and stride.
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	// Unrolled grid-stride loop.
	for (int idx = tid; idx < N; idx += stride * UNROLL)
	{
#pragma unroll
		for (int u = 0; u < UNROLL; u++)
		{
			int i = idx + u * stride;
			if (i < N)
			{
				// Compute the corresponding sample index.
				int samp = i % W;

				// Prefetch d_Cf[i] into a register.
				float cf_val = d_Cf[i];

				// Get the corresponding d_Cfmaxpool value.
				// If W is small (fits in shared memory), use shared memory.
				// Otherwise, use __ldg to prefetch via the read-only cache.
				float cfmax_val = (W <= blockDim.x) ? s_Cfmaxpool[samp] : __ldg(&d_Cfmaxpool[samp]);

				// Check the predicate: compare (nearly equal) and ensure cfmax_val >= Th.
				if (fabsf(cfmax_val - cf_val) < 0.0001f && cfmax_val >= Th)
				{
					// Write out the index atomically.
					int pos = atomicAdd(globalCount, 1);
					out[pos] = i;
				}
			}
		}
	}
}

extern "C"
__global__ void median_remove_kernel(float* __restrict__ data, int C, int W)
{
	// Each block handles one sample index t.
	int t = blockIdx.x;
	if (t >= W)
		return;

	if (C > MAX_CHANNELS)
		return;

	__shared__ float s_orig[MAX_CHANNELS];  // Only indices [0, C-1] hold valid data.
	__shared__ float s_sort[MAX_CHANNELS];  // Entire array is used for bitonic sort.

	int tid = threadIdx.x;
	if (tid < MAX_CHANNELS) {
		float x = (tid < C) ? data[tid + t * C] : CUDART_INF_F;
		if (tid < C)
			s_orig[tid] = x;
		s_sort[tid] = x;
	}
	__syncthreads();

	// bitonic sort on fixed number of channels, padding empty entries with infinity
	for (int k = 2; k <= MAX_CHANNELS; k *= 2) {
#pragma unroll
		for (int j = k / 2; j > 0; j /= 2) {
			if (tid < MAX_CHANNELS) {
				// Compute partner index for tid.
				int ixj = tid ^ j;
				if (ixj > tid) {
					// Determine the sort direction for this k-segment.
					bool ascending = ((tid & k) == 0);
					// Swap to enforce the bitonic order.
					float a = s_sort[tid];
					float b = s_sort[ixj];
					if ((ascending && a > b) || (!ascending && a < b)) {
						s_sort[tid] = b;
						s_sort[ixj] = a;
					}
				}
			}
			__syncthreads();
		}
	}
	// s_sort[] is now sorted in ascending order.

	// Compute the median from sorted data
	float median = 0.0f;
	if (tid == 0) {
		if (C & 1) median = s_sort[C / 2];
		else median = s_sort[(C - 1) / 2]; // aligns with pytorch.median, which is what kilosort uses

		s_sort[0] = median;
	}
	__syncthreads();
	median = s_sort[0];

	// Each thread subtracts the computed median from its channel's original value.
	if (tid < C) {
		float result = s_orig[tid] - median;
		data[tid + t * C] = result;
	}
}

__global__ void reciprocal_kernel(float* d_A, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
		d_A[idx] = 1 / d_A[idx];
	}
}

void scale(float* d_data, const float coeff, const int len)
{
	static const char* ptLabel = { "myGPUHelpers::scale()" };
	int threadsPerBlock = 256;
	int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
	scale_kernel <<<blocksPerGrid, threadsPerBlock >>> (d_data, len, coeff);
	_CUDA_CALL(cudaDeviceSynchronize());
}

void shift(float* d_data, float* d_out, int currBatchNumSamples, int C)
{
	static const char* ptLabel = { "myGPUHelpers::shift" };
	int threadsPerBlock = 256;
	int blocksPerGrid = (currBatchNumSamples * C + threadsPerBlock - 1) / threadsPerBlock;
	shift_kernel <<<blocksPerGrid, threadsPerBlock >> > (d_data, d_out, currBatchNumSamples, C);
	_CUDA_CALL(cudaDeviceSynchronize());
}

long RemoveDCGPU(cublasHandle_t& Handle, float *fY, float *fDC, long lW, long lC, long lCt, float *OnesArray, float *Means) {
	static const char *ptLabel = { "RemoveDCGPU" };

	//Recalculate weight
	float fWeight = fmin( ( (float) lCt) / ( (float) (lCt + lW) ), .99f);

	//Alpha to scale the sums
	float alpha = 1.f / lW;
	float alpha2 = (1.f - fWeight);

	//Zero and One constants
	static const float ZeroBeta  = 0.f;
	static const float OneAlpha = 1.f;
	static const float NegativeAlpha = -1.f;

	//Scale fDC with the recalculated weight
	cublasSscal(Handle, lC, &fWeight, fDC, 1);

	_CUDA_CALL(cudaDeviceSynchronize());

	//Calculate mean along rows
	if (cublasSgemv(Handle, CUBLAS_OP_N, lC, lW, &alpha, fY, lC, OnesArray, 1, &ZeroBeta, Means, 1) != CUBLAS_STATUS_SUCCESS) {
		_RUN_ERROR(ptLabel, "Error in Calculing the mean along the rows.");
	}

	_CUDA_CALL(cudaDeviceSynchronize());
	
	//Add to 
	if (cublasSaxpy(Handle, lC, &alpha2, Means, 1, fDC, 1) != CUBLAS_STATUS_SUCCESS) {
		_RUN_ERROR(ptLabel, "Error in scaling the means");
	}
	
	_CUDA_CALL(cudaDeviceSynchronize());

	for (int ii = 0; ii < lC; ii++) {
		if (cublasSaxpy(Handle, lW, &NegativeAlpha, fDC + ii, 0, fY + ii, lC) != CUBLAS_STATUS_SUCCESS) {
			_RUN_ERROR(ptLabel, "Error in substracting the DC component from the data");
		}
	}

	_CUDA_CALL(cudaDeviceSynchronize());

	return (lCt + lW);
}

void DriftCorrectOnGPU(cublasHandle_t& Handle, float *matA, float *matB, float *matC, long lW, long lC) {
	static const char *ptLabel = { "DriftCorrectOnGPU" };

	static const float alpha = 1.f;
	static const float beta = 0.f;

	// Computes C = alpha * op(A) * op(B) + beta * C
	if (cublasSgemm(Handle,
		CUBLAS_OP_T, // transa
		CUBLAS_OP_N, // transb
		lC, lW, lC, // m, n, k
		&alpha, // alpha
		matA, lC, // A, lda
		matB, lC, // B, ldb
		&beta, // beta 
		matC, lC // C, ldc
	) != CUBLAS_STATUS_SUCCESS) {
		_RUN_ERROR(ptLabel, "Error in drift correcting the data");
	}

}

void ComputeDriftMat(int y_shift, float* d_result)
{
	static const char *ptLabel = { "ComputeDriftMat" };
}

void WhitenOnGPU(cublasHandle_t& Handle, float *matA, float *matB, float *matC, long lW, long lC) {
	static const char *ptLabel = { "WhitenOnGPU" };

	static const float alpha = 1.f;
	static const float beta =  0.f;
	
	// Computes C = alpha * op(A) * op(B) + beta * C
	if (cublasSgemm(Handle, 
		CUBLAS_OP_N, // transa
		CUBLAS_OP_N, // transb
		lC, lW, lC, // m, n, k
		&alpha, // alpha
		matA , lC, // A, lda
		matB, lC, // B, ldb
		&beta, // beta 
		matC, lC // C, ldc
	) != CUBLAS_STATUS_SUCCESS) {
		_RUN_ERROR(ptLabel, "Error in whitening the data");
	}
}

// assumes d_A is row-major
void transpose(float* d_A, float* d_At, int numRows, int numCols) {
	dim3 block(16, 16);
	dim3 grid((numCols + block.x - 1) / block.x,
		(numRows + block.y - 1) / block.y);

	transpose_kernel <<<grid, block >>> (d_A, d_At, numRows, numCols);
}

void highpass(cufftHandle planForward, cufftHandle planInverse, float* d_batch, float* d_highpassed, const std::complex<float>* fwav, cufftComplex* d_hpworkspace, long C, long W) 
{
	static const char* ptLabel = { "myGPUHelpers::highpass()" };
	long total = C * W;
	const int block_size = 256;


	// Execute the forward FFT (in-place).
	cufftExecR2C(planForward, d_batch, d_hpworkspace);
	_CUDA_CALL(cudaDeviceSynchronize());

	// Multiply every frequency component by the conjugate of fwav.
	// We reinterpret fwav as a cufftComplex pointer.
	int gridSize = (total + block_size - 1) / block_size;

	const cufftComplex* d_fwav = reinterpret_cast<const cufftComplex*>(fwav);
	mult_conj <<<gridSize, block_size>>> (d_hpworkspace, d_fwav, total, W);
	_CUDA_CALL(cudaDeviceSynchronize());

	// Execute the inverse FFT (in-place).
	cufftExecC2R(planInverse, d_hpworkspace, d_batch);
	_CUDA_CALL(cudaDeviceSynchronize());

	// Scale the inverse FFT result and extract its real part back into batch.
	scale_and_extract_real <<<gridSize, block_size>>> (d_hpworkspace, d_batch, total, W);
	_CUDA_CALL(cudaDeviceSynchronize());

	fftshift_kernel <<<gridSize, block_size>>> (d_batch, d_highpassed, C, W);
	_CUDA_CALL(cudaDeviceSynchronize());
}

void meanRemove(float* d_batch, float* d_workspace, long W, long C)
{
	const int threadsPerBlock = 256;
	int blocksPerGrid = C;
	size_t sharedMemSize = threadsPerBlock * sizeof(float);
	compute_channel_means << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (d_batch, d_workspace, W, C);
	blocksPerGrid = (W * C + threadsPerBlock - 1) / threadsPerBlock;
	subtract_channel_mean << <blocksPerGrid, threadsPerBlock >> > (d_batch, d_workspace, W, C);
}

/*
 * Assumptions:
 *  - A is MxK (row-major) on device memory, pointed to by d_A
 *  - B is KxN (row-major) on device memory, pointed to by d_B
 *  - C is MxN (row-major) on device memory, pointed to by d_C
 *  - We want: C = A * B  (in row-major sense)
 *
 */
void matMul(cublasHandle_t& handle, const float* d_A, const float* d_B, float* d_C, int M, int K, int N) {
	float alpha = 1.0f, beta = 0.0f;
	cublasSgemm(handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		N,
		M,
		K,
		&alpha,
		d_B,
		N,
		d_A,
		K,
		&beta,
		d_C,
		N);
}

void updateResidual(
	const thrust::device_vector<long>& d_spikeIndices,  // spike indices vector
	float* d_amps,               // amplitudes array on device
	float* d_templateWaveforms,  // template waveforms on device
	int M,                     // number of samples in a template
	int C,                     // number of channels
	int currBatchNumSamples,   // number of samples in current batch
	float* d_ctc,              // ctc matrix on device
	int unclu_T,               // dimension for convolution output
	float* d_residual,         // residual array on device
	float* d_convResult
) 
{
	static const char* ptLabel = { "myGPUhelpers::updateResidual()" };

	// Calculate sizes for residual and convolution contributions per spike.
	int resSize = M * C;
	int convSize = unclu_T * (2 * M + 1);
	int totalPerSpike = resSize + convSize;

	// Total threads to cover all contributions of all spikes.
	int totalThreads = static_cast<int>(d_spikeIndices.size()) * totalPerSpike;

	// Define threads per block.
	int threadsPerBlock = 256;

	// Compute the required number of blocks.
	int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

	// Launch the kernel.
	subtractSpikeContributions << <blocksPerGrid, threadsPerBlock >> > (
		thrust::raw_pointer_cast(d_spikeIndices.data()),
		d_amps,
		d_templateWaveforms,
		M,
		C,
		currBatchNumSamples,
		d_ctc,
		unclu_T,
		static_cast<int>(d_spikeIndices.size()),
		d_residual,
		d_convResult
		);

	/*
	threadsPerBlock = 256;
	totalThreads = static_cast<int>(d_spikeIndices.size()) * convSize;
	blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
	subtractSpikeContributions_convOnly << <blocksPerGrid, threadsPerBlock >> > (
		thrust::raw_pointer_cast(d_spikeIndices.data()), // spike indices array (length: numSpikes)
		d_amps,        // amplitudes (length: numSpikes)
		M,                   // Template length (used to determine offset)
		currBatchNumSamples, // Number of samples in the current batch
		d_ctc,       // [numTemplates x unclu_T x (2*M+1)]
		unclu_T,             // Number of convolution time bins
		static_cast<int>(d_spikeIndices.size()),            // Number of spikes in d_spikeIndices
		d_convResult);      // [unclu_T x currBatchNumSamples]*/

	_CUDA_CALL(cudaDeviceSynchronize());
}

void applyFilter(cufftComplex* d_batch, cufftComplex* d_filter, long currBatchNumSamples, long C)
{
	static const char* ptLabel = { "myGPUHelpers::applyFilter()" };
	int threadsPerBlock = 256;
	//int numFreq = currBatchNumSamples / 2 + 1;
	int numFreq = currBatchNumSamples;
	int blocksPerGrid = (numFreq * C + threadsPerBlock - 1) / threadsPerBlock;
	multiplyByConjugateKernel <<<blocksPerGrid, threadsPerBlock >>> (d_batch, d_filter, numFreq, C);
	_CUDA_CALL(cudaDeviceSynchronize());
}

void float_to_cufftComplex(float* d_in, cufftComplex* d_out, int len) {
	static const char* ptLabel = { "myGPUHelpers::float_to_cufftComplex()" };

	int threadsPerBlock = 256;
	int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
	real_to_complex << <blocksPerGrid, threadsPerBlock >> > (d_in, d_out, len);
	_CUDA_CALL(cudaDeviceSynchronize());
}

void cufftComplex_to_float(cufftComplex* d_in, float* d_out, int len) {
	static const char* ptLabel = { "myGPUHelpers::cufftComplex_to_float()" };

	int threadsPerBlock = 256;
	int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
	complex_to_real << <blocksPerGrid, threadsPerBlock >> > (d_in, d_out, len);
	_CUDA_CALL(cudaDeviceSynchronize());
}

void crossCorrelation(float* d_A, float* d_B, float* d_result, int unclu_T, int K, int C, int currBatchNumSamples) {
	static const char* ptLabel = { "myGPUHelpers::crossCorrelation()" };
	auto block = dim3(16, 16);
	auto grid = dim3((unclu_T + block.x - 1) / block.x,
		(currBatchNumSamples + block.y - 1) / block.y);

	cross_correlation_kernel <<<grid, block >>> (d_A, d_B, d_result, unclu_T, K, C, currBatchNumSamples);
}

void normalizeConv(float* d_convResult, float* d_nm, float* d_result, int unclu_T, int currBatchNumSamples, int M) {
	static const char* ptLabel = { "myGPUHelpers::normalizeConv()" };
	int blocksPerGrid = (unclu_T * currBatchNumSamples + DEFAULT_TPB - 1) / DEFAULT_TPB;
	normalize_conv_kernel << <blocksPerGrid, DEFAULT_TPB >> > (d_convResult, d_nm, d_result, unclu_T, currBatchNumSamples, M);
}

void projectToPCA(
	const float* d_batch,         // [C, currBatchNumSamples]
	const float* d_wPCA_permuted, // [K, M]
	float*       d_batchPCA,      // [C, K, currBatchNumSamples]
	int          K,
	int          M,
	int          C,
	int          currBatchNumSamples
)
{
	// Number of total output elements = C*K*currBatchNumSamples
	int totalOutputs = C * K * currBatchNumSamples;

	// Launch configuration
	int blockSize = DEFAULT_TPB;
	int gridSize = (totalOutputs + blockSize - 1) / blockSize;

	conv1d << <gridSize, blockSize >> > (
		d_batch,
		d_wPCA_permuted,
		d_batchPCA,
		C,
		K,
		M,
		currBatchNumSamples
		);
}

void reduceToTimeDimByMax(float* d_convNormalized, float* d_maxAtTime, long* d_imax, int unclu_T, int currBatchNumSamples)
{
	auto blocksPerGrid = (currBatchNumSamples + DEFAULT_TPB - 1) / DEFAULT_TPB;
	reduce_to_time_dim_by_max_kernel << <blocksPerGrid, DEFAULT_TPB >> > (d_convNormalized, d_maxAtTime, d_imax, unclu_T, currBatchNumSamples);
}

void findMatchingIndices(const float* d_Cfmaxpool,
	const float* d_Cf,
	float Th,
	long T, int currBatchNumSamples, long M,
	thrust::device_vector<long>& matching_indices, int* d_count)
{
	static const char* ptLabel = { "OnlineSpikesV2::find_matching_indices_custom" };

	// Total number of elements in d_Cf.
	int N = T * currBatchNumSamples;

	// Allocate (or resize) the output vector to hold up to N indices.
	matching_indices.resize(currBatchNumSamples / M);

	// Allocate a device counter and initialize it to zero.
	cudaMemset(d_count, 0, sizeof(int));

	// We want enough blocks so that each thread processes UNROLL elements per iteration.
	int blocks = (N + DEFAULT_TPB * UNROLL - 1) / (DEFAULT_TPB * UNROLL);

	// If W is small enough, load all d_Cfmaxpool into shared memory.
	// (Otherwise, we set shared memory size to 0.)
	int sharedMemSize = (currBatchNumSamples <= DEFAULT_TPB) ? currBatchNumSamples * sizeof(float) : 0;

	// Get the raw pointer to the output data.
	long* d_out = thrust::raw_pointer_cast(matching_indices.data());

	// Launch the kernel.
	find_matching_indices_kernel << <blocks, DEFAULT_TPB, sharedMemSize >> > (d_Cfmaxpool,
		d_Cf,
		Th,
		T, currBatchNumSamples,
		d_out,
		N,
		d_count);
	_CUDA_CALL(cudaDeviceSynchronize()); // (For debugging; remove in production)

	// Copy the global counter back to host.
	int h_count = 0;
	cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

	// Resize the matching_indices vector to the number of matching indices found.
	matching_indices.resize(h_count);
}


void medianRemove(float* data, int C, int currBatchNumSamples) {
	static const char* ptLabel = { "OnlineSpikesV2::removeMedianHost" };

	int blockSize = MAX_CHANNELS; // we use sorting algorithm that operates on fixed number of entries
	int numBlocks = currBatchNumSamples;
	median_remove_kernel << <numBlocks, blockSize >> > (data, C, currBatchNumSamples);
	_CUDA_CALL(cudaDeviceSynchronize());
}

void reciprocal(float* d_nm, long unclu_T) {
	static const char* ptLabel = { "myGPUHelpers::reciprocal" };
	reciprocal_kernel <<< (unclu_T + DEFAULT_TPB - 1) / DEFAULT_TPB, DEFAULT_TPB >> > (d_nm, unclu_T);
}

myDCRemover::myDCRemover(){


}


myDCRemover::~myDCRemover() {
	static const char *ptLabel = { "myDCRemover::~myDCRemover" };

	_CUDA_CALL(cudaFree(m_fMeans));
	_CUDA_CALL(cudaFree(m_fOnes));

}

void myDCRemover::InitArrays(const long lC, const long lN) {
	static const char *ptLabel = { "RemyDCRemover::InitArrays" };

	_CUDA_CALL(cudaMalloc((void**)&m_fOnes, lN * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_fMeans, lC * sizeof(float)));
	_CUDA_CALL(cudaMemset(m_fMeans, 0, lC * sizeof(float)));
	//SetVal << <1, 1 >> > (m_fOnes, lN, 1.f);

	float *temp = (float*)malloc(lN * sizeof(float));
	for (int ii = 0; ii < lN; ii++)
		temp[ii] = 1.f;


	_CUDA_CALL(cudaMemcpy(m_fOnes, temp, lN * sizeof(float), cudaMemcpyHostToDevice));
	free(temp);
}

long myDCRemover::RemoveDC(cublasHandle_t& Handle, float *fY, float *fDC, long lW, long lC, long lCt) {
	static const char *ptLabel = { "myDCRemover::RemoveDC" };

	long out = RemoveDCGPU(Handle, fY, fDC, lW, lC, lCt, m_fOnes, m_fMeans);

	return out;
}

