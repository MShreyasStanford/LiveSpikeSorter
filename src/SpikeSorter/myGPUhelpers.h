#pragma once

#include "device_launch_parameters.h"
#include <math.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <complex>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

void DriftCorrectOnGPU(cublasHandle_t& Handle, float *fW, float *fYW, float *fY, long lW, long lC);
void WhitenOnGPU(cublasHandle_t& Handle, float *fW, float *fYW, float *fY, long lW, long lC);
void matMul(cublasHandle_t& handle, const float* d_A, const float* d_B, float* d_C, int M, int K, int N);
long RemoveDCGPU(cublasHandle_t& Handle, float *fY, float *fDC, long lW, long lC, long lCt, float *OnesArray, float *Means);
void transpose(float* d_A, float* d_At, int numRows, int numCols);
void highpass(cufftHandle planForward, cufftHandle planInverse, float* d_batch, float* d_highpassed, const std::complex<float>* fwav, cufftComplex* d_hpworkspace, long C, long W);
void meanRemove(float* d_batch, float* d_workspace, long W, long C);
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
);

void applyFilter(cufftComplex* d_batch, cufftComplex* d_filter, long currBatchNumSamples, long C);
void scale(float* d_data, const float coeff, const int len);
void shift(float* d_data, float* d_out, int currBatchNumSamples, int C);
void float_to_cufftComplex(float* d_in, cufftComplex* d_out, int len);
void cufftComplex_to_float(cufftComplex* d_in, float* d_out, int len);
void crossCorrelation(float* d_A, float* d_B, float* d_result, int unclu_T, int K, int C, int currBatchNumSamples);
void normalizeConv(float* d_convResult, float* d_nm, float* d_result, int unclu_T, int currBatchNumSamples, int M);
void projectToPCA(const float* d_batch, const float* d_wPCA_permuted, float* d_batchPCA, int K, int M, int C, int currBatchNumSamples);
void reduceToTimeDimByMax(float* d_convNormalized, float* d_maxAtTime, long* d_imax, int unclu_T, int currBatchNumSamples);
void findMatchingIndices(const float* d_Cfmaxpool,
	const float* d_Cf,
	float Th,
	long T, int currBatchNumSamples, long M,
	thrust::device_vector<long>& matching_indices, int* d_count);
void medianRemove(float* data, int C, int currBatchNumSamples);
void reciprocal(float* d_nm, long unclu_T);

class myDCRemover {
public:
	myDCRemover();
	~myDCRemover();

	void InitArrays(const long lC, const long lN);
	long RemoveDC(cublasHandle_t& Handle, float *fY, float *fDC, long lW, long lC, long lCt);

protected:
	float *m_fOnes;
	float *m_fMeans;
};