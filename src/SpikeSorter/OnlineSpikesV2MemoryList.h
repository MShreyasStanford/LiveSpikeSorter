#pragma once

/*
	Hacky X-macros to help make life much easier, but obfuscates compilation errors to near unparsability.
	Defining a new entry here will automatically populate class members in OnlineSpikesV2.h, populate code inOnlineSpikesV2::allocateMemory(),
	while also populating OnlineSpikesV2::~OnlineSpikesV2(). 
	
	If the amount of memory you want is dependent on a variable `foo`, one must have `foo` be defined as a member 
	of OnlineSpikesV2 prior to the allocateMemory() call performed in the constructor.

	TODO: Remove some now-unused entries from here.
*/

enum MemoryType {
	Host,		// memory that will be allocated on the heap as an array with: name = new type[size];
	Device,		// memory that will be allocated on the device (GPU) with: _CUDA_CALL(cudaMalloc((void**)&name, sizeof(type) * size));
	Pinned		// memory that will be allocated to pinned memory on the host with: _CUDA_CALL(cudaMallocHost((void**)&name, sizeof(type) * size));
				// put memory that is often moved between host and device onto pinned memory, it lives "closer" to the device (to my knowledge)
};


#define MEMORY_VARIABLES \
    /* Host Memory */ \
    X(float, D_chan_samp_temp, Host, C * M * T) \
	X(bool,  isSpike, Host, T * W) \
    \
    /* Pinned Host Memory */ \
    X(float, fetchBuf, Pinned, W * C) \
    X(float, omp_x, Pinned, W * T) \
	X(long, spikeTemplates, Pinned, unclu_T * W) \
	X(long, spikeTimes, Pinned, unclu_T * W) \
	X(float, spikeAmplitudes, Pinned, unclu_T * W) \
	X(float, closest_x, Pinned, unclu_T * W) \
	X(float, closest_y, Pinned, unclu_T * W) \
    \
    /* Device Memory */ \
    X(float, d_convResult, Device, 5 * (W + 2 * (M - 1)) * unclu_T) \
	X(float, d_convNormalized, Device, 5 * (W + 2 * (M - 1)) * unclu_T) \
	X(float, d_maxPoolResult, Device, W * T) \
	X(bool,  d_isSpike, Device, W * T) \
    X(float, d_fetchBuf2, Device, W * C) \
    X(float, d_whitening, Device, C * C) \
    X(float, d_fetchBuf, Device, W * C) \
    X(float, d_means, Device, C) \
    X(float, d_omp_x, Device, W * T) \
    X(Npp8u, d_nppMaxBuf, Device, maxBufferSize) \
    X(Npp8u, d_nppMinBuf, Device, minBufferSize) \
    X(float, d_nppValBuf, Device, 1) \
    X(int, d_nppIndBuf, Device, 1) \
    X(float, d_absBuf, Device, W * T) \
    X(float, d_rearranged, Device, C * W) \
    X(cufftComplex, d_freq_data, Device, C * (W / 2 + 1)) \
	X(float, d_ctc, Device, unclu_T * unclu_T * (2 * M + 1)) \
	X(int64_t, d_iU, Device, unclu_T) \
	X(int64_t, d_iCC, Device, numNearestChans * C) \
	X(float, d_Ucc, Device, numNearestChans * unclu_T * K) \
	X(float, d_wPCA, Device, K * M) \
	X(float, d_wPCA_permuted, Device, K * M) \
	X(float, d_Wall3, Device, unclu_T * K * C) \
	X(float, d_templateWaveforms, Device, unclu_T * M * C) \
	X(float, d_driftMatrix, Device, C * C) \
	X(float, d_nm, Device, unclu_T) \
	X(float, d_batchPCA, Device, C * K * W) \
	X(float, d_maxAtTime, Device, W) \
	X(long, d_imax, Device, W) \
	X(float, d_Cfmaxpool, Device, W) \
	X(float, d_amps, Device, W * unclu_T) \
	X(float, d_residual, Device, C * W) \
	X(float, d_residualContribution, Device, C * W) \
	X(float, d_convContribution, Device, unclu_T * W) \
	X(long, d_spikeTemplates, Device, W * unclu_T) \
	X(long, d_spikeTimes, Device, W * unclu_T) \
	X(float, d_batchTransposed, Device, C * W) \
	X(float, d_convResultTransposed, Device, 5 * (W + 2 * (M - 1)) * unclu_T) \
	X(cufftComplex, d_hpworkspace, Device, C * W) \
	X(cufftComplex, d_hpworkspace2, Device, C * W) \
	X(float, d_highpassed, Device, C * W) \
	X(int, d_count, Device, 1) \
	X(float, d_hpFilterFull, Device, 60000) \
	X(float, d_hpFilterSub, Device, W) \
	X(cufftComplex, d_hpFilterFreq, Device, W) \
	X(cufftComplex, d_batchFreq, Device, C * W) \
	X(float, d_shifted, Device, C * W) \
	X(float, d_xfeat, Device, numNearestChans * W * unclu_T * K) \
	X(float, d_tF, Device, numNearestChans * W * unclu_T * K) \
	X(float, d_xc, Device, C) \
	X(float, d_yc, Device, C) \
	X(float, d_xs, Device, W * unclu_T) \
	X(float, d_ys, Device, W * unclu_T) \
	X(float, d_p2pBatch, Device, W * C) \
	X(float, d_clusterCentroidsPca, Device, T * K) \