#include "DriftEstimator.h"
#include "myGPUhelpers.h"
#include "../Helpers/Utils.h"
#include "CNPY/cnpy.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#define DE_TPB 256

// Upper bound on Kilosort's nearest_chans
#define DE_MAX_NEAREST_CHANS 16

// Below this a window is skipped rather than registered
static const long DE_MIN_SPIKES = 300;

// ==========================================================================
// ==========================================================================


__global__ void universal_match_kernel(
	const float* __restrict__ B,       // [C, nk, N]
	const int*   __restrict__ iC,      // [nC, nFilt]
	const float* __restrict__ weigh,   // [nSizes, nC, nFilt]
	int nk, int nC, int nSizes, long nFilt, long N,
	float* __restrict__ As,            // [nFilt, N]
	int*   __restrict__ imaxs)         // [nFilt, N]
{
	extern __shared__ char smem[];
	int*   s_iC    = reinterpret_cast<int*>(smem);            // nC
	float* s_weigh = reinterpret_cast<float*>(s_iC + nC);     // nSizes * nC

	const long filt = blockIdx.y;
	if (filt >= nFilt) return;

	for (int i = threadIdx.x; i < nC; i += blockDim.x)
		s_iC[i] = iC[(long)i * nFilt + filt];
	for (int i = threadIdx.x; i < nSizes * nC; i += blockDim.x) {
		int size = i / nC, c = i % nC;
		s_weigh[i] = weigh[((long)size * nC + c) * nFilt + filt];
	}
	__syncthreads();

	const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= N) return;

	float bestAbs = -1.0f;
	int   bestVariant = 0;
	float bestSigned = 0.0f;

	for (int k = 0; k < nk; ++k) {
		// Load this temporal template's response on all nC channels once,

		float b[DE_MAX_NEAREST_CHANS];     // nC <= this, enforced at load()
		#pragma unroll
		for (int c = 0; c < DE_MAX_NEAREST_CHANS; ++c)
			b[c] = (c < nC) ? B[((long)s_iC[c] * nk + k) * N + t] : 0.0f;

		for (int size = 0; size < nSizes; ++size) {
			float v = 0.0f;
			#pragma unroll
			for (int c = 0; c < DE_MAX_NEAREST_CHANS; ++c)
				v += (c < nC) ? (s_weigh[size * nC + c] * b[c]) : 0.0f;
			float av = fabsf(v);
			if (av > bestAbs) {
				bestAbs = av;
				bestVariant = size * nk + k;   // to match KS's reshape order
				bestSigned = v;
			}
		}
	}

	As[filt * N + t] = bestAbs;
	imaxs[filt * N + t] = (bestSigned < 0.0f) ? -(bestVariant + 1)
	                                          : (bestVariant + 1);
}

// Amaxs[filt, t] = max over the nC2 nearest template positions of As[.., t].
__global__ void spatial_nms_kernel(
	const float* __restrict__ As,      // [nFilt, N]
	const int*   __restrict__ iC2,     // [nC2, nFilt]
	int nC2, long nFilt, long N, long guard,
	float* __restrict__ Amaxs)
{
	extern __shared__ char smem[];
	int* s_iC2 = reinterpret_cast<int*>(smem);

	const long filt = blockIdx.y;
	if (filt >= nFilt) return;

	for (int i = threadIdx.x; i < nC2; i += blockDim.x)
		s_iC2[i] = iC2[(long)i * nFilt + filt];
	__syncthreads();

	const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= N) return;
	if (t < guard || t >= N - guard) {
		Amaxs[filt * N + t] = 0.0f;
		return;
	}

	float m = -1.0f;
	for (int i = 0; i < nC2; ++i) {
		float v = As[(long)s_iC2[i] * N + t];
		if (v > m) m = v;
	}
	Amaxs[filt * N + t] = m;
}

__global__ void collect_peaks_kernel(
	const float* __restrict__ As,
	const float* __restrict__ Amaxs,
	int nt0min, long nFilt, long N, long guard, float th,
	int* __restrict__ peakFilt, int* __restrict__ peakTime,
	int* __restrict__ count, int capacity)
{
	const long filt = blockIdx.y;
	if (filt >= nFilt) return;
	const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= N) return;

	if (t < guard || t >= N - guard) return;

	const float a = As[filt * N + t];
	if (a <= th) return;

	long lo = t - nt0min; if (lo < 0) lo = 0;
	long hi = t + nt0min; if (hi > N - 1) hi = N - 1;
	float pooled = -1.0f;
	for (long u = lo; u <= hi; ++u) {
		float v = Amaxs[filt * N + u];
		if (v > pooled) pooled = v;
	}
	if (pooled != a) return;

	int slot = atomicAdd(count, 1);
	if (slot < capacity) {
		peakFilt[slot] = (int)filt;
		peakTime[slot] = (int)t;
	}
}

__global__ void yweighted_kernel(
	const float* __restrict__ B,       // [C, nk, N]
	const float* __restrict__ As,      // [nFilt, N]
	const int*   __restrict__ imaxs,   // [nFilt, N]
	const int*   __restrict__ iC,      // [nC, nFilt]
	const float* __restrict__ yc,      // [C]
	const int*   __restrict__ peakFilt,
	const int*   __restrict__ peakTime,
	int nk, int nC, long nFilt, long N, int nSpikes,
	float* __restrict__ depth, float* __restrict__ amp)
{
	int s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= nSpikes) return;

	const long filt = peakFilt[s];
	const long t    = peakTime[s];

	const int code = imaxs[filt * N + t];
	const float sign = (code < 0) ? -1.0f : 1.0f;
	const int variant = (code < 0 ? -code : code) - 1;
	const int k = variant % nk;          // KS: imax % nk recovers the template

	float wsum = 0.0f, ysum = 0.0f;
	for (int c = 0; c < nC; ++c) {
		const int chan = iC[(long)c * nFilt + filt];
		float a = B[((long)chan * nk + k) * N + t] * sign;
		if (a > 0.0f) {                  // relu
			wsum += a;
			ysum += a * yc[chan];
		}
	}

	depth[s] = (wsum > 0.0f) ? (ysum / wsum) : yc[iC[filt]];
	amp[s]   = As[filt * N + t];
}

// ==========================================================================

DriftEstimator::DriftEstimator() {}

DriftEstimator::~DriftEstimator()
{
	if (m_worker.joinable()) {
		{
			std::lock_guard<std::mutex> lk(m_mutex);
			m_stop = true;
		}
		m_cv.notify_all();
		m_worker.join();
	}
	if (m_cublas) cublasDestroy(m_cublas);
	if (m_stream) cudaStreamDestroy(m_stream);

	// d_matA / d_matB are NOT freed here: they belong to OnlineSpikesV2's X-macro tensor pool
	cudaFree(d_wTEMP);  cudaFree(d_iC);    cudaFree(d_iC2);  cudaFree(d_weigh);
	cudaFree(d_B);      cudaFree(d_As);    cudaFree(d_imaxs);
	cudaFree(d_Amaxs);  cudaFree(d_peakFilt); cudaFree(d_peakTime);
	cudaFree(d_spikeDepth); cudaFree(d_spikeAmp); cudaFree(d_peakCount);
	cudaFree(d_iKxx);   cudaFree(d_Kyx);
	if (h_spikeDepth) cudaFreeHost(h_spikeDepth);
	if (h_spikeAmp)   cudaFreeHost(h_spikeAmp);
}

// ==========================================================================
// Loading

bool DriftEstimator::loadScalars(const std::string& dir)
{
	std::ifstream inf(dir + "misc.txt");
	if (!inf.is_open()) {
		std::cout << "[Drift] cannot open " << dir << "misc.txt" << std::endl;
		return false;
	}
	std::map<std::string, std::string> p;
	std::string line;
	while (std::getline(inf, line)) {
		size_t pos = line.find(':');
		if (pos == std::string::npos) continue;
		p[line.substr(0, pos)] = line.substr(pos + 1);
	}

	auto need = [&](const char* key, float& out) -> bool {
		auto it = p.find(key);
		if (it == p.end() || it->second.empty()) {
			std::cout << "[Drift] misc.txt is missing '" << key
			          << "'; re-run Kilosort4 once to regenerate oss_input/."
			          << std::endl;
			return false;
		}
		try { out = std::stof(it->second); }
		catch (const std::exception&) {
			std::cout << "[Drift] misc.txt value for '" << key
			          << "' is not a number: '" << it->second << "'" << std::endl;
			return false;
		}
		return true;
	};

	float thUniv, nTemplates, templateSizes, nearestTemplates, nt, nFilt;
	float nt0min, nearestChans, binningDepth, sigInterp, ycMin, ycMax;
	float dshiftLast, nAmpBins, batchSize;
	if (!need("Th_universal", thUniv) || !need("n_templates", nTemplates) ||
	    !need("template_sizes", templateSizes) ||
	    !need("nearest_templates", nearestTemplates) || !need("nt", nt) ||
	    !need("n_filters", nFilt) || !need("nt0min", nt0min) ||
	    !need("numNearestChans", nearestChans) ||
	    !need("binning_depth", binningDepth) || !need("sig_interp", sigInterp) ||
	    !need("yc_min", ycMin) || !need("yc_max", ycMax) ||
	    !need("dshift_last", dshiftLast) || !need("n_amp_bins", nAmpBins) ||
	    !need("batch_size", batchSize))
		return false;

	m_thUniversal  = thUniv;
	m_nk           = (int)nTemplates;
	m_nSizes       = (int)templateSizes;
	m_nC2          = (int)nearestTemplates;
	m_nt           = (int)nt;
	m_nFilt        = (long)nFilt;
	m_nt0min       = (int)nt0min;
	m_nC           = (int)nearestChans;
	m_binningDepth = binningDepth;
	m_sigInterp    = sigInterp;
	m_ycMin        = ycMin;
	m_ycMax        = ycMax;
	m_dshiftLast   = dshiftLast;
	m_nAmpBins     = (int)nAmpBins;
	m_batchSize    = (long)batchSize;

	m_dmax = driftfp::fingerprintRows(m_ycMin, m_ycMax, m_binningDepth);

	if (m_nC > DE_MAX_NEAREST_CHANS) {
		std::cout << "[Drift] nearest_chans=" << m_nC << " exceeds the "
		          << DE_MAX_NEAREST_CHANS << "-channel register array in "
		             "universal_match_kernel." << std::endl;
		return false;
	}
	if (m_thUniversal <= 0.0f || m_binningDepth <= 0.0f || m_dmax <= 1) {
		std::cout << "[Drift] nonsensical scalars in misc.txt (Th_universal="
		          << m_thUniversal << ", binning_depth=" << m_binningDepth
		          << ", dmax=" << m_dmax << ")." << std::endl;
		return false;
	}
	return true;
}

bool DriftEstimator::load(const std::string& dir, long C, long maxBatchSamples,
                          int deviceIndex, float sampleRate,
                          const float* xc, const float* yc,
                          float* matA, float* matB, const DriftParams& params)
{
	static const char* ptLabel = { "DriftEstimator::load" };

	m_params = params;
	m_C = C;
	m_maxSamples = maxBatchSamples;
	m_deviceIndex = deviceIndex;
	m_sampleRate = sampleRate;
	d_xc = xc; d_yc = yc; d_matA = matA; d_matB = matB;
	m_activeMatrix.store(matA, std::memory_order_release);

	if (!params.enabled) return false;
	if (!loadScalars(dir)) return false;

	try {
		auto npW    = cnpy::npy_load(dir + "ud_wTEMP.npy");
		auto npiC   = cnpy::npy_load(dir + "ud_iC.npy");
		auto npiC2  = cnpy::npy_load(dir + "ud_iC2.npy");
		auto npWe   = cnpy::npy_load(dir + "ud_weigh.npy");
auto npF0   = cnpy::npy_load(dir + "ks_F0.npy");
		auto npiKxx = cnpy::npy_load(dir + "iKxx.npy");

		auto rank = [](const cnpy::NpyArray& a, size_t r, const char* name) -> bool {
			if (a.shape.size() != r) {
				std::cout << "[Drift] " << name << " has rank " << a.shape.size()
				          << ", expected " << r << "." << std::endl;
				return false;
			}
			return true;
		};
		if (!rank(npW, 2, "ud_wTEMP.npy") || !rank(npiC, 2, "ud_iC.npy") ||
		    !rank(npiC2, 2, "ud_iC2.npy") || !rank(npWe, 3, "ud_weigh.npy") ||
		    !rank(npF0, 2, "ks_F0.npy") || !rank(npiKxx, 2, "iKxx.npy"))
			return false;

		if (npW.word_size != sizeof(float) || npWe.word_size != sizeof(float) ||
		    npF0.word_size != sizeof(float) || npiKxx.word_size != sizeof(float) ||
		    npiC.word_size != sizeof(int) || npiC2.word_size != sizeof(int)) {
			std::cout << "[Drift] unexpected word size in the drift exports; "
			             "re-run Kilosort4 with the current launcher."
			          << std::endl;
			return false;
		}

		auto dim = [](const cnpy::NpyArray& a, int i) -> long {
			return (long)a.shape[i];
		};
		if (dim(npW, 0) != m_nk || dim(npW, 1) != m_nt ||
		    dim(npiC, 0) != m_nC || dim(npiC, 1) != m_nFilt ||
		    dim(npiC2, 0) != m_nC2 || dim(npiC2, 1) != m_nFilt ||
		    dim(npWe, 0) != m_nSizes || dim(npWe, 1) != m_nC ||
		    dim(npWe, 2) != m_nFilt) {
			std::cout << "[Drift] universal-template exports disagree with "
			             "misc.txt; re-run Kilosort4 to regenerate oss_input/."
			          << std::endl;
			return false;
		}
		if (dim(npF0, 0) != m_dmax || dim(npF0, 1) != m_nAmpBins) {
			std::cout << "[Drift] ks_F0.npy is " << dim(npF0, 0) << "x"
			          << dim(npF0, 1) << ", expected " << m_dmax << "x"
			          << m_nAmpBins << "." << std::endl;
			return false;
		}
		if (dim(npiKxx, 0) != C || dim(npiKxx, 1) != C) {
			std::cout << "[Drift] iKxx.npy is not " << C << "x" << C << "."
			          << std::endl;
			return false;
		}

		m_spikeCapacity = std::max<long>(8192, maxBatchSamples * 4);

		const size_t szB = (size_t)C * m_nk * maxBatchSamples * sizeof(float);
		const size_t szF = (size_t)m_nFilt * maxBatchSamples;

		_CUDA_CALL(cudaMalloc(&d_wTEMP, (size_t)m_nk * m_nt * sizeof(float)));
		_CUDA_CALL(cudaMalloc(&d_iC,    (size_t)m_nC * m_nFilt * sizeof(int)));
		_CUDA_CALL(cudaMalloc(&d_iC2,   (size_t)m_nC2 * m_nFilt * sizeof(int)));
		_CUDA_CALL(cudaMalloc(&d_weigh, (size_t)m_nSizes * m_nC * m_nFilt * sizeof(float)));
		_CUDA_CALL(cudaMalloc(&d_B,     szB));
		_CUDA_CALL(cudaMalloc(&d_As,    szF * sizeof(float)));
		_CUDA_CALL(cudaMalloc(&d_imaxs, szF * sizeof(int)));
		_CUDA_CALL(cudaMalloc(&d_Amaxs, szF * sizeof(float)));
		_CUDA_CALL(cudaMalloc(&d_peakFilt, (size_t)m_spikeCapacity * sizeof(int)));
		_CUDA_CALL(cudaMalloc(&d_peakTime, (size_t)m_spikeCapacity * sizeof(int)));
		_CUDA_CALL(cudaMalloc(&d_spikeDepth, (size_t)m_spikeCapacity * sizeof(float)));
		_CUDA_CALL(cudaMalloc(&d_spikeAmp,   (size_t)m_spikeCapacity * sizeof(float)));
		_CUDA_CALL(cudaMalloc(&d_peakCount, sizeof(int)));
		_CUDA_CALL(cudaMallocHost(&h_spikeDepth, (size_t)m_spikeCapacity * sizeof(float)));
		_CUDA_CALL(cudaMallocHost(&h_spikeAmp,   (size_t)m_spikeCapacity * sizeof(float)));

		_CUDA_CALL(cudaMemcpy(d_wTEMP, npW.data<float>(),
		                      (size_t)m_nk * m_nt * sizeof(float), cudaMemcpyHostToDevice));
		_CUDA_CALL(cudaMemcpy(d_weigh, npWe.data<float>(),
		                      (size_t)m_nSizes * m_nC * m_nFilt * sizeof(float), cudaMemcpyHostToDevice));
		_CUDA_CALL(cudaMemcpy(d_iC, npiC.data<int>(),
		                      (size_t)m_nC * m_nFilt * sizeof(int), cudaMemcpyHostToDevice));
		_CUDA_CALL(cudaMemcpy(d_iC2, npiC2.data<int>(),
		                      (size_t)m_nC2 * m_nFilt * sizeof(int), cudaMemcpyHostToDevice));

		_CUDA_CALL(cudaMalloc(&d_iKxx, (size_t)C * C * sizeof(float)));
		_CUDA_CALL(cudaMalloc(&d_Kyx,  (size_t)C * C * sizeof(float)));
		_CUDA_CALL(cudaMemcpy(d_iKxx, npiKxx.data<float>(),
		                      (size_t)C * C * sizeof(float), cudaMemcpyHostToDevice));

		m_reference.assign(npF0.data<float>(),
		                   npF0.data<float>() + (size_t)m_dmax * m_nAmpBins);
		driftfp::meanSubtractDepth(m_reference, m_dmax, m_nAmpBins);
	}
	catch (const std::exception& e) {
		std::cout << "[Drift] could not load drift tensors (" << e.what()
		          << "). Live drift estimation stays OFF; the static "
		             "drift_matrix.npy is still applied." << std::endl;
		return false;
	}

	m_windowSamples = (long)(m_params.windowSeconds * m_sampleRate);
	if (m_windowSamples < 1) m_windowSamples = 1;
	m_winDepths.reserve(1 << 16);
	m_winAmps.reserve(1 << 16);

	m_enabled = true;
	m_worker = std::thread(&DriftEstimator::workerLoop, this);

	std::cout << "[Drift] live estimation ON: " << m_nFilt
	          << " template positions, " << m_dmax << " depth bins, window "
	          << m_params.windowSeconds << " s (" << m_windowSamples
	          << " samples), max shift " << m_params.maxShiftUm << " um, "
	          << "Th_universal " << m_thUniversal << std::endl;
	return true;
}


void DriftEstimator::detectAndAccumulate(const float* d_whitened, long N,
                                         long batchEndCt)
{
	static const char* ptLabel = { "DriftEstimator::detectAndAccumulate" };
	if (!m_enabled || N <= 2 * m_nt) return;

	if (N > m_maxSamples) {
		static bool warned = false;
		if (!warned) {
			std::cout << "[Drift] batch of " << N << " samples exceeds the "
			          << m_maxSamples << " the detector was sized for; "
			          << "skipping drift detection for oversized batches."
			          << std::endl;
			warned = true;
		}
		return;
	}

	if (m_windowStartCt < 0) m_windowStartCt = batchEndCt;

	// On batch edges: we ignore the outer nt samples at each end (Kilosort's
	// rule, since the convolution there runs off the end of the data), while
	// the sorter re-reads the last `lookback` = 2*M samples of the previous
	// batch. With the usual M == nt == 61 those exactly cancel, so coverage is
	// seamless and nothing is counted twice. If they ever differ the result is
	// a small uniform over- or under-count in the overlap, which a depth
	// histogram compared by correlation is insensitive to -- it shifts every
	// depth bin equally and is removed by the per-column mean subtraction.

	// 1. Project every channel onto the universal temporal templates.
	//    Identical operation and layout to Kilosort's conv1d(X, wTEMP): this
	//    is the existing PCA projection primitive with a different basis.
	projectToPCA(d_whitened, d_wTEMP, d_B, m_nk, m_nt, (int)m_C, (int)N);

	dim3 block(DE_TPB);
	dim3 grid((unsigned)((N + DE_TPB - 1) / DE_TPB), (unsigned)m_nFilt);

	// 2. Spatially weighted match against all 30 universal variants.
	size_t smem = (size_t)m_nC * sizeof(int) +
	              (size_t)m_nSizes * m_nC * sizeof(float);
	universal_match_kernel <<<grid, block, smem>>> (
		d_B, d_iC, d_weigh, m_nk, m_nC, m_nSizes, m_nFilt, N, d_As, d_imaxs);

	// 3. Spatial non-maximum suppression over the nearest template positions.
	spatial_nms_kernel <<<grid, block, (size_t)m_nC2 * sizeof(int)>>> (
		d_As, d_iC2, m_nC2, m_nFilt, N, m_nt, d_Amaxs);

	// 4. Temporal NMS + thresholding + compaction.
	_CUDA_CALL(cudaMemset(d_peakCount, 0, sizeof(int)));
	collect_peaks_kernel <<<grid, block>>> (
		d_As, d_Amaxs, m_nt0min, m_nFilt, N, m_nt, m_thUniversal,
		d_peakFilt, d_peakTime, d_peakCount, (int)m_spikeCapacity);

	int nSpikes = 0;
	_CUDA_CALL(cudaMemcpy(&nSpikes, d_peakCount, sizeof(int), cudaMemcpyDeviceToHost));
	if (nSpikes > m_spikeCapacity) {
		static bool warned = false;
		if (!warned) {
			std::cout << "[Drift] WARNING: " << nSpikes << " detections exceed "
			          << "the " << m_spikeCapacity << "-spike buffer; the "
			          << "fingerprint is being built from a truncated set."
			          << std::endl;
			warned = true;
		}
		nSpikes = (int)m_spikeCapacity;
	}

	if (nSpikes > 0) {
		// 5. Depth (ReLU-weighted COM) and amplitude per detection.
		int blocks = (int)((nSpikes + DE_TPB - 1) / DE_TPB);
		yweighted_kernel <<<blocks, DE_TPB>>> (
			d_B, d_As, d_imaxs, d_iC, d_yc, d_peakFilt, d_peakTime,
			m_nk, m_nC, m_nFilt, N, nSpikes, d_spikeDepth, d_spikeAmp);

		_CUDA_CALL(cudaMemcpy(h_spikeDepth, d_spikeDepth,
		                      (size_t)nSpikes * sizeof(float), cudaMemcpyDeviceToHost));
		_CUDA_CALL(cudaMemcpy(h_spikeAmp, d_spikeAmp,
		                      (size_t)nSpikes * sizeof(float), cudaMemcpyDeviceToHost));

		m_winDepths.insert(m_winDepths.end(), h_spikeDepth, h_spikeDepth + nSpikes);
		m_winAmps.insert(m_winAmps.end(), h_spikeAmp, h_spikeAmp + nSpikes);
	}
	{
		long s = batchEndCt - N + m_nt;
		const long e = batchEndCt - m_nt;
		if (s < m_lastSearchedEnd) s = m_lastSearchedEnd;
		if (e > s) {
			m_winScanned += (e - s);
			m_lastSearchedEnd = e;
		}
	}
	m_winBatches++;

	if (batchEndCt - m_windowStartCt >= m_windowSamples) closeWindow(batchEndCt);
}

void DriftEstimator::closeWindow(long endCt)
{
	{
		std::lock_guard<std::mutex> lk(m_mutex);
		if (!m_jobReady) {           // worker idle -- hand the window over
			m_jobDepths.swap(m_winDepths);
			m_jobAmps.swap(m_winAmps);
			m_jobEndCt = endCt;
			m_jobScanned = m_winScanned;
			m_jobSpan = (m_windowStartCt >= 0) ? (endCt - m_windowStartCt)
			                                   : m_windowSamples;
			m_jobBatches = m_winBatches;
			m_jobReady = true;
			m_cv.notify_one();
		}
		// If the worker is still busy this window is dropped rather than queued
	}
	m_winDepths.clear();
	m_winAmps.clear();
	m_winScanned = 0;
	m_winBatches = 0;
	m_windowStartCt = endCt;
	// m_lastSearchedEnd survives the window boundary 
}

DriftStatus DriftEstimator::status() const
{
	DriftStatus s;
	s.shiftUm    = m_shiftUm.load(std::memory_order_relaxed);
	s.updateCt   = m_updateCt.load(std::memory_order_relaxed);
	s.spikeCount = m_spikeCount.load(std::memory_order_relaxed);
	s.clamped    = m_clamped.load(std::memory_order_relaxed);
	return s;
}

// Worker thread
void DriftEstimator::workerLoop()
{
	static const char* ptLabel = { "DriftEstimator::workerLoop" };

	_CUDA_CALL(cudaSetDevice(m_deviceIndex));
	_CUDA_CALL(cudaStreamCreate(&m_stream));
	if (cublasCreate(&m_cublas) != CUBLAS_STATUS_SUCCESS)
		_RUN_ERROR(ptLabel, "failed to create the drift cuBLAS handle");
	cublasSetStream(m_cublas, m_stream);

	resolveMatrixOrientation();

	for (;;) {
		std::vector<float> depths, amps;
		long endCt = 0, scanned = 0, span = 0, batches = 0;
		{
			std::unique_lock<std::mutex> lk(m_mutex);
			m_cv.wait(lk, [&] { return m_jobReady || m_stop; });
			if (m_stop) break;
			depths  = std::move(m_jobDepths);
			amps    = std::move(m_jobAmps);
			endCt   = m_jobEndCt;
			scanned = m_jobScanned;
			span    = m_jobSpan;
			batches = m_jobBatches;
			m_jobDepths.clear();
			m_jobAmps.clear();
			m_jobReady = false;
		}
		processWindow(std::move(depths), std::move(amps), endCt,
		              scanned, span, batches);
	}
}

// Decide whether the on-disk drift_matrix.npy convention is M or M^T.
//
// The launcher saves get_drift_matrix(...).T (run_online_spikes.py), while the
// sorter applies the matrix as a plain row-major left-multiply. Rather than
// hard-code an assumption about which way round that lands, rebuild the matrix
// here at the SAME shift the launcher used and keep whichever orientation
// reproduces the file. If the export convention ever changes, this notices.
void DriftEstimator::resolveMatrixOrientation()
{
	static const char* ptLabel = { "DriftEstimator::resolveMatrixOrientation" };

	const size_t n = (size_t)m_C * m_C;
	std::vector<float> onDisk(n), rebuilt(n);
	_CUDA_CALL(cudaMemcpy(onDisk.data(), d_matA, n * sizeof(float),
	                      cudaMemcpyDeviceToHost));

	auto maxAbsDiff = [&](bool transpose) -> double {
		ComputeDriftMat(m_cublas, m_stream, d_xc, d_yc, d_iKxx, d_Kyx,
		                m_sigInterp, m_dshiftLast, (int)m_C, d_matB, transpose);
		_CUDA_CALL(cudaStreamSynchronize(m_stream));
		_CUDA_CALL(cudaMemcpy(rebuilt.data(), d_matB, n * sizeof(float),
		                      cudaMemcpyDeviceToHost));
		double m = 0.0;
		for (size_t i = 0; i < n; ++i)
			m = std::max(m, (double)std::fabs(rebuilt[i] - onDisk[i]));
		return m;
	};

	const double diffPlain = maxAbsDiff(false);
	const double diffTrans = maxAbsDiff(true);
	m_matrixNeedsTranspose = (diffTrans < diffPlain);

	std::cout << "[Drift] matrix orientation: maxdiff(M)=" << diffPlain
	          << ", maxdiff(M^T)=" << diffTrans << " -> using "
	          << (m_matrixNeedsTranspose ? "M^T" : "M") << std::endl;

	if (std::min(diffPlain, diffTrans) > 1e-2) {
		std::cout << "[Drift] WARNING: neither orientation reproduces "
		             "drift_matrix.npy. sig_interp / iKxx / dshift_last in "
		             "oss_input/ are probably inconsistent with the build that "
		             "wrote it. Live estimates will be applied anyway, but "
		             "verify before trusting them." << std::endl;
	}
}

void DriftEstimator::processWindow(std::vector<float> depths,
                                   std::vector<float> amps, long windowEndCt,
                                   long scanned, long span, long batches)
{
	static const char* ptLabel = { "DriftEstimator::processWindow" };

	const float coverage = (span > 0) ? (float)scanned / (float)span : 0.0f;

	m_windowCount++;
	if ((long)depths.size() < DE_MIN_SPIKES) {

		std::cout << "[Drift] window " << m_windowCount << " skipped ("
		          << depths.size() << " spikes < " << DE_MIN_SPIKES
		          << ", coverage " << coverage << ", " << batches << " batches)"
		          << std::endl;
		return;
	}

	std::vector<float> F;
	driftfp::buildFingerprint(depths, amps, m_ycMin, m_binningDepth,
	                          m_thUniversal, m_dmax, m_nAmpBins, F);

	bool clamped = false, coarseRailed = false;
	const float shiftUm = driftfp::registerRigid(
		F, m_reference, m_dmax, m_nAmpBins, m_binningDepth,
		m_params.maxShiftUm, &clamped, nullptr, &coarseRailed);
	if (clamped) m_clampCount++;

	const float* active = m_activeMatrix.load(std::memory_order_acquire);
	float* target = (active == d_matA) ? d_matB : d_matA;
	ComputeDriftMat(m_cublas, m_stream, d_xc, d_yc, d_iKxx, d_Kyx,
	                m_sigInterp, shiftUm, (int)m_C, target,
	                m_matrixNeedsTranspose);
	_CUDA_CALL(cudaStreamSynchronize(m_stream));

	m_activeMatrix.store(target, std::memory_order_release);
	m_shiftUm.store(shiftUm, std::memory_order_relaxed);
	m_updateCt.store(windowEndCt, std::memory_order_relaxed);
	m_spikeCount.store((long)depths.size(), std::memory_order_relaxed);
	m_clamped.store(clamped, std::memory_order_relaxed);


	// -----------------------------------------------------------------------------
	// DIAGNOSTIC OUTPUT TO PRINT ONCE PER WINDOW 
	// -----------------------------------------------------------------------------

	// Needed for drift diagnostic script - copy output into run_log.txt and run `python drift_compare.py run_log.txt`
	// Spikes detected, shift, how many windows clamped, coverage, batches, search range hit 
	std::cout << "[Drift] window " << m_windowCount << ": " << depths.size()
	          << " spikes, shift " << shiftUm << " um"
	          << (clamped ? "  [CLAMPED]" : "")
	          << (coarseRailed ? "  [SEARCH RANGE HIT]" : "")
	          << "  (coverage " << coverage << ", " << batches << " batches"
	          << ", clamped " << m_clampCount << "/" << m_windowCount << ")"
	          << std::endl;

	// -----------------------------------------------------------------------------
	// // Vram usage
	// size_t freeB, totalB; cudaMemGetInfo(&freeB, &totalB);
	// std::cout << "[Drift] VRAM used " << (totalB - freeB) / (1<<20) << "MB\n";

}
