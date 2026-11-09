#ifndef DRIFTESTIMATOR_H
#define DRIFTESTIMATOR_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "DriftFingerprint.h"

//
// DriftEstimator -- live rigid drift estimation for OnlineSpikesV2.
//
//   Every batch, scans the whitened (but NOT yet drift-corrected) data with
//   Kilosort's universal templates, giving a set of (depth, amplitude) pairs.
//   Those accumulate over a window; when the window closes a worker thread
//   bins them into a depth x amplitude fingerprint, registers that against a
//   training-derived reference to get a vertical shift in microns, rebuilds the
//   drift-correction matrix for that shift, and publishes it by pointer swap.
//
//   The shift is measured against a fixed reference on UNCORRECTED data and
//   assigned, never integrated. An integrator here random-walked to a +/-10 um
//   rail in a previous attempt because the per-window residual carries no
//   feedback about the correction already applied.


struct DriftParams {
	bool  enabled        = false;
	float windowSeconds  = 10.0f;
	float maxShiftUm     = 50.0f;
};

// Snapshot of the estimator's state for the GUI / payload.
struct DriftStatus {
	float shiftUm     = 0.0f;   // latest estimate, Kilosort dshift convention
	long  updateCt    = 0;      // stream sample count at window close
	long  spikeCount  = 0;      // detections in that window
	bool  clamped     = false;  // estimate hit the +/-maxShiftUm rail
};

class DriftEstimator {
public:
	DriftEstimator();
	~DriftEstimator();

	DriftEstimator(const DriftEstimator&) = delete;
	DriftEstimator& operator=(const DriftEstimator&) = delete;

	// Load the universal-template detector and reference from oss_input/
	bool load(const std::string& ossInputDir, long C, long maxBatchSamples,
	          int deviceIndex, float sampleRate,
	          const float* d_xc, const float* d_yc,
	          float* d_matA, float* d_matB, const DriftParams& params);

	bool isEnabled() const { return m_enabled; }
	void detectAndAccumulate(const float* d_whitened, long N, long batchEndCt);
	const float* activeDriftMatrix() const {
		return m_activeMatrix.load(std::memory_order_acquire);
	}

	DriftStatus status() const;

private:
	void  workerLoop();
	void  processWindow(std::vector<float> depths, std::vector<float> amps,
	                    long windowEndCt, long scanned, long span, long batches);
	void  resolveMatrixOrientation();
	bool  loadScalars(const std::string& ossInputDir);
	void  closeWindow(long endCt);

	bool  m_enabled = false;
	DriftParams m_params;

	// geometry / detector shapes
	long  m_C = 0, m_maxSamples = 0;
	int   m_deviceIndex = 0;
	float m_sampleRate = 30000.0f;
	int   m_nk = 6;          // temporal universal templates
	int   m_nt = 61;         // template length
	int   m_nt0min = 20;     // temporal NMS half-window
	int   m_nC = 10;         // channels per template position
	int   m_nC2 = 100;       // template positions in the spatial NMS
	int   m_nSizes = 5;      // spatial envelope widths
	long  m_nFilt = 0;       // surviving universal-template grid positions
	long  m_spikeCapacity = 0;

	// fingerprint / registration parameters
	float m_thUniversal = 9.0f;
	float m_binningDepth = 5.0f;
	float m_sigInterp = 20.0f;
	float m_ycMin = 0.0f, m_ycMax = 0.0f;
	float m_dshiftLast = 0.0f;
	long  m_dmax = 0;
	int   m_nAmpBins = 20;
	long  m_batchSize = 60000;   // Kilosort's batch length, for reference bins

	std::vector<float> m_reference;   // [dmax * nAmpBins], mean-subtracted
	float*   d_wTEMP   = nullptr;   // [nk, nt]
	int*     d_iC      = nullptr;   // [nC, nFilt]
	int*     d_iC2     = nullptr;   // [nC2, nFilt]
	float*   d_weigh   = nullptr;   // [nSizes, nC, nFilt]
	float*   d_B       = nullptr;   // [C, nk, N]
	float*   d_As      = nullptr;   // [nFilt, N]
	int*     d_imaxs   = nullptr;   // [nFilt, N] signed 1-based variant id
	float*   d_Amaxs   = nullptr;   // [nFilt, N]
	int*     d_peakFilt = nullptr;  // [spikeCapacity]
	int*     d_peakTime = nullptr;  // [spikeCapacity]
	float*   d_spikeDepth = nullptr;
	float*   d_spikeAmp   = nullptr;
	int*     d_peakCount  = nullptr;

	float*   h_spikeDepth = nullptr;  // pinned
	float*   h_spikeAmp   = nullptr;  // pinned

	// borrowed from OnlineSpikesV2's pool
	const float* d_xc = nullptr;
	const float* d_yc = nullptr;
	float*   d_iKxx = nullptr;
	float*   d_Kyx = nullptr;
	float*   d_matA = nullptr;
	float*   d_matB = nullptr;

	std::atomic<const float*> m_activeMatrix{ nullptr };
	bool m_matrixNeedsTranspose = false;

	// window accumulation (sorter thread only)
	std::vector<float> m_winDepths, m_winAmps;
	long m_windowStartCt = -1;
	long m_windowSamples = 0;
	long m_winScanned = 0;
	long m_winBatches = 0;
	long m_lastSearchedEnd = -1;

	// worker handoff
	std::thread m_worker;
	mutable std::mutex m_mutex;
	std::condition_variable m_cv;
	std::vector<float> m_jobDepths, m_jobAmps;
	long m_jobEndCt = 0;
	long m_jobScanned = 0;
	long m_jobSpan = 0;
	long m_jobBatches = 0;
	bool m_jobReady = false;
	bool m_stop = false;
	cudaStream_t m_stream = nullptr;
	cublasHandle_t m_cublas = nullptr;

	// published status
	std::atomic<float> m_shiftUm{ 0.0f };
	std::atomic<long>  m_updateCt{ 0 };
	std::atomic<long>  m_spikeCount{ 0 };
	std::atomic<bool>  m_clamped{ false };
	long m_clampCount = 0;
	long m_windowCount = 0;
};

#endif
