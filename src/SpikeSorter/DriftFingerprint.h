#ifndef DRIFTFINGERPRINT_H
#define DRIFTFINGERPRINT_H


#include <algorithm>
#include <cmath>
#include <vector>

namespace driftfp {

// Not tunable -- these are structural constants of align_block2.
static const int   N_COARSE     = 15;    // integer search, +/-15 bins
static const int   N_FINE       = 5;     // residual search, +/-5 bins
static const int   UPSAMPLE     = 10;    // bin resolution (10 -> 0.1)
static const float KERNEL_SIGMA = 1.0f;  // Gaussian interpolation width, BINS
static const int   ALIGN_NITER  = 10;    // reference alignment iterations
static const float AMP_CLIP     = 99.0f;
static const float AMP_TOP      = 100.0f;

inline long fingerprintRows(float ycMin, float ycMax, float binningDepth)
{
	return (long)(1 + std::ceil((ycMax - (ycMin - 1.0f)) / binningDepth));
}

inline void buildFingerprint(const std::vector<float>& depths,
                             const std::vector<float>& amps,
                             float ycMin, float binningDepth,
                             float thUniversal, long dmax, int nAmpBins,
                             std::vector<float>& F)
{
	F.assign((size_t)dmax * nAmpBins, 0.0f);
	const float dmin = ycMin - 1.0f;
	const float denom = std::log10(AMP_TOP) - std::log10(thUniversal);

	const size_t n = (std::min)(depths.size(), amps.size());
	for (size_t i = 0; i < n; ++i) {
		long row = (long)((depths[i] - dmin) / binningDepth);
		if (row < 0) row = 0;
		if (row >= dmax) row = dmax - 1;

		float a = (std::log10((std::min)(AMP_CLIP, amps[i])) -
		           std::log10(thUniversal)) / denom;
		long col = (long)(1e-5f + a * nAmpBins);
		if (col < 0) col = 0;
		if (col >= nAmpBins) col = nAmpBins - 1;

		F[(size_t)row * nAmpBins + col] += 1.0f;
	}
	for (size_t k = 0; k < F.size(); ++k) F[k] = std::log2(1.0f + F[k]);
}

// Mean over DEPTH, 
inline void meanSubtractDepth(std::vector<float>& F, long dmax, int nAmpBins)
{
	for (int c = 0; c < nAmpBins; ++c) {
		double mean = 0.0;
		for (long r = 0; r < dmax; ++r) mean += F[(size_t)r * nAmpBins + c];
		mean /= (double)dmax;
		for (long r = 0; r < dmax; ++r) F[(size_t)r * nAmpBins + c] -= (float)mean;
	}
}

// mean over (depth, amp) of roll(F, shift) * F0.
inline double corrAtShift(const std::vector<float>& F,
                          const std::vector<float>& F0,
                          long dmax, int nAmpBins, int shift)
{
	double acc = 0.0;
	for (long r = 0; r < dmax; ++r) {
		long src = ((r - shift) % dmax + dmax) % dmax;
		const float* fr = &F[(size_t)src * nAmpBins];
		const float* f0 = &F0[(size_t)r * nAmpBins];
		for (int c = 0; c < nAmpBins; ++c) acc += (double)fr[c] * (double)f0[c];
	}
	return acc / (double)(dmax * nAmpBins);
}

inline float upsampledPeak(const std::vector<double>& dc, int nFine)
{
	const int taps = 2 * nFine + 1;
	const int nUp = 2 * nFine * UPSAMPLE + 1;
	double best = -1e300;
	float bestShift = 0.0f;
	for (int j = 0; j < nUp; ++j) {
		const float sf = -(float)nFine +
		                 (float)j * (2.0f * nFine) / (float)(nUp - 1);
		double acc = 0.0;
		for (int i = 0; i < taps; ++i) {
			const float d = sf - (float)(i - nFine);
			acc += std::exp(-(d * d) / (2.0f * KERNEL_SIGMA * KERNEL_SIGMA)) * dc[i];
		}
		if (acc > best) { best = acc; bestShift = sf; }
	}
	return bestShift;
}

// Roll a fingerprint circularly along depth by `shift` bins.
inline void rollDepth(const std::vector<float>& in, std::vector<float>& out,
                      long dmax, int nAmpBins, int shift)
{
	out.resize(in.size());
	for (long r = 0; r < dmax; ++r) {
		long src = ((r - shift) % dmax + dmax) % dmax;
		std::copy_n(&in[(size_t)src * nAmpBins], nAmpBins,
		            &out[(size_t)r * nAmpBins]);
	}
}

inline float registerRigid(const std::vector<float>& Fin,
                           const std::vector<float>& F0,
                           long dmax, int nAmpBins, float binningDepth,
                           float maxShiftUm, bool* clamped = nullptr,
                           int* coarseOut = nullptr, bool* coarseRailed = nullptr)
{
	std::vector<float> F = Fin;
	meanSubtractDepth(F, dmax, nAmpBins);

	int coarse = 0;
	double best = -1e300;
	for (int t = -N_COARSE; t <= N_COARSE; ++t) {
		const double v = corrAtShift(F, F0, dmax, nAmpBins, t);
		if (v > best) { best = v; coarse = t; }
	}

	std::vector<double> dcFine(2 * N_FINE + 1);
	for (int i = 0; i < (int)dcFine.size(); ++i)
		dcFine[i] = corrAtShift(F, F0, dmax, nAmpBins, coarse + (i - N_FINE));
	const float fine = upsampledPeak(dcFine, N_FINE);

	float shiftUm = ((float)coarse + fine) * binningDepth;

	if (clamped) *clamped = false;
	if (maxShiftUm > 0.0f) {
		if (shiftUm > maxShiftUm)  { shiftUm =  maxShiftUm; if (clamped) *clamped = true; }
		if (shiftUm < -maxShiftUm) { shiftUm = -maxShiftUm; if (clamped) *clamped = true; }
	}
	if (coarseOut) *coarseOut = coarse;
	if (coarseRailed) *coarseRailed = (coarse == N_COARSE || coarse == -N_COARSE);
	return shiftUm;
}

inline bool alignReference(const std::vector<std::vector<float> >& stack,
                           long dmax, int nAmpBins, std::vector<float>& F0)
{
	const long nb = (long)stack.size();
	if (nb < 2) return false;
	const size_t n = (size_t)dmax * nAmpBins;

	std::vector<std::vector<float> > Fg(stack);
	for (long b = 0; b < nb; ++b) meanSubtractDepth(Fg[b], dmax, nAmpBins);

	// Kilosort seeds with batch 300
	F0 = Fg[(size_t)std::min<long>(300, nb / 2)];

	std::vector<float> rolled, accum(n);
	for (int iter = 0; iter < ALIGN_NITER; ++iter) {
		if (iter < ALIGN_NITER - 1) {
			for (long b = 0; b < nb; ++b) {
				int bestShift = 0;
				double best = -1e300;
				for (int t = -N_COARSE; t <= N_COARSE; ++t) {
					const double v = corrAtShift(Fg[b], F0, dmax, nAmpBins, t);
					if (v > best) { best = v; bestShift = t; }
				}
				if (bestShift != 0) {
					rollDepth(Fg[b], rolled, dmax, nAmpBins, bestShift);
					Fg[b].swap(rolled);
				}
			}
		}
		std::fill(accum.begin(), accum.end(), 0.0f);
		for (long b = 0; b < nb; ++b)
			for (size_t k = 0; k < n; ++k) accum[k] += Fg[b][k];
		for (size_t k = 0; k < n; ++k) accum[k] /= (float)nb;
		F0 = accum;
	}
	return true;
}

}

#endif
