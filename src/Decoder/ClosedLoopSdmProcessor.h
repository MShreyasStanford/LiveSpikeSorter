#ifndef CLOSED_LOOP_SDM_PROCESSOR_H_
#define CLOSED_LOOP_SDM_PROCESSOR_H_

#include "SdmProcessor.h"
#include <unordered_map>
#include <unordered_set>
#include <string>

// Closed-loop SDM processor: splits spikes into FS and RS populations 
//
// Two statistic modes (sdm_param "mode"; keys declared in src/Python/sdm_processors.py):
// (low/high), so FS and RS need not share a value and the deadband need not be symmetric:
//   "median" : state = -1 (low)  if V_P(bin) < median_P - offLow_P
//                      +1 (high) if V_P(bin) > median_P + offHigh_P, else 0.
//   "zscore" : z = (V_P(bin)-mean_P)/sd_P;
//              -1 if z < -zLow_P, +1 if z > +zHigh_P, else 0.
// where V_P(bin) = (total spikes of pop P in the bin) / n_P  -- the same value
// closedloop_baseline.py uses, so training and live agree.
//
// Wire format: 16 bytes:
// LSSClosedLoopSocket in et_trialNEW.m:
//     int32 FS_state | int32 RS_state | uint64 glxSampleCt
class ClosedLoopSdmProcessor : public SdmProcessor {
public:
	ClosedLoopSdmProcessor();
	~ClosedLoopSdmProcessor() override;

	void init(const InputParameters& params,
	          const std::vector<long>& activitySubset) override;

	void onSpikes(const std::vector<long>& times,
	              const std::vector<long>& templates,
	              long streamSampleCt) override;

	float computeBinValue(long binEndSampleCt, int8_t& direction) override;

	// Emits the 16-byte [FS,RS,sample] packet.
	void sendPacket(Sock& sdmSock, uint64_t glxSampleCt, long binEndSampleCt) override;

private:
	enum Mode { MODE_MEDIAN = 0, MODE_ZSCORE = 1 };

	bool loadLabels(const std::string& path);
	bool loadStats(const std::string& path);

	int stateFor(double popValue, int nPop,
	             double median, double mean, double sd,
	             double offLow, double offHigh, double zLow, double zHigh) const;

	long takeBinCount(std::unordered_map<long, long>& counts, long binIdx) const;

	Mode m_mode;

	float m_fsOffLow, m_fsOffHigh, m_rsOffLow, m_rsOffHigh;   // median deadband
	float m_fsZLow, m_fsZHigh, m_rsZLow, m_rsZHigh;           // zscore triggers
	float m_samplingRateHz;
	int m_binMs;
	long m_binSamples;

	std::unordered_set<long> m_fsIds;
	std::unordered_set<long> m_rsIds;
	int m_fsN;
	int m_rsN;

	double m_fsMedian, m_rsMedian;
	double m_fsMean, m_fsSd;
	double m_rsMean, m_rsSd;
	bool m_statsLoaded;

	std::unordered_map<long, long> m_fsBinCounts;
	std::unordered_map<long, long> m_rsBinCounts;
};

#endif /* CLOSED_LOOP_SDM_PROCESSOR_H_ */
