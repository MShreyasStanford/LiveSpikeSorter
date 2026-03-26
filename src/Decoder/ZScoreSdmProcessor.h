#ifndef ZSCORE_SDM_PROCESSOR_H_
#define ZSCORE_SDM_PROCESSOR_H_

#include "SdmProcessor.h"
#include <atomic>
#include <thread>
#include <unordered_map>

class ZScoreSdmProcessor : public SdmProcessor {
public:
	ZScoreSdmProcessor();
	~ZScoreSdmProcessor() override;

	void init(const InputParameters& params,
	          const std::vector<long>& activitySubset) override;

	void onSpikes(const std::vector<long>& times,
	              const std::vector<long>& channels,
	              long streamSampleCt) override;

	float computeBinValue(long binEndSampleCt, int8_t& direction) override;

private:
	float m_triggerZ;
	float m_baselineMinSeconds;
	float m_samplingRateHz;
	int m_binMs;
	long m_binSamples;

	// Baseline state
	std::atomic<bool> m_baselineFrozen{ false };
	std::atomic<double> m_manualMean{ 0.0 };
	std::atomic<double> m_manualSd{ 0.0 };

	long m_baselineBinsSeen;
	double m_baselineMean;
	double m_baselineM2;

	// Per-bin spike counts keyed by bin index
	std::unordered_map<long, long> m_binCounts;
};

#endif /* ZSCORE_SDM_PROCESSOR_H_ */
