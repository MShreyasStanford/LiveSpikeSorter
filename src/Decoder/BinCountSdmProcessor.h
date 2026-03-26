#ifndef BINCOUNT_SDM_PROCESSOR_H_
#define BINCOUNT_SDM_PROCESSOR_H_

#include "SdmProcessor.h"
#include <vector>
#include <unordered_map>
#include <deque>
#include <utility>

class BinCountSdmProcessor : public SdmProcessor {
public:
	BinCountSdmProcessor();
	~BinCountSdmProcessor() override;

	void init(const InputParameters& params,
	          const std::vector<long>& activitySubset) override;

	void onSpikes(const std::vector<long>& times,
	              const std::vector<long>& channels,
	              long streamSampleCt) override;

	float computeBinValue(long binEndSampleCt, int8_t& direction) override;

	void sendHello(Sock& sdmSock) override;
	void sendPacket(Sock& sdmSock, uint64_t glxSampleCt, long binEndSampleCt) override;
	void onBatchComplete(Sock& sdmSock, uint64_t glxSampleCt, long streamSampleCt) override;
	void setNumTemplates(long numTemplates) override;

private:
	void rebuildChannelList(long numTemplates);

	// Sorted channel IDs (same order sent in hello and used for data packets)
	std::vector<long> m_channels;

	// Mapping from channel ID to index in m_channels for fast lookup
	std::unordered_map<long, int> m_channelIndex;

	long m_windowSamples;

	// True when the user left the subset field empty (use all templates)
	bool m_useAllTemplates;

	// Sliding window of recent spike events: (sampleTime, channelIndex)
	std::deque<std::pair<long, int>> m_spikeEvents;
};

#endif /* BINCOUNT_SDM_PROCESSOR_H_ */
