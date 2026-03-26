#ifndef SDM_PROCESSOR_H_
#define SDM_PROCESSOR_H_

#include <vector>
#include <cstdint>
#include "../Networking/inputParameters.h"

class Sock;

class SdmProcessor {
public:
	virtual ~SdmProcessor() = default;

	// Called once at startup with config and the activity subset channels
	virtual void init(const InputParameters& params,
	                  const std::vector<long>& activitySubset) = 0;

	// Feed incoming spikes (already filtered to subset by Decoder)
	virtual void onSpikes(const std::vector<long>& times,
	                      const std::vector<long>& channels,
	                      long streamSampleCt) = 0;

	// Called at each bin boundary. Returns the float value to send in the SDM packet.
	// Sets `direction` (1 = above threshold, 0 = below, -1 = below negative threshold)
	virtual float computeBinValue(long binEndSampleCt, int8_t& direction) = 0;

	// Called once after connection to send a hello/handshake packet.
	// Default: no-op (zscore/logreg use the existing 13-byte hello in Decoder).
	virtual void sendHello(Sock& sdmSock);

	// Called at each bin boundary to send the SDM packet.
	// Default: packs computeBinValue() result into 13-byte [dir|float|uint64] and sends.
	virtual void sendPacket(Sock& sdmSock, uint64_t glxSampleCt, long binEndSampleCt);

	// Called once per batch (after onSpikes) to allow sliding-window processors
	// to send a packet covering the trailing window.  Default: no-op.
	virtual void onBatchComplete(Sock& sdmSock, uint64_t glxSampleCt, long streamSampleCt);

	// Called after receiving SorterParameters so the processor knows the real
	// template count loaded from disk.  Default: no-op.
	virtual void setNumTemplates(long numTemplates);
};

#endif /* SDM_PROCESSOR_H_ */
