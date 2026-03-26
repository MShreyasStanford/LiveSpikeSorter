#include "SdmProcessor.h"
#include "../Networking/Sock.h"
#include <cstring>

void SdmProcessor::sendHello(Sock& /*sdmSock*/) {
	// No-op for zscore/logreg — they use the existing 13-byte hello in Decoder.
}

void SdmProcessor::sendPacket(Sock& sdmSock, uint64_t glxSampleCt, long binEndSampleCt) {
	int8_t dir;
	float value = computeBinValue(binEndSampleCt, dir);
	uint8_t sdmBuf[13];
	sdmBuf[0] = static_cast<uint8_t>(dir);
	std::memcpy(&sdmBuf[1], &value, sizeof(float));
	std::memcpy(&sdmBuf[5], &glxSampleCt, sizeof(uint64_t));
	sdmSock.sendData(sdmBuf, static_cast<uint>(sizeof(sdmBuf)));
}

void SdmProcessor::onBatchComplete(Sock& /*sdmSock*/, uint64_t /*glxSampleCt*/, long /*streamSampleCt*/) {
	// No-op — only overridden by sliding-window processors (e.g. BinCountSdmProcessor).
}

void SdmProcessor::setNumTemplates(long /*numTemplates*/) {
	// No-op — only used by BinCountSdmProcessor when no subset is specified.
}
