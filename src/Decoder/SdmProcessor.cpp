#include "SdmProcessor.h"
#include "../Networking/Sock.h"
#include <cstring>
#include <iostream>
#include <map>

bool SdmProcessor::useTcp() const {
	return false;
}

void SdmProcessor::sendConnectHello(Sock& sdmSock) {
	uint8_t sdmHello[13] = { 0 };
	const uint sent = sdmSock.sendData(sdmHello, static_cast<uint>(sizeof(sdmHello)));
	if (sent == 0)
		std::cerr << "SDM hello send failed: " << sdmSock.errorReason() << std::endl;
	else
		std::cout << "SDM hello sent (" << sent << " bytes)." << std::endl;
}

void SdmProcessor::sendHello(Sock& /*sdmSock*/) {
	//
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

// REGISTRATION
static std::map<std::string, SdmProcessorFactory>& sdmRegistry() {
	static std::map<std::string, SdmProcessorFactory> registry;
	return registry;
}

SdmProcessorRegistrar::SdmProcessorRegistrar(const std::string& name, SdmProcessorFactory factory) {
	sdmRegistry()[name] = std::move(factory);
}

std::unique_ptr<SdmProcessor> createSdmProcessor(const std::string& name) {
	auto& registry = sdmRegistry();
	auto it = registry.find(name);
	if (it == registry.end()) {
		std::cerr << "[SDM] Unknown processor '" << name << "'; using 'zscore'. Registered:";
		for (const auto& kv : registry)
			std::cerr << " " << kv.first;
		std::cerr << std::endl;
		it = registry.find("zscore");
		if (it == registry.end())
			return nullptr;
	}
	return it->second();
}

std::vector<std::string> registeredSdmProcessors() {
	std::vector<std::string> names;
	for (const auto& kv : sdmRegistry())
		names.push_back(kv.first);
	return names;
}
