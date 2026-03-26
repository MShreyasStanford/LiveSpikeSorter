#include "BinCountSdmProcessor.h"
#include "../Networking/Sock.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

BinCountSdmProcessor::BinCountSdmProcessor()
	: m_windowSamples(1500)
	, m_useAllTemplates(false)
{
}

BinCountSdmProcessor::~BinCountSdmProcessor() = default;

void BinCountSdmProcessor::init(const InputParameters& params,
                                const std::vector<long>& activitySubset)
{
	m_useAllTemplates = activitySubset.empty();

	if (m_useAllTemplates) {
		// Temporarily empty — will be filled by setNumTemplates() once the
		// sorter reports the real template count loaded from disk.
		m_channels.clear();
	} else {
		m_channels = activitySubset;
	}

	// Sort so channel order is deterministic
	std::sort(m_channels.begin(), m_channels.end());

	m_channelIndex.clear();
	for (int i = 0; i < static_cast<int>(m_channels.size()); ++i) {
		m_channelIndex[m_channels[i]] = i;
	}

	m_windowSamples = std::max<long>(1, static_cast<long>(std::llround(
		(static_cast<double>(params.sdmTriggerBinMs) / 1000.0) *
		static_cast<double>(params.fImecSamplingRate))));

	m_spikeEvents.clear();

	if (m_useAllTemplates) {
		std::cout << "BinCountSdmProcessor: waiting for sorter to report template count, "
		          << "windowSamples=" << m_windowSamples << std::endl;
	} else {
		std::cout << "BinCountSdmProcessor: " << m_channels.size()
		          << " channels, windowSamples=" << m_windowSamples << std::endl;
	}
}

void BinCountSdmProcessor::rebuildChannelList(long numTemplates)
{
	m_channels.resize(numTemplates);
	for (long i = 0; i < numTemplates; ++i)
		m_channels[i] = i;

	std::sort(m_channels.begin(), m_channels.end());

	m_channelIndex.clear();
	for (int i = 0; i < static_cast<int>(m_channels.size()); ++i) {
		m_channelIndex[m_channels[i]] = i;
	}
}

void BinCountSdmProcessor::setNumTemplates(long numTemplates)
{
	if (!m_useAllTemplates)
		return; // user specified an explicit subset — ignore

	rebuildChannelList(numTemplates);
	std::cout << "BinCountSdmProcessor: using all " << numTemplates
	          << " templates from sorter" << std::endl;
}

void BinCountSdmProcessor::onSpikes(const std::vector<long>& times,
                                    const std::vector<long>& channels,
                                    long /*streamSampleCt*/)
{
	for (size_t i = 0; i < times.size(); ++i) {
		auto it = m_channelIndex.find(channels[i]);
		if (it != m_channelIndex.end()) {
			m_spikeEvents.emplace_back(times[i], it->second);
		}
	}
}

float BinCountSdmProcessor::computeBinValue(long /*binEndSampleCt*/, int8_t& direction)
{
	// Stub — onBatchComplete is the real output path for bincounts
	direction = 0;
	return 0.0f;
}

void BinCountSdmProcessor::sendHello(Sock& sdmSock)
{
	const uint16_t numChannels = static_cast<uint16_t>(m_channels.size());
	const size_t pktSize = 8 + 2 + 4 * static_cast<size_t>(numChannels);
	std::vector<uint8_t> buf(pktSize);

	// sampleCt = 0 (marker for hello)
	const uint64_t zero = 0;
	std::memcpy(&buf[0], &zero, sizeof(uint64_t));

	// numChannels
	std::memcpy(&buf[8], &numChannels, sizeof(uint16_t));

	// channel IDs as int32
	for (int i = 0; i < static_cast<int>(numChannels); ++i) {
		const int32_t chId = static_cast<int32_t>(m_channels[i]);
		std::memcpy(&buf[10 + 4 * i], &chId, sizeof(int32_t));
	}

	const uint sent = sdmSock.sendData(buf.data(), static_cast<uint>(buf.size()));
	if (sent == 0) {
		std::cerr << "BinCountSdmProcessor: hello send failed: " << sdmSock.errorReason() << std::endl;
	} else {
		std::cout << "BinCountSdmProcessor: hello sent (" << sent << " bytes, "
		          << numChannels << " channels)." << std::endl;
	}
}

void BinCountSdmProcessor::sendPacket(Sock& /*sdmSock*/, uint64_t /*glxSampleCt*/, long /*binEndSampleCt*/)
{
	// No-op — bin-boundary calls still fire from Decoder but are harmless.
	// The real sending happens in onBatchComplete().
}

void BinCountSdmProcessor::onBatchComplete(Sock& sdmSock, uint64_t glxSampleCt, long streamSampleCt)
{
	// Don't send until we know the channel count
	if (m_channels.empty())
		return;

	// 1. Trim spikes older than the trailing window
	const long windowStart = streamSampleCt - m_windowSamples;
	while (!m_spikeEvents.empty() && m_spikeEvents.front().first < windowStart) {
		m_spikeEvents.pop_front();
	}

	// 2. Count per-channel spikes in the window
	const uint16_t numChannels = static_cast<uint16_t>(m_channels.size());
	std::vector<float> counts(numChannels, 0.0f);
	for (const auto& ev : m_spikeEvents) {
		counts[ev.second] += 1.0f;
	}

	// 3. Build and send packet: [uint64 sampleCt][uint16 N][float x N]
	const size_t pktSize = 8 + 2 + 4 * static_cast<size_t>(numChannels);
	std::vector<uint8_t> buf(pktSize);

	std::memcpy(&buf[0], &glxSampleCt, sizeof(uint64_t));
	std::memcpy(&buf[8], &numChannels, sizeof(uint16_t));
	for (int i = 0; i < static_cast<int>(numChannels); ++i) {
		std::memcpy(&buf[10 + 4 * i], &counts[i], sizeof(float));
	}

	sdmSock.sendData(buf.data(), static_cast<uint>(buf.size()));
}
