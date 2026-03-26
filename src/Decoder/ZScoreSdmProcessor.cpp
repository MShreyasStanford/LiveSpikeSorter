#include "ZScoreSdmProcessor.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

ZScoreSdmProcessor::ZScoreSdmProcessor()
	: m_triggerZ(1.0f)
	, m_baselineMinSeconds(10.0f)
	, m_samplingRateHz(30000.0f)
	, m_binMs(50)
	, m_binSamples(1500)
	, m_baselineBinsSeen(0)
	, m_baselineMean(0.0)
	, m_baselineM2(0.0)
{
}

ZScoreSdmProcessor::~ZScoreSdmProcessor() = default;

void ZScoreSdmProcessor::init(const InputParameters& params,
                              const std::vector<long>& /*activitySubset*/)
{
	m_triggerZ = params.sdmTriggerZ;
	m_baselineMinSeconds = params.sdmBaselineMinSeconds;
	m_samplingRateHz = params.fImecSamplingRate;
	m_binMs = params.sdmTriggerBinMs;
	m_binSamples = std::max<long>(1, static_cast<long>(std::llround(
		(static_cast<double>(m_binMs) / 1000.0) * static_cast<double>(m_samplingRateHz))));

	m_baselineBinsSeen = 0;
	m_baselineMean = 0.0;
	m_baselineM2 = 0.0;
	m_binCounts.clear();
	m_baselineFrozen.store(false);

	// Launch baseline input thread (detached, same as original)
	std::thread baselineThread([this]() {
		std::cout << "SDM baseline input required. Enter: <mean> <sd> (counts per bin). Example: 4.2 1.1" << std::endl;
		std::cout << "Type 'help' to reprint this message." << std::endl;
		while (true) {
			std::cout << "SDM baseline> " << std::flush;
			std::string line;
			if (!std::getline(std::cin, line))
				return;

			if (line == "help" || line == "h" || line == "?") {
				std::cout << "Enter: <mean> <sd> (counts per bin). Example: 4.2 1.1" << std::endl;
				continue;
			}

			std::istringstream iss(line);
			double mean = 0.0;
			double sd = 0.0;
			if (!(iss >> mean >> sd)) {
				std::cout << "Invalid input. Expected: <mean> <sd>." << std::endl;
				continue;
			}
			if (sd <= 0.0) {
				std::cout << "Invalid SD. Must be > 0." << std::endl;
				continue;
			}

			m_manualMean.store(mean);
			m_manualSd.store(sd);
			m_baselineFrozen.store(true);

			const double threshold = mean + static_cast<double>(m_triggerZ) * sd;
			std::cout << "Baseline set: mean=" << mean << " sd=" << sd
			          << " z=" << m_triggerZ << " -> threshold=" << threshold
			          << " counts/bin" << std::endl;
			return;
		}
	});
	baselineThread.detach();
}

void ZScoreSdmProcessor::onSpikes(const std::vector<long>& times,
                                  const std::vector<long>& /*channels*/,
                                  long /*streamSampleCt*/)
{
	// Bin each spike by its time
	for (size_t i = 0; i < times.size(); ++i) {
		const long binIdx = times[i] / m_binSamples;
		m_binCounts[binIdx]++;
	}
}

float ZScoreSdmProcessor::computeBinValue(long binEndSampleCt, int8_t& direction)
{
	// The completed bin index
	const long binIdx = (binEndSampleCt / m_binSamples) - 1;

	// Extract and remove the count for this bin
	long count = 0;
	auto it = m_binCounts.find(binIdx);
	if (it != m_binCounts.end()) {
		count = it->second;
		m_binCounts.erase(it);
	}

	if (!m_baselineFrozen.load()) {
		// Welford's online algorithm for running mean/variance
		m_baselineBinsSeen++;
		const double x = static_cast<double>(count);
		const double delta = x - m_baselineMean;
		m_baselineMean += delta / static_cast<double>(m_baselineBinsSeen);
		const double delta2 = x - m_baselineMean;
		m_baselineM2 += delta * delta2;

		direction = 0;
		return 0.0f;
	}

	const double mean = m_manualMean.load();
	const double sd = m_manualSd.load();
	const double zThresh = static_cast<double>(m_triggerZ);

	if (sd <= 0.0) {
		direction = 0;
		return 0.0f;
	}

	const double zVal = (static_cast<double>(count) - mean) / sd;
	direction = 0;
	if (zVal > zThresh)
		direction = 1;
	else if (zVal < -zThresh)
		direction = -1;

	return static_cast<float>(zVal);
}
