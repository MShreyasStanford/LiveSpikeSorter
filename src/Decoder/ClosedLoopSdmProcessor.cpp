#include "ClosedLoopSdmProcessor.h"
#include "../Networking/Sock.h"

#include <cstring>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

ClosedLoopSdmProcessor::ClosedLoopSdmProcessor()
	: m_mode(MODE_MEDIAN)
	, m_fsOffLow(0.0f), m_fsOffHigh(0.0f), m_rsOffLow(0.0f), m_rsOffHigh(0.0f)
	, m_fsZLow(1.0f), m_fsZHigh(1.0f), m_rsZLow(1.0f), m_rsZHigh(1.0f)
	, m_samplingRateHz(30000.0f)
	, m_binMs(100)
	, m_binSamples(3000)
	, m_fsN(0)
	, m_rsN(0)
	, m_fsMedian(0.0), m_rsMedian(0.0)
	, m_fsMean(0.0), m_fsSd(1.0)
	, m_rsMean(0.0), m_rsSd(1.0)
	, m_statsLoaded(false)
{
}

ClosedLoopSdmProcessor::~ClosedLoopSdmProcessor() = default;

void ClosedLoopSdmProcessor::init(const InputParameters& params,
                                  const std::vector<long>& /*activitySubset*/)
{
	SdmParams p(params.mapSdmParams);
	m_mode = (p.getString("mode", "median") == "zscore") ? MODE_ZSCORE : MODE_MEDIAN;

	// Per-population/per-direction threshold (4 conditions for FS/RS). Each falls
	// back to the symmetric "offset" / "trigger_z" when not given.
	const float offset = p.getFloat("offset", 0.0f);
	const float triggerZ = p.getFloat("trigger_z", 1.0f);
	m_fsOffLow  = p.getFloat("fs_offset_low",  offset);
	m_fsOffHigh = p.getFloat("fs_offset_high", offset);
	m_rsOffLow  = p.getFloat("rs_offset_low",  offset);
	m_rsOffHigh = p.getFloat("rs_offset_high", offset);
	m_fsZLow    = p.getFloat("fs_z_low",  triggerZ);
	m_fsZHigh   = p.getFloat("fs_z_high", triggerZ);
	m_rsZLow    = p.getFloat("rs_z_low",  triggerZ);
	m_rsZHigh   = p.getFloat("rs_z_high", triggerZ);
	const std::string rsFsPath = p.getString("rs_fs_path");
	const std::string statsPath = p.getString("stats_path");

	m_samplingRateHz = params.fImecSamplingRate;
	m_binMs = params.sdmTriggerBinMs;
	m_binSamples = std::max<long>(1, static_cast<long>(std::llround(
		(static_cast<double>(m_binMs) / 1000.0) * static_cast<double>(m_samplingRateHz))));

	m_fsBinCounts.clear();
	m_rsBinCounts.clear();

	if (!loadLabels(rsFsPath)) {
		std::cerr << "[ClosedLoop] ERROR: could not read RS/FS labels from '"
		          << rsFsPath << "'. FS/RS states will all be 0." << std::endl;
	}
	if (!loadStats(statsPath)) {
		std::cerr << "[ClosedLoop] ERROR: could not read stats from '"
		          << statsPath << "'. FS/RS states will all be 0 until provided."
		          << std::endl;
	}

	std::cout << "[ClosedLoop] mode=" << (m_mode == MODE_ZSCORE ? "zscore" : "median")
	          << " binMs=" << m_binMs << std::endl;
	std::cout << "[ClosedLoop] median offsets  FS low/high=" << m_fsOffLow << "/" << m_fsOffHigh
	          << "  RS low/high=" << m_rsOffLow << "/" << m_rsOffHigh << std::endl;
	std::cout << "[ClosedLoop] zscore triggers  FS low/high=" << m_fsZLow << "/" << m_fsZHigh
	          << "  RS low/high=" << m_rsZLow << "/" << m_rsZHigh << std::endl;
	std::cout << "[ClosedLoop] FS n=" << m_fsN << " median=" << m_fsMedian
	          << " mean=" << m_fsMean << " sd=" << m_fsSd << std::endl;
	std::cout << "[ClosedLoop] RS n=" << m_rsN << " median=" << m_rsMedian
	          << " mean=" << m_rsMean << " sd=" << m_rsSd << std::endl;
}

bool ClosedLoopSdmProcessor::loadLabels(const std::string& path)
{
	if (path.empty())
		return false;
	std::ifstream in(path);
	if (!in.is_open())
		return false;

	m_fsIds.clear();
	m_rsIds.clear();
	std::string line;
	bool first = true;
	while (std::getline(in, line)) {
		if (line.empty())
			continue;
		if (first) {                    // skip header row if present
			first = false;
			if (line.find("template_index") != std::string::npos)
				continue;
		}
		std::stringstream ss(line);
		std::string idStr, label;
		if (!std::getline(ss, idStr, ','))
			continue;
		if (!std::getline(ss, label, ','))
			continue;
		// trim + upcase label
		label.erase(0, label.find_first_not_of(" \t\r\n"));
		label.erase(label.find_last_not_of(" \t\r\n") + 1);
		std::transform(label.begin(), label.end(), label.begin(), ::toupper);
		long tid = 0;
		try { tid = std::stol(idStr); } catch (...) { continue; }
		if (label == "FS")
			m_fsIds.insert(tid);
		else if (label == "RS")
			m_rsIds.insert(tid);
	}
	m_fsN = static_cast<int>(m_fsIds.size());
	m_rsN = static_cast<int>(m_rsIds.size());
	return (m_fsN + m_rsN) > 0;
}

bool ClosedLoopSdmProcessor::loadStats(const std::string& path)
{
	if (path.empty())
		return false;
	std::ifstream in(path);
	if (!in.is_open())
		return false;

	std::string key;
	double val = 0.0;
	while (in >> key >> val) {
		if      (key == "fs_median") m_fsMedian = val;
		else if (key == "rs_median") m_rsMedian = val;
		else if (key == "fs_mean")   m_fsMean = val;
		else if (key == "fs_sd")     m_fsSd = (val > 0.0) ? val : 1.0;
		else if (key == "rs_mean")   m_rsMean = val;
		else if (key == "rs_sd")     m_rsSd = (val > 0.0) ? val : 1.0;
		// bin_ms / n_bins / fs_n / rs_n are informational; bin size comes from params
	}
	m_statsLoaded = true;
	return true;
}

void ClosedLoopSdmProcessor::onSpikes(const std::vector<long>& times,
                                      const std::vector<long>& templates,
                                      long /*streamSampleCt*/)
{
	const size_t n = std::min<size_t>(times.size(), templates.size());
	for (size_t i = 0; i < n; ++i) {
		const long binIdx = times[i] / m_binSamples;
		const long tid = templates[i];
		if (m_fsIds.find(tid) != m_fsIds.end())
			m_fsBinCounts[binIdx]++;
		else if (m_rsIds.find(tid) != m_rsIds.end())
			m_rsBinCounts[binIdx]++;
	}
}

int ClosedLoopSdmProcessor::stateFor(double popValue, int nPop,
                                     double median, double mean, double sd,
                                     double offLow, double offHigh,
                                     double zLow, double zHigh) const
{
	if (nPop <= 0)
		return 0;
	if (m_mode == MODE_MEDIAN) {
		// This math means offlow should be positive
		if (popValue < median - offLow)  return -1;   // low
		if (popValue > median + offHigh) return +1;   // high
		return 0;
	}
	// zscore, independent negative/positive trigger magnitudes
	if (sd <= 0.0)
		return 0;
	const double z = (popValue - mean) / sd;
	if (z < -zLow)  return -1;
	if (z >  zHigh) return +1;
	return 0;
}

long ClosedLoopSdmProcessor::takeBinCount(std::unordered_map<long, long>& counts, long binIdx) const
{
	auto it = counts.find(binIdx);
	if (it == counts.end())
		return 0;
	const long c = it->second;
	counts.erase(it);
	return c;
}

float ClosedLoopSdmProcessor::computeBinValue(long binEndSampleCt, int8_t& direction)
{
	// Read-only
	const long binIdx = (binEndSampleCt / m_binSamples) - 1;
	auto it = m_fsBinCounts.find(binIdx);
	const long fsCount = (it != m_fsBinCounts.end()) ? it->second : 0;
	const double fsVal = (m_fsN > 0) ? static_cast<double>(fsCount) / m_fsN : 0.0;
	direction = static_cast<int8_t>(stateFor(fsVal, m_fsN, m_fsMedian, m_fsMean, m_fsSd,
	                                          m_fsOffLow, m_fsOffHigh, m_fsZLow, m_fsZHigh));
	if (m_mode == MODE_ZSCORE && m_fsSd > 0.0)
		return static_cast<float>((fsVal - m_fsMean) / m_fsSd);
	return static_cast<float>(fsVal);
}

void ClosedLoopSdmProcessor::sendPacket(Sock& sdmSock, uint64_t glxSampleCt, long binEndSampleCt)
{
	const long binIdx = (binEndSampleCt / m_binSamples) - 1;

	const long fsCount = takeBinCount(m_fsBinCounts, binIdx);
	const long rsCount = takeBinCount(m_rsBinCounts, binIdx);

	const double fsVal = (m_fsN > 0) ? static_cast<double>(fsCount) / m_fsN : 0.0;
	const double rsVal = (m_rsN > 0) ? static_cast<double>(rsCount) / m_rsN : 0.0;

	const int32_t fsState = stateFor(fsVal, m_fsN, m_fsMedian, m_fsMean, m_fsSd,
	                                 m_fsOffLow, m_fsOffHigh, m_fsZLow, m_fsZHigh);
	const int32_t rsState = stateFor(rsVal, m_rsN, m_rsMedian, m_rsMean, m_rsSd,
	                                 m_rsOffLow, m_rsOffHigh, m_rsZLow, m_rsZHigh);

	// 16 bytes LE: int32 FS | int32 RS | uint64 glxSampleCt  (matches et_trialNEW.m)
	uint8_t buf[16];
	std::memcpy(&buf[0], &fsState, sizeof(int32_t));
	std::memcpy(&buf[4], &rsState, sizeof(int32_t));
	std::memcpy(&buf[8], &glxSampleCt, sizeof(uint64_t));
	sdmSock.sendData(buf, static_cast<uint>(sizeof(buf)));

	// Diagnostic print bin statistics; indicate EARLY RELEASE if FS/RS are in opposite states (as per protocol)
	if (fsState != 0 || rsState != 0) {
		std::cout << "[ClosedLoop] bin=" << binIdx
		          << " FS=" << fsState << " (" << fsVal << ")"
		          << " RS=" << rsState << " (" << rsVal << ")"
		          << (((fsState == -1 && rsState == 1) || (fsState == 1 && rsState == -1))
		                  ? "  <-- EARLY RELEASE" : "")
		          << std::endl;
	}
}

REGISTER_SDM_PROCESSOR("closedloop", ClosedLoopSdmProcessor);
