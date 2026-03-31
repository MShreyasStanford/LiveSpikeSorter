#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct LatencySnapshot {
	float latestProcessMs{ 0.0f };
	float latestBudgetMs{ 0.0f };
	float latestRatio{ 0.0f };
	float meanProcessMs{ 0.0f };
	float p95ProcessMs{ 0.0f };
	float maxProcessMs{ 0.0f };
	float meanRatio{ 0.0f };
	float p95Ratio{ 0.0f };
	float maxRatio{ 0.0f };
	float skipRatePerMin{ 0.0f };
	float meanSpikeYield{ 0.0f };
	int consecutiveHighRatio{ 0 };
	long totalSkips{ 0 };
};

class LatencyMonitor {
public:
	explicit LatencyMonitor(std::size_t windowBatches = 100, float highRatioThreshold = 0.9f)
		: m_windowBatches(windowBatches)
		, m_highRatioThreshold(highRatioThreshold) {}

	void recordStage(const std::string& stageName, long long durationUs) {
		if (stageName.empty()) {
			return;
		}

		auto& stageHistory = m_stageDurationsMs[stageName];
		stageHistory.push_back(static_cast<float>(durationUs) / 1000.0f);
		trim(stageHistory);
	}

	void onBatch(long processMs, float acquisitionBudgetMs, bool skipped, long spikeYield) {
		m_latestProcessMs = static_cast<float>(processMs);
		m_latestBudgetMs = acquisitionBudgetMs;
		m_latestRatio = (acquisitionBudgetMs > 0.0f) ? (m_latestProcessMs / acquisitionBudgetMs) : 0.0f;
		m_latestSpikeYield = static_cast<float>(spikeYield);

		m_processTimesMs.push_back(m_latestProcessMs);
		m_ratios.push_back(m_latestRatio);
		m_budgetTimesMs.push_back(std::max(0.0f, acquisitionBudgetMs));
		m_spikeYields.push_back(m_latestSpikeYield);
		m_skipFlags.push_back(skipped ? 1 : 0);
		trim(m_processTimesMs);
		trim(m_ratios);
		trim(m_budgetTimesMs);
		trim(m_spikeYields);
		trim(m_skipFlags);

		m_totalSkips += skipped ? 1 : 0;
		m_consecutiveHighRatio = (m_latestRatio > m_highRatioThreshold) ? (m_consecutiveHighRatio + 1) : 0;
	}

	LatencySnapshot getSnapshot() const {
		LatencySnapshot out;
		out.latestProcessMs = m_latestProcessMs;
		out.latestBudgetMs = m_latestBudgetMs;
		out.latestRatio = m_latestRatio;
		out.meanProcessMs = mean(m_processTimesMs);
		out.p95ProcessMs = percentile(m_processTimesMs, 0.95f);
		out.maxProcessMs = maxValue(m_processTimesMs);
		out.meanRatio = mean(m_ratios);
		out.p95Ratio = percentile(m_ratios, 0.95f);
		out.maxRatio = maxValue(m_ratios);
		out.meanSpikeYield = mean(m_spikeYields);
		out.consecutiveHighRatio = m_consecutiveHighRatio;
		out.totalSkips = m_totalSkips;

		float windowDurationMs = std::accumulate(m_budgetTimesMs.begin(), m_budgetTimesMs.end(), 0.0f);
		int skippedInWindow = std::accumulate(m_skipFlags.begin(), m_skipFlags.end(), 0);
		out.skipRatePerMin = (windowDurationMs > 0.0f) ? ((static_cast<float>(skippedInWindow) * 60000.0f) / windowDurationMs) : 0.0f;
		return out;
	}

	std::vector<std::pair<std::string, float>> topStageMeansMs(std::size_t topN) const {
		std::vector<std::pair<std::string, float>> stageMeans;
		stageMeans.reserve(m_stageDurationsMs.size());
		for (const auto& kv : m_stageDurationsMs) {
			stageMeans.push_back({ kv.first, mean(kv.second) });
		}

		std::sort(stageMeans.begin(), stageMeans.end(),
			[](const auto& a, const auto& b) { return a.second > b.second; });

		if (stageMeans.size() > topN) {
			stageMeans.resize(topN);
		}
		return stageMeans;
	}

private:
	template <typename T>
	void trim(std::deque<T>& values) const {
		while (values.size() > m_windowBatches) {
			values.pop_front();
		}
	}

	static float mean(const std::deque<float>& values) {
		if (values.empty()) {
			return 0.0f;
		}
		float sum = std::accumulate(values.begin(), values.end(), 0.0f);
		return sum / static_cast<float>(values.size());
	}

	static float maxValue(const std::deque<float>& values) {
		if (values.empty()) {
			return 0.0f;
		}
		return *std::max_element(values.begin(), values.end());
	}

	static float percentile(const std::deque<float>& values, float q) {
		if (values.empty()) {
			return 0.0f;
		}

		std::vector<float> sorted(values.begin(), values.end());
		std::sort(sorted.begin(), sorted.end());

		const float clampedQ = std::min(1.0f, std::max(0.0f, q));
		const float idxF = clampedQ * static_cast<float>(sorted.size() - 1);
		const std::size_t idx = static_cast<std::size_t>(idxF);
		return sorted[idx];
	}

	std::size_t m_windowBatches;
	float m_highRatioThreshold;
	int m_consecutiveHighRatio{ 0 };
	long m_totalSkips{ 0 };
	float m_latestProcessMs{ 0.0f };
	float m_latestBudgetMs{ 0.0f };
	float m_latestRatio{ 0.0f };
	float m_latestSpikeYield{ 0.0f };

	std::deque<float> m_processTimesMs;
	std::deque<float> m_ratios;
	std::deque<float> m_budgetTimesMs;
	std::deque<float> m_spikeYields;
	std::deque<int> m_skipFlags;
	std::unordered_map<std::string, std::deque<float>> m_stageDurationsMs;
};
