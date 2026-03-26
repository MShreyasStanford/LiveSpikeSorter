#include "LogRegSdmProcessor.h"
#include "../Helpers/FileWriter.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>

LogRegSdmProcessor::LogRegSdmProcessor() = default;
LogRegSdmProcessor::~LogRegSdmProcessor() = default;

double LogRegSdmProcessor::sigmoid(double z) {
	return 1.0 / (1.0 + std::exp(-z));
}

double LogRegSdmProcessor::dotProduct(const std::vector<double>& x) const {
	double result = 0.0;
	for (size_t i = 0; i < m_theta.size(); ++i)
		result += m_theta[i] * x[i];
	return result;
}

void LogRegSdmProcessor::gradientDescent(std::vector<std::vector<double>>& X,
                                         std::vector<int16_t>& y,
                                         double alpha, int iterations) {
	size_t m = X.size();
	size_t n = X[0].size();

	for (int iter = 0; iter < iterations; ++iter) {
		std::vector<double> gradient(n, 0.0);
		for (size_t i = 0; i < m; ++i) {
			double h = sigmoid(dotProduct(X[i]));
			double error = h - y[i];
			for (size_t j = 0; j < n; ++j)
				gradient[j] += error * X[i][j];
		}
		for (size_t j = 0; j < n; ++j)
			m_theta[j] -= (alpha / m) * gradient[j];
	}
}

void LogRegSdmProcessor::trainFromFiles(const std::string& spikesFile,
                                        const std::string& eventFile,
                                        const std::string& workFolder,
                                        int windowLength, int binLength, int windowOffset,
                                        const std::unordered_set<long>* channelFilter) {
	// Use a temporary DataBinner to produce binned training data
	DataBinner trainBinner(windowLength, binLength, windowOffset);
	std::string binnedFile = workFolder + "sdm_binnedSpikes.txt";
	FileWriter fw;
	fw.FileLoad(binnedFile);

	trainBinner.readInSpikes(spikesFile.c_str(), eventFile.c_str(), binnedFile.c_str(), &fw, channelFilter);
	fw.FileClose();

	// Parse the libsvm-format binned file to build X and y
	std::ifstream fin(binnedFile);
	if (!fin.is_open()) {
		std::cerr << "LogRegSdmProcessor: cannot open binned file " << binnedFile << std::endl;
		return;
	}

	// First pass: collect all channel indices and count samples
	std::set<long> allChannels;
	std::vector<std::string> lines;
	{
		std::string line;
		while (std::getline(fin, line)) {
			if (line.empty()) continue;
			lines.push_back(line);

			// Parse channel indices from "label ch:val ch:val ..."
			std::istringstream iss(line);
			std::string token;
			iss >> token; // skip label
			while (iss >> token) {
				auto colonPos = token.find(':');
				if (colonPos != std::string::npos) {
					long ch = std::stol(token.substr(0, colonPos));
					allChannels.insert(ch);
				}
			}
		}
	}
	fin.close();

	if (lines.empty()) {
		std::cerr << "LogRegSdmProcessor: no training data found." << std::endl;
		return;
	}

	// Build ordered channel list (bias term at index 0, then channels)
	m_channelOrder.assign(allChannels.begin(), allChannels.end());
	const size_t nFeatures = m_channelOrder.size() + 1; // +1 for bias

	// Build channel-to-feature-index map
	std::map<long, size_t> chToIdx;
	for (size_t i = 0; i < m_channelOrder.size(); ++i)
		chToIdx[m_channelOrder[i]] = i + 1; // index 0 is bias

	// Second pass: build X and y
	std::vector<std::vector<double>> X(lines.size(), std::vector<double>(nFeatures, 0.0));
	std::vector<int16_t> y(lines.size(), 0);

	for (size_t row = 0; row < lines.size(); ++row) {
		X[row][0] = 1.0; // bias term
		std::istringstream iss(lines[row]);
		std::string token;
		iss >> token; // label (may have leading '+')
		if (!token.empty() && token[0] == '+')
			token = token.substr(1);
		y[row] = static_cast<int16_t>(std::stoi(token));

		while (iss >> token) {
			auto colonPos = token.find(':');
			if (colonPos != std::string::npos) {
				long ch = std::stol(token.substr(0, colonPos));
				double val = std::stod(token.substr(colonPos + 1));
				auto it = chToIdx.find(ch);
				if (it != chToIdx.end())
					X[row][it->second] = val;
			}
		}
	}

	// Compute per-feature min/max for scaling (skip bias at index 0)
	m_featureMin.assign(nFeatures, 0.0);
	m_featureMax.assign(nFeatures, 1.0);
	for (size_t j = 1; j < nFeatures; ++j) {
		double fmin = X[0][j], fmax = X[0][j];
		for (size_t i = 1; i < X.size(); ++i) {
			if (X[i][j] < fmin) fmin = X[i][j];
			if (X[i][j] > fmax) fmax = X[i][j];
		}
		m_featureMin[j] = fmin;
		m_featureMax[j] = fmax;
	}

	// Scale features to [0, 1]
	for (size_t i = 0; i < X.size(); ++i) {
		for (size_t j = 1; j < nFeatures; ++j) {
			double range = m_featureMax[j] - m_featureMin[j];
			if (range > 0.0)
				X[i][j] = (X[i][j] - m_featureMin[j]) / range;
			else
				X[i][j] = 0.0;
		}
	}

	// Train/test split (80/20)
	size_t trainSize = static_cast<size_t>(X.size() * 0.8);
	std::vector<size_t> indices(X.size());
	for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
	std::shuffle(indices.begin(), indices.end(), std::default_random_engine(42));

	std::vector<std::vector<double>> X_train, X_test;
	std::vector<int16_t> y_train, y_test;
	for (size_t i = 0; i < indices.size(); ++i) {
		if (i < trainSize) {
			X_train.push_back(X[indices[i]]);
			y_train.push_back(y[indices[i]]);
		} else {
			X_test.push_back(X[indices[i]]);
			y_test.push_back(y[indices[i]]);
		}
	}

	// Initialize weights and train
	m_theta.assign(nFeatures, 0.0);
	std::cout << "LogRegSdmProcessor: Training on " << X_train.size()
	          << " samples, " << nFeatures << " features..." << std::endl;

	gradientDescent(X_train, y_train, 0.1, 10000);

	// Print accuracy on test set
	if (!X_test.empty()) {
		size_t correct = 0;
		for (size_t i = 0; i < X_test.size(); ++i) {
			double p = sigmoid(dotProduct(X_test[i]));
			int16_t pred = (p >= 0.5) ? 1 : 0;
			if (pred == y_test[i]) ++correct;
		}
		double accuracy = static_cast<double>(correct) / X_test.size();
		std::cout << "LogRegSdmProcessor: Test accuracy = " << accuracy
		          << " (" << correct << "/" << X_test.size() << ")" << std::endl;
	}

	std::cout << "LogRegSdmProcessor: Training complete. Weights: ";
	for (size_t i = 0; i < std::min<size_t>(m_theta.size(), 10); ++i)
		std::cout << m_theta[i] << " ";
	if (m_theta.size() > 10) std::cout << "...";
	std::cout << std::endl;
}

void LogRegSdmProcessor::init(const InputParameters& params,
                              const std::vector<long>& activitySubset) {
	// Build subset channel set
	m_subsetChannels.clear();
	for (auto ch : activitySubset)
		m_subsetChannels.insert(ch);

	const std::unordered_set<long>* filter = m_subsetChannels.empty() ? nullptr : &m_subsetChannels;

	// Determine window parameters for the SDM decoder binner
	// sdmDecoderWindowMs controls how many ms of spikes feed the feature vector
	int windowMs = params.sdmDecoderWindowMs;
	int samplesPerMs = static_cast<int>(params.fImecSamplingRate / 1000.0f);
	int windowLength = windowMs * samplesPerMs;
	int binLength = params.sdmTriggerBinMs * samplesPerMs;
	int windowOffset = windowLength; // no offset for SDM (predict at current time)

	// Train from offline data files
	if (!params.sSdmSpikesFile.empty() && !params.sSdmEventFile.empty()) {
		std::string workFolder = params.sSdmDecoderWorkFolder.empty()
		                         ? params.sDecoderWorkFolder
		                         : params.sSdmDecoderWorkFolder;
		trainFromFiles(params.sSdmSpikesFile, params.sSdmEventFile, workFolder,
		               windowLength, binLength, windowOffset, filter);
	} else {
		std::cerr << "LogRegSdmProcessor: No training files specified. "
		          << "Use --sdm_spikes_file and --sdm_event_file." << std::endl;
	}

	// Create runtime binner for streaming spike accumulation
	m_binner = std::make_unique<DataBinner>(windowLength, binLength, windowOffset);
}

void LogRegSdmProcessor::onSpikes(const std::vector<long>& times,
                                  const std::vector<long>& channels,
                                  long streamSampleCt) {
	if (m_binner) {
		m_binner->insert(
			const_cast<std::vector<long>&>(times),
			const_cast<std::vector<long>&>(channels));
		m_binner->updateTime(streamSampleCt);
	}
}

float LogRegSdmProcessor::computeBinValue(long /*binEndSampleCt*/, int8_t& direction) {
	if (m_theta.empty() || !m_binner) {
		direction = 0;
		return 0.0f;
	}

	// Get current data window from binner
	std::map<long, double> window = m_binner->getDataWindow();

	// Build feature vector: [bias, ch0_count, ch1_count, ...]
	std::vector<double> x(m_theta.size(), 0.0);
	x[0] = 1.0; // bias

	for (size_t i = 0; i < m_channelOrder.size(); ++i) {
		size_t featureIdx = i + 1;
		auto it = window.find(m_channelOrder[i]);
		double val = (it != window.end()) ? it->second : 0.0;

		// Apply same scaling as training
		if (featureIdx < m_featureMin.size()) {
			double range = m_featureMax[featureIdx] - m_featureMin[featureIdx];
			if (range > 0.0)
				val = (val - m_featureMin[featureIdx]) / range;
			else
				val = 0.0;
		}
		x[featureIdx] = val;
	}

	double prob = sigmoid(dotProduct(x));
	direction = (prob > 0.5) ? 1 : 0;
	return static_cast<float>(prob);
}
