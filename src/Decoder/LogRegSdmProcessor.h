#ifndef LOGREG_SDM_PROCESSOR_H_
#define LOGREG_SDM_PROCESSOR_H_

#include "SdmProcessor.h"
#include "dataBinner.h"
#include <vector>
#include <unordered_set>
#include <memory>

class LogRegSdmProcessor : public SdmProcessor {
public:
	LogRegSdmProcessor();
	~LogRegSdmProcessor() override;

	void init(const InputParameters& params,
	          const std::vector<long>& activitySubset) override;

	void onSpikes(const std::vector<long>& times,
	              const std::vector<long>& channels,
	              long streamSampleCt) override;

	float computeBinValue(long binEndSampleCt, int8_t& direction) override;

private:
	// Logistic regression helpers
	static double sigmoid(double z);
	double dotProduct(const std::vector<double>& x) const;

	// Training
	void trainFromFiles(const std::string& spikesFile,
	                    const std::string& eventFile,
	                    const std::string& workFolder,
	                    int windowLength, int binLength, int windowOffset,
	                    const std::unordered_set<long>* channelFilter);

	void gradientDescent(std::vector<std::vector<double>>& X,
	                     std::vector<int16_t>& y,
	                     double alpha, int iterations);

	// Model weights
	std::vector<double> m_theta;

	// Feature scaling params (per-feature min/max from training)
	std::vector<double> m_featureMin;
	std::vector<double> m_featureMax;

	// Mapping from channel index to feature index (sorted subset channels)
	std::vector<long> m_channelOrder;

	// Runtime spike accumulation
	std::unique_ptr<DataBinner> m_binner;
	std::unordered_set<long> m_subsetChannels;
};

#endif /* LOGREG_SDM_PROCESSOR_H_ */
