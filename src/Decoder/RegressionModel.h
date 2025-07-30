#ifndef REGRESSIONMODEL_H_
#define REGRESSIONMODEL_H_

#include "BaseModel.h"

class RegressionModel : public BaseModel {
public:
	RegressionModel();
	~RegressionModel();

	void init(const std::string spikeFileName, const std::string workFolderPathName) override;

	std::vector<double> predict(std::map<long, double> &data, int16_t label, int16_t &predictLabel) override;

protected:
	// Model Parameters
	std::vector<double> theta;

	// Dataset constructed for MLPack use
	std::vector<std::vector<double>> X_train;
	std::vector<std::vector<double>> X_test;
	std::vector<int16_t> y_train;
	std::vector<int16_t> y_test;

	// Needed for training
	const char *error_msg;

	// Model Training functions
	void train(const char* input_file_name) override;
	void create_dataset(const char *scaledDataFileName);

	// Model Utils
	void splitData(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double splitRatio);

	// Logistic Regression Algorithm
	double sigmoid(double z);
	double hypothesis(const std::vector<double>& x);
	void gradientDescent(std::vector<std::vector<double>>& X, std::vector<int16_t>& y, double alpha, int iterations);
	int16_t predictOnce(const std::vector<double>& x);
	double calculateAccuracy(const std::vector<int16_t>& predictions, const std::vector<int16_t>& y_test);

};

#endif /* REGRESSIONMODEL_H_ */
