#ifndef SVMMODEL_H_
#define SVMMODEL_H_

#include "../External/libsvm/svm.h"
#include "BaseModel.h"


class SVMModel : public BaseModel {
public:
	SVMModel();
	~SVMModel();

	void init(const std::string spikeFileName, const std::string workFolderPathName) override;


	std::vector<double> predict(std::map<long, double> &data, int16_t label, int16_t &predictLabel) override;

protected:
	struct svm_model *model;
	int predict_probability;

	// Needed for training
	const char *error_msg;
	struct svm_problem prob;
	struct svm_parameter param;
	struct svm_node *x_space;

	// Model Training functions
	void train(const char* input_file_name) override;
	void read_problem(const char *filename);
	float do_cross_validation(struct svm_problem &prob, struct svm_parameter &param);

	// Model Prediction Functions
	void svmPredict(std::map<long, double> &data, int16_t label, int16_t &predictLabel);

};

#endif /* SVMMODEL_H_ */
