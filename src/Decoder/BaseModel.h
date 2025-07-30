#ifndef BaseModel_H_
#define BaseModel_H_

#include <vector>
#include <map>

#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))
#define LOWER 0.0 // Since feature values are non-negative and sparse, use 0 rather than -1 as lowerbound
#define UPPER 1.0
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class BaseModel {
public:
	BaseModel();
	~BaseModel();

	virtual void init(const std::string spikeFileName, const std::string workFolderPathName);


	virtual std::vector<double> predict(std::map<long, double> &data, int16_t label, int16_t &predictLabel) = 0;

protected:
	//struct svm_model *model;
	int predict_probability;

	//// Needed for training
	//const char *error_msg;
	//struct svm_problem prob;
	//struct svm_parameter param;
	//struct svm_node *x_space;


	// Vectors for decoder functionality
	std::vector<double> featureMaxs;
	std::vector<double> featureMins;
	std::vector<double> probEstimates;
	std::vector<int>	labels;


	// General helper function
	char* readline(FILE *input, char **line, int *max_line_len);

	// Data Scaling Functions
	void computeScaleParams(const char* inputFileName);
	void scaleFileData(const char* inputFileName, const char* spikeFileName, double lower, double upper, bool y_scaling, double y_lower, double y_upper);
	void scaleData(std::map<long, double> &data, double lower, double upper);
	void output(FILE *fpOut, int index, double value, double lower, double upper);
	void output_target(FILE *fpOut, double value, bool y_scaling, double y_lower, double y_upper, double y_min, double y_max);

	//// Model Training functions
	virtual void train(const char* input_file_name) = 0;
	//void read_problem(const char *filename);
	//float do_cross_validation(struct svm_problem &prob, struct svm_parameter &param);

	// Model Prediction Functions
	//void svmPredict(std::map<long, double> &data, int16_t label, int16_t &predictLabel);
};

#endif /* BaseModel_H_ */