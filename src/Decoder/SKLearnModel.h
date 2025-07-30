#ifndef _SK_LEARN_MODEL_
#define _SK_LEARN_MODEL_
#include <Python.h>
#include <vector>
#include <map>
#include "BaseModel.h"
#include <mutex>

class SKLearnModel : public BaseModel {
public:
	enum ModelType {
		LinearRegression,
		SVM,
		RandomForest,
		KNeighbors,
		GaussianNB,
		LogisticRegression,
		DecisionTree,
		AdaBoost,
		GradientBoosting,
		Ridge,
		Lasso,
		ElasticNet,
		KMeans,
		DBSCAN,
		SGDClassifier
	};

private:
	// PyObject corresponding to the scikit-learn model
	PyObject* pyModel;

	// Enum of model types
	ModelType modelType;

	PyObject* scalerInstance;

	// Testing and training data
	std::vector<std::vector<double>> X_train;
	std::vector<std::vector<double>> X_test;
	std::vector<double> y_train;
	std::vector<double> y_test;
	
	// Mutex to avoid race conditions initializing twice
	std::mutex initMutex;

public:
	SKLearnModel(ModelType type);
	~SKLearnModel();

	void init(const std::string spikeFileName, const std::string workFolderPathName);

	// train spike + trial data in file
	void train(const char* input_file_name) override;

	// the usual train with samples and labels
	void train(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
	void create_dataset(const char *scaledDataFileName);

	// split data for training
	void splitData(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double splitRatio);

	// predict with binned data
	std::vector<double> predict(std::map<long, double> &data, int16_t label, int16_t &predictLabel) override;

	// the usual predict with just a vector as input
	std::vector<double> predict(const std::vector<std::vector<double>>& X);
	void loadModel(const std::string& path);
	void saveModel(const std::string& path);

private:
	void initializePython();
	int initializeNumpy();
	void finalizePython();
	PyObject* toNpArray(const std::vector<double>& data);
	PyObject* toNpArray(const std::vector<std::vector<double>>& data);
	PyObject* matToPyObj(const std::vector<std::vector<double>>& data);
	PyObject* vecToPyObj(const std::vector<double>& data);
};
#endif