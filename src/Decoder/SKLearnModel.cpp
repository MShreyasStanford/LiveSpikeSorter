#include "SKLearnModel.h"
#include <iostream>
#include <fstream> // Temporary
#include <algorithm> // For std::shuffle
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include <random>
#include <atomic>
#include <numpy/arrayobject.h>

SKLearnModel::SKLearnModel(ModelType type) 
	: modelType(type)
	, pyModel(nullptr) 
{
	initializePython();
	PyGILState_STATE gil_state = PyGILState_Ensure();

	switch (type) {
	case LinearRegression:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.linear_model"), "LinearRegression", NULL);
		break;
	case SVM:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.svm"), "SVC", NULL);
		break;
	case RandomForest:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.ensemble"), "RandomForestClassifier", NULL);
		break;
	case KNeighbors:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.neighbors"), "KNeighborsClassifier", NULL);
		break;
	case GaussianNB:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.naive_bayes"), "GaussianNB", NULL);
		break;
	case LogisticRegression:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.linear_model"), "LogisticRegression", NULL);
		break;
	case DecisionTree:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.tree"), "DecisionTreeClassifier", NULL);
		break;
	case AdaBoost:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.ensemble"), "AdaBoostClassifier", NULL);
		break;
	case GradientBoosting:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.ensemble"), "GradientBoostingClassifier", NULL);
		break;
	case Ridge:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.linear_model"), "Ridge", NULL);
		break;
	case Lasso:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.linear_model"), "Lasso", NULL);
		break;
	case ElasticNet:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.linear_model"), "ElasticNet", NULL);
		break;
	case KMeans:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.cluster"), "KMeans", NULL);
		break;
	case DBSCAN:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.cluster"), "DBSCAN", NULL);
		break;
	case SGDClassifier:
		pyModel = PyObject_CallMethod(PyImport_ImportModule("sklearn.linear_model"), "SGDClassifier", NULL);
		break;
	default:
		PyErr_SetString(PyExc_RuntimeError, "Unsupported model type");
		PyErr_Print();
		break;
	}

	if (PyErr_Occurred()) PyErr_Print();
	if (!pyModel) throw std::runtime_error("Model not initialized.\n");
	PyGILState_Release(gil_state);
}

void SKLearnModel::init(const std::string spikeFileName, const std::string workFolderPathName) {
	BaseModel::init(spikeFileName, workFolderPathName);
}


void SKLearnModel::initializePython() {
	// Python interpreter + numpy + module imports only needs to be done once
	// per application lifetime
	static std::atomic<bool> isInitialized = false;

	if (isInitialized.load()) return;

	std::unique_lock<std::mutex> lock(initMutex);

	// initializes Python interpreter
	Py_Initialize();

	// creates Python GIL for Python <= 3.6, does nothing for other versions
	PyEval_InitThreads();

	// Import NumPy first
	PyObject* numpy = PyImport_ImportModule("numpy");
	if (!numpy) {
		PyErr_Print();
		std::cerr << "Failed to import NumPy." << std::endl;
		Py_DECREF(numpy);
		exit(EXIT_SUCCESS);
	}
	else {
		Py_DECREF(numpy);
		std::cout << "NumPy imported successfully." << std::endl;
	}

	// initialize numpy
	initializeNumpy();

	// Now safe to import other modules that depend on NumPy
	PyObject* sklearn = PyImport_ImportModule("sklearn");
	if (!sklearn) {
		PyErr_Print();
		std::cerr << "Failed to import scikit-learn." << std::endl;
		Py_DECREF(sklearn);
		exit(EXIT_SUCCESS);
	}
	else {
		Py_DECREF(sklearn);
		std::cout << "scikit-learn imported successfully." << std::endl;
	}

	isInitialized = true;
}

/*
	Refer to:
	https://stackoverflow.com/questions/52828873/how-does-import-array-in-numpy-c-api-work
*/
int SKLearnModel::initializeNumpy() {
	import_array1(0);
	return 0;
}

void SKLearnModel::finalizePython() {
	Py_Finalize();
}

SKLearnModel::~SKLearnModel() {
	PyGILState_STATE gil_state = PyGILState_Ensure();
	Py_DECREF(pyModel);
	PyGILState_Release(gil_state);
	finalizePython();
}

void SKLearnModel::train(const char* input_file_name) {
	static const char *ptLabel = { "Model::train" };

	create_dataset(input_file_name);

	size_t n_features = X_train[0].size();
	std::cout << "Scitkit-learn training...\n";
	train(X_train, y_train);
}

void SKLearnModel::train(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
	PyObject* pyX = toNpArray(X);
	PyObject* pyY = toNpArray(y);

	if (!pyX || !pyY) {
		std::cerr << "Failed to convert X or y to PyObject." << std::endl;
		if (pyX) Py_DECREF(pyX);  // Properly decrement reference count if not NULL
		if (pyY) Py_DECREF(pyY);  // Properly decrement reference count if not NULL
		return; // Early exit or handle the error as needed
	}


	if (modelType == LogisticRegression) {
		PyObject* scalerModule = PyImport_ImportModule("sklearn.preprocessing");
		PyObject* StandardScalerClass = PyObject_GetAttrString(scalerModule, "StandardScaler");
		PyObject* scalerInstance = PyObject_CallObject(StandardScalerClass, NULL);
		// Fit the scaler on training data (pyX_train should be created similarly as in predict)
		PyObject* fittedScaler = PyObject_CallMethod(scalerInstance, "fit", "O", pyX);
		// Save scalerInstance as a member variable for later use
		this->scalerInstance = scalerInstance; // make sure to handle ref counts appropriately

		Py_DECREF(fittedScaler);
		Py_DECREF(StandardScalerClass);
		Py_DECREF(scalerModule);
	}

	PyObject* result = PyObject_CallMethod(pyModel, "fit", "OO", pyX, pyY);
	if (!result) {
		PyErr_Print();
		std::cerr << "An error occurred while calling the fit method." << std::endl;
		Py_DECREF(pyX);
		Py_DECREF(pyY);
		exit(EXIT_SUCCESS);
	}

	// Evaluate the model accuracy on the training set
	PyObject* score = PyObject_CallMethod(pyModel, "score", "OO", pyX, pyY);
	if (!score) {
		PyErr_Print();
		std::cerr << "An error occurred while calling the score method." << std::endl;
	}
	else {
		double accuracy = PyFloat_AsDouble(score);
		if (PyErr_Occurred()) {
			PyErr_Print();
		}
		else {
			std::cout << "Model training accuracy: " << accuracy * 100 << "%" << std::endl;
		}
		Py_DECREF(score);
	}

	Py_DECREF(pyX);
	Py_DECREF(pyY);
	Py_DECREF(result);
}


void SKLearnModel::create_dataset(const char *scaledDataFileName)
{
	int max_index, feature_index, i;
	size_t len_data, j;
	FILE *fp = fopen(scaledDataFileName, "r");
	char *endptr;
	char *idx, *val, *label;

	std::cout << "Scikitlearn create_dataset()" << std::endl;
	if (fp == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", scaledDataFileName);
		exit(1);
	}

	len_data = 0;
	max_index = 0;

	// Run through dataset once to determine size
	int max_line_len = 1024;
	char* line = Malloc(char, max_line_len);
	while (readline(fp, &line, &max_line_len) != NULL)
	{
		char *p = strtok(line, " \t"); // label

		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");
			if (val == NULL)
				break;

			feature_index = (int)strtol(idx, &endptr, 10);
			if (feature_index > max_index)
				max_index = feature_index;
		}
		++len_data;
	}
	rewind(fp);

	// Initialize dataset objects
	std::vector<std::vector<double>> X(len_data, std::vector<double>(max_index + 1, 0.0));
	std::vector<double> y(len_data, 0.0);

	std::cout << "size of x: " << len_data << " " << max_index << std::endl;

	max_index = 0;

	for (i = 0; i < len_data; i++)
	{
		readline(fp, &line, &max_line_len);
		//prob.x[i] = &x_space[j];
		label = strtok(line, " \t\n");
		if (label == NULL) { // empty line
			fprintf(stderr, "Empty line at line %d\n", i + 1);
			exit(1);
		}

		y[i] = (int)strtol(label, &endptr, 10);
		if (endptr == label || *endptr != '\0') {
			fprintf(stderr, "No data at line %d\n", i);
			exit(1);
		}

		j = 0;

		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;

			feature_index = (int)strtol(idx, &endptr, 10);
			X[i][feature_index] = strtod(val, &endptr);
			++j;
		}
	}

	// Save X and y to file
	std::ofstream file("X_data.csv");
	for (const auto& vec : X) {
		for (size_t i = 0; i < vec.size(); ++i) {
			file << vec[i];
			if (i < vec.size() - 1) {
				file << ",";  // Use ',' as a separator for CSV format
			}
		}
		file << "\n";
	}
	file.close();

	std::ofstream file1("Y_data.csv");
	for (size_t i = 0; i < y.size(); ++i) {
		file1 << y[i];
		file1 << "\n";
	}
	file1.close();

	// Create train test split, fills X_train, X_test, y_train, y_test
	splitData(X, y, 0.8);

	free(line);
	fclose(fp);
}

void SKLearnModel::splitData(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double splitRatio) {
	// Set random seed for reproducibility
	std::srand(static_cast<unsigned>(std::time(nullptr)));

	size_t dataSize = X.size();
	size_t trainSize = static_cast<size_t>(dataSize * splitRatio);

	// Create a random permutation of indices
	std::vector<size_t> indices(dataSize);
	for (size_t i = 0; i < dataSize; ++i) {
		indices[i] = i;
	}
	std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

	// Split the data based on the random indices
	for (size_t i = 0; i < dataSize; ++i) {
		if (i < trainSize) {
			X_train.push_back(X[indices[i]]);
			y_train.push_back(y[indices[i]]);
		}
		else {
			X_test.push_back(X[indices[i]]);
			y_test.push_back(y[indices[i]]);
		}
	}
}

std::vector<double> SKLearnModel::predict(std::map<long, double> &data, int16_t label, int16_t &predictLabel) {
	scaleData(data, LOWER, UPPER); // TODO make cleaner

	std::vector<double> x(X_train[0].size(), 0.0);

	// Convert binned data to vector
	for (auto const &[idx, val] : data)
	{
		if (idx > x.size()) {
			std::cout << "Warning: Ignoring feature (likely template) with index " << idx
				<< " as it is out-of-bound (trained feature size: " << x.size() << ")." << std::endl;
		}
		x[idx] = val;
	}

	predictLabel = predict({ x })[0];
	std::vector<double> probEstimates(2, 0.0);

	return probEstimates;
}

std::vector<double> SKLearnModel::predict(const std::vector<std::vector<double>>& X) {
	if (!pyModel) {
		std::cerr << "Model not initialized." << std::endl;
		return std::vector<double>();  // Return empty vector on failure
	}
	PyObject* pyX = toNpArray(X);
	//PyObject* pyX = matToPyObj(X);

	if (!pyX) {
		std::cerr << "Failed to create Python object from input data." << std::endl;
		Py_DECREF(pyX);
		return std::vector<double>();  // Return empty vector on failure
	}

	
	/*
	if (modelType == LogisticRegression && scalerInstance) {
		PyObject* scaledX = PyObject_CallMethod(scalerInstance, "transform", "O", pyX);
		if (!scaledX) {
			PyErr_Print();
			std::cerr << "StandardScaler transform failed." << std::endl;
		}
		else {
			Py_DECREF(pyX);
			pyX = scaledX;
		}
	}*/

	// Print out the normalized X
	/*
	PyObject* pyX_repr = PyObject_Repr(pyX);
	if (pyX_repr) {
		std::cout << "Normalized X: " << PyUnicode_AsUTF8(pyX_repr) << std::endl;
		Py_DECREF(pyX_repr);
	}
	else {
		std::cerr << "Failed to obtain string representation of normalized X." << std::endl;
	}
	*/

	PyObject* pyResult = PyObject_CallMethod(pyModel, "predict", "O", pyX);
	Py_DECREF(pyX);  // Decrease reference count of pyX after usage

	if (!pyResult) {
		PyErr_Print();
		std::cerr << "sklearn predict() failed\n";
		return std::vector<double>();  // Return empty vector on failure
	}

	std::vector<double> predictions;

	if (PyArray_Check(pyResult)) {
		PyArrayObject* pyArray = reinterpret_cast<PyArrayObject*>(pyResult);
		if (PyArray_TYPE(pyArray) == NPY_DOUBLE) {
			double* data = static_cast<double*>(PyArray_DATA(pyArray));
			npy_intp size = PyArray_SIZE(pyArray);
			predictions.assign(data, data + size);
		}
	}
	else {
		PyErr_Print();
		std::cerr << "Expected a numpy array but got something else.\n";
	}

	
	Py_DECREF(pyResult);
	return predictions;
}

PyObject* SKLearnModel::toNpArray(const std::vector<double>& data) {
	npy_intp dims[1] = { static_cast<npy_intp>(data.size()) }; // For 1D array
	PyObject* numpyArray = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	double* numpyArrayData = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(numpyArray)));

	std::copy(data.begin(), data.end(), numpyArrayData);
	return numpyArray;
}

PyObject* SKLearnModel::toNpArray(const std::vector<std::vector<double>>& data) {
	npy_intp dims[2] = { static_cast<npy_intp>(data.size()), static_cast<npy_intp>(data[0].size()) };
	PyObject* numpyArray = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	double* numpyArrayData = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(numpyArray)));

	for (size_t i = 0; i < data.size(); i++) {
		std::copy(data[i].begin(), data[i].end(), numpyArrayData + i * data[i].size());
	}

	return numpyArray;
}

PyObject* SKLearnModel::matToPyObj(const std::vector<std::vector<double>>& data) {
	PyObject* pyList = PyList_New(data.size());
	for (size_t i = 0; i < data.size(); ++i) {
		PyList_SetItem(pyList, i, vecToPyObj(data[i]));
	}
	return pyList;
}

PyObject* SKLearnModel::vecToPyObj(const std::vector<double>& data) {
	PyObject* pyList = PyList_New(data.size());
	for (size_t i = 0; i < data.size(); ++i) {
		PyList_SetItem(pyList, i, PyFloat_FromDouble(data[i]));
	}
	return pyList;
}

void SKLearnModel::loadModel(const std::string& path) {
	PyGILState_STATE gil_state = PyGILState_Ensure();

	PyObject* joblib = PyImport_ImportModule("joblib");
	if (!joblib) {
		PyErr_Print();
		PyGILState_Release(gil_state);
		return;
	}

	PyObject* load_func = PyObject_GetAttrString(joblib, "load");
	if (!load_func || !PyCallable_Check(load_func)) {
		PyErr_Print();
		Py_DECREF(joblib);
		PyGILState_Release(gil_state);
		return;
	}

	PyObject* args = Py_BuildValue("(s)", path.c_str());
	PyObject* loaded_model = PyObject_CallObject(load_func, args);
	Py_DECREF(args);
	Py_DECREF(load_func);
	Py_DECREF(joblib);

	if (!loaded_model) {
		PyErr_Print();
		PyGILState_Release(gil_state);
		return;
	}

	// Replace the old model with the new one
	Py_XDECREF(pyModel);  // Decrease the reference count of the old model
	pyModel = loaded_model;  // Assign the loaded model

	PyGILState_Release(gil_state);
}

void SKLearnModel::saveModel(const std::string& path) {
	PyGILState_STATE gil_state = PyGILState_Ensure();

	PyObject* joblib = PyImport_ImportModule("joblib");
	if (!joblib) {
		PyErr_Print();
		PyGILState_Release(gil_state);
		return;
	}

	PyObject* dump_func = PyObject_GetAttrString(joblib, "dump");
	if (!dump_func || !PyCallable_Check(dump_func)) {
		PyErr_Print();
		Py_DECREF(joblib);
		PyGILState_Release(gil_state);
		return;
	}

	PyObject* args = Py_BuildValue("(Os)", pyModel, path.c_str());
	PyObject* result = PyObject_CallObject(dump_func, args);
	Py_DECREF(args);
	Py_DECREF(dump_func);
	Py_DECREF(joblib);

	if (!result) {
		PyErr_Print();
	}
	else {
		Py_DECREF(result);
	}

	PyGILState_Release(gil_state);
}
