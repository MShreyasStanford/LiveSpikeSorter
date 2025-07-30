#include "RegressionModel.h"
#include <iostream>
#include <fstream> // Temporary
#include <algorithm> // For std::shuffle
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include <random>

RegressionModel::RegressionModel()
{
	;
}


RegressionModel::~RegressionModel() {
	;
}

void RegressionModel::init(const std::string spikeFileName, const std::string workFolderPathName) {
	// Loads and Scales data
	BaseModel::init(spikeFileName, workFolderPathName);

}

/*
-------------------------------------------------------------------------------------
--------------------------------------Training---------------------------------------
-------------------------------------------------------------------------------------
*/

void RegressionModel::train(const char* input_file_name)
{
	static const char *ptLabel = { "Model::train" };

	create_dataset(input_file_name);
	
	size_t n_features = X_train[0].size();

	// Initialize weights
	theta = std::vector<double>(n_features, 0.0);

	// Set hyperparameters
	double alpha = 0.1; // Learning rate
	int iterations = 10000; // Number of iterations

	// Train
	std::cout << "Training..." << std::endl;
	gradientDescent(X_train, y_train, alpha, iterations);

	// Print theta
	for (auto th : theta) {
		std::cout << th << " ";
	}
	std::cout << std::endl;
	// Test on test set
	std::vector<int16_t> predictions;
	for (auto x : X_test) {
		predictions.push_back(predictOnce(x));
	}
	double accuracy = calculateAccuracy(predictions, y_test);

	std::cout << "Accuracy: " << accuracy << std::endl;
}


// read in a problem
void RegressionModel::create_dataset(const char *scaledDataFileName)
{
	int max_index, feature_index, i;
	size_t len_data, j;
	FILE *fp = fopen(scaledDataFileName, "r");
	char *endptr;
	char *idx, *val, *label;

	

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
			if(feature_index > max_index)
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
			fprintf(stderr, "Empty line at line %d\n", i+1);
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

			//errno = 0;
			//x_space[j].index = (int)strtol(idx, &endptr, 10);
			//if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
			//	exit_input_error(i + 1);
			//else
			//	inst_max_index = x_space[j].index;

			//errno = 0;
			//x_space[j].value = strtod(val, &endptr);
			//if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
			//	exit_input_error(i + 1);

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

void RegressionModel::splitData(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double splitRatio) {
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


///*
//-------------------------------------------------------------------------------------
//--------------------------------------Prediction-------------------------------------
//-------------------------------------------------------------------------------------
//*/
//
////static int(*info)(const char *fmt, ...) = &printf;
//
std::vector<double> RegressionModel::predict(std::map<long, double> &data, int16_t label, int16_t &predictLabel) {
	scaleData(data, LOWER, UPPER); // TODO make cleaner

	std::vector<double> x(theta.size(), 0.0);
	// Convert binned data to vector
	for (auto const &[idx, val] : data)
	{
		if (idx > x.size()) {
			std::cout << "Data out of bounds from the bins that were trained." << std::endl;
			exit(1);
		}
		x[idx] = val;
	}

	predictLabel = predictOnce(x);
	std::vector<double> probEstimates(2, 0.0);

	//svmPredict(data, label, predictLabel);
	return probEstimates;
}


//-------------------------------------------------------------------------------------
//--------------------------Logistic Regression Algorithm------------------------------
//-------------------------------------------------------------------------------------

// Define the logistic function
double RegressionModel::sigmoid(double z) {
	return 1.0 / (1.0 + exp(-z));
}

// Implement the hypothesis function
double RegressionModel::hypothesis(const std::vector<double>& x) {
	double result = 0.0;
	for (size_t i = 0; i < theta.size(); ++i) {
		result += theta[i] * x[i];
	}
	return sigmoid(result);
}

// Implement the gradient descent algorithm
void RegressionModel::gradientDescent(std::vector<std::vector<double>>& X, std::vector<int16_t>& y, double alpha, int iterations) {
	size_t m = X.size(); // Number of training examples
	size_t n = X[0].size(); // Number of features

	for (int iter = 0; iter < iterations; ++iter) {
		std::vector<double> gradient(n, 0.0);

		for (size_t i = 0; i < m; ++i) {
			double h = hypothesis(X[i]);
			double error = h - y[i];

			for (size_t j = 0; j < n; ++j) {
				gradient[j] += error * X[i][j];
			}
		}

		for (size_t j = 0; j < n; ++j) {
			theta[j] -= (alpha / m) * gradient[j];
		}
	}
}

int16_t RegressionModel::predictOnce(const std::vector<double>& x) {

	double probability = hypothesis(x);
	int16_t prediction = (probability >= 0.5) ? 1 : 0;

	return prediction;
}

double RegressionModel::calculateAccuracy(const std::vector<int16_t>& predictions, const std::vector<int16_t>& y_test) {
	size_t correctPredictions = 0;

	for (size_t i = 0; i < predictions.size(); ++i) {
		if (predictions[i] == y_test[i]) {
			correctPredictions++;
		}
	}
	std::cout << "Correct predictions: " << correctPredictions << std::endl;
	std::cout << "Total predictions: " << predictions.size() << std::endl;

	return static_cast<double>(correctPredictions) / predictions.size();
}