#include <iostream>

#include "SVMModel.h"
#include "../Helpers/Utils.h"

void print_null(const char *s) {}

SVMModel::SVMModel()
{
	/*
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC		(multi-class classification)\n"
	"	1 -- nu-SVC		(multi-class classification)\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR	(regression)\n"
	"	4 -- nu-SVR		(regression)\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	*/
	
	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF; 
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1; // Always set to 1 to allow for threshold decision making
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	void(*print_func)(const char*) = NULL;	// default printing to stdout
	print_func = &print_null; // Comment to make svm not quiet
	svm_set_print_string_function(print_func);
}


SVMModel::~SVMModel() {
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
}

void SVMModel::init(const std::string spikeFileName, const std::string workFolderPathName) {
	// Loads and Scales data
	BaseModel::init(spikeFileName, workFolderPathName);	

	int nr_class = svm_get_nr_class(model);
	probEstimates.resize(nr_class, 0);
	
	// Fill in Labels vector, which is needed because libsvm's probability output appears in the order the class labels are ordered in training data
	int *labelsTemp = (int *)malloc(nr_class * sizeof(int));
	svm_get_labels(model, labelsTemp);
	labels = std::vector<int>(labelsTemp, labelsTemp + nr_class);
	free(labelsTemp);
}

/*
-------------------------------------------------------------------------------------
--------------------------------------Training---------------------------------------
-------------------------------------------------------------------------------------
*/

void SVMModel::train(const char* input_file_name)
{
	static const char *ptLabel = { "Model::train" };

	read_problem(input_file_name);

	// Parameter search to determine C
	_DEBUG_PUT_0(ptLabel, "Beginning model SVM parameter search");
	std::vector<double> Cs{ 2E-2, 2E-1, 1, 2, 5, 10, 20, 50, 100, 500 };
	float bestCrossValAcc = 0.0;
	double bestC = NULL;
	for (double C : Cs) {
		param.C = C;
		float crossValAcc = do_cross_validation(prob, param);
		if (crossValAcc > bestCrossValAcc) {
			_DEBUG_PUT_0(ptLabel, "C=" << C << " achieved a new best cross-validation accuracy of " << crossValAcc);
			bestCrossValAcc = crossValAcc;
			bestC = C;
		}
	}
	_DEBUG_PUT_0(ptLabel, "The model SVM will use parameters C=" << bestC);
	param.C = bestC;

	error_msg = svm_check_parameter(&prob, &param);

	if (error_msg)
	{
		fprintf(stderr, "ERROR: %s\n", error_msg);
		exit(1);
	}
	
	model = svm_train(&prob, &param);
}



float SVMModel::do_cross_validation(struct svm_problem &prob, struct svm_parameter &param)
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double *target = Malloc(double, prob.l);
	int nr_fold = 5; // Can change this!

	svm_cross_validation(&prob, &param, nr_fold, target);
	for (i = 0; i < prob.l; i++) {
		if (target[i] == prob.y[i])
			++total_correct;
	}
	free(target);
	float accuracy = 100.0*total_correct / prob.l;
	return accuracy;
}

// read in a problem (in svmlight format)
void SVMModel::read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename, "r");
	char *endptr;
	char *idx, *val, *label;

	if (fp == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	int max_line_len = 1024;
	char* line = Malloc(char, max_line_len);
	while (readline(fp, &line, &max_line_len) != NULL)
	{
		char *p = strtok(line, " \t"); // label

		// features
		while (1)
		{
			p = strtok(NULL, " \t");
			if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = 0;
	j = 0;
	for (i = 0; i < prob.l; i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp, &line, &max_line_len);
		prob.x[i] = &x_space[j];
		label = strtok(line, " \t\n");
		if (label == NULL) // empty line
			exit_input_error(i + 1);

		prob.y[i] = strtod(label, &endptr);
		if (endptr == label || *endptr != '\0')
			exit_input_error(i + 1);

		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int)strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i + 1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i + 1);

			++j;
		}

		if (inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if (param.gamma == 0 && max_index > 0)
		param.gamma = 1.0 / max_index;

	if (param.kernel_type == PRECOMPUTED)
		for (i = 0; i < prob.l; i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	free(line);
	fclose(fp);
}




/*
-------------------------------------------------------------------------------------
--------------------------------------Prediction-------------------------------------
-------------------------------------------------------------------------------------
*/

static int(*info)(const char *fmt, ...) = &printf;

void SVMModel::svmPredict(std::map<long, double> &data, int16_t label, int16_t &predictLabel)
{
	int max_nr_attr = 64;
	struct svm_node *x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));
	if (svm_check_probability_model(model) == 0)
	{
		fprintf(stderr, "Model does not support probabiliy estimates\n");
		exit(1);
	}

	int i = 0;
	double target_label = double(label);

	for (auto const &[idx, val] : data)
	{
		if (i >= max_nr_attr - 1)	// need one more for index = -1
		{
			max_nr_attr *= 2;
			x = (struct svm_node *) realloc(x, max_nr_attr * sizeof(struct svm_node));
		}

		if (val == NULL || val == 0)
			continue;
		x[i].index = idx + 1; // + 1 is because the label is idx=0. Could remove since we don't scale label

		x[i].value = val;
		++i;
	}
	x[i].index = -1;

	// Use model to accquire the probabilities of the different labels
	double *prob_estimates = (double *)malloc(probEstimates.size() * sizeof(double));
	predictLabel = svm_predict_probability(model, x, prob_estimates); // Fill double pointer array in c function
	for (int j = 0; j < probEstimates.size(); j++) // Copy to vector
		probEstimates[labels[j]] = prob_estimates[j]; 
	free(prob_estimates); 


	// Prediction without probability can be better in performance?
	//predictLabel = svm_predict(model, x);

	free(x);
}


std::vector<double> SVMModel::predict(std::map<long, double> &data, int16_t label, int16_t &predictLabel) {
	scaleData(data, LOWER, UPPER); // TODO make cleaner
	svmPredict(data, label, predictLabel);
	return probEstimates;
}