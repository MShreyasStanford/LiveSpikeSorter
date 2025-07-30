#include "BaseModel.h"
#include "../Helpers/Utils.h"

BaseModel::BaseModel() {
	;
}

BaseModel::~BaseModel() {
	;
}

void BaseModel::init(const std::string spikeFileName, const std::string workFolderPathName) {
	std::string scaledDataFileName = workFolderPathName + "binnedSpikesScaled.txt";

	// Compute Scale params and apply to training data
	computeScaleParams(spikeFileName.c_str());
	scaleFileData(spikeFileName.c_str(), scaledDataFileName.c_str(), LOWER, UPPER, false, 0.0, 0.0);

	// Train model on scaled data
	train(scaledDataFileName.c_str());
}

// Helper function for parsing txt files
char* BaseModel::readline(FILE *input, char **line, int *max_line_len)
{
	int len;

	if (fgets(*line, *max_line_len, input) == NULL)
		return NULL;

	while (strrchr(*line, '\n') == NULL)
	{
		*max_line_len *= 2;
		*line = (char *)realloc(*line, *max_line_len);
		len = (int)strlen(*line);
		if (fgets(*line + len, *max_line_len - len, input) == NULL)
			break;
	}
	return *line;
}

void BaseModel::computeScaleParams(const char* inputFileName) {
	int i, index;
	FILE *fpIn = NULL;
	long int num_nonzeros = 0;
	long int new_num_nonzeros = 0;
	double y_min = DBL_MAX;
	double y_max = -DBL_MAX;

	fpIn = fopen(inputFileName, "r");

	if (fpIn == NULL)
	{
		fprintf(stderr, "can't open file %s\n", inputFileName);
		exit(1);
	}

	int max_line_len = 1024;
	char *line = (char *)malloc(max_line_len * sizeof(char));

#define SKIP_TARGET\
	while(isspace(*p)) ++p;\
	while(!isspace(*p)) ++p;

#define SKIP_ELEMENT\
	while(*p!=':') ++p;\
	++p;\
	while(isspace(*p)) ++p;\
	while(*p && !isspace(*p)) ++p;

	/* assumption: min index of attributes is 1 */
	/* pass 1: find out max index of attributes */
	int max_index = 0;
	int min_index = 0;

	while (readline(fpIn, &line, &max_line_len) != NULL)
	{
		char *p = line;

		SKIP_TARGET

			while (sscanf(p, "%d:%*f", &index) == 1)
			{
				max_index = max(max_index, index);
				min_index = min(min_index, index);
				SKIP_ELEMENT
					num_nonzeros++; // TODO remove?
			}
	}

	if (min_index < 0)
		fprintf(stderr,
			"WARNING: minimal feature index is %d, but indices should start from 0\n", min_index);

	rewind(fpIn);

	featureMaxs = std::vector<double>(max_index + 1, 0);
	featureMins = std::vector<double>(max_index + 1, 0);

	for (i = 0; i <= max_index; i++) {
		featureMaxs[i] = -DBL_MAX;
		featureMins[i] = DBL_MAX;
	}

	/* pass 2: find out min/max value */
	while (readline(fpIn, &line, &max_line_len) != NULL)
	{
		char *p = line;
		int next_index = 0;
		double target;
		double value;

		if (sscanf(p, "%lf", &target) != 1) {
			free(line);
			fclose(fpIn);
			return;
		}
		y_max = max(y_max, target);
		y_min = min(y_min, target);

		SKIP_TARGET

			while (sscanf(p, "%d:%lf", &index, &value) == 2)
			{
				for (i = next_index; i < index; i++) {
					featureMaxs[i] = max(featureMaxs[i], 0);
					featureMins[i] = min(featureMins[i], 0);
				}

				featureMaxs[index] = max(featureMaxs[index], value);
				featureMins[index] = min(featureMins[index], value);

				SKIP_ELEMENT
					next_index = index + 1;
			}

		for (i = next_index; i <= max_index; i++) {
			featureMaxs[i] = max(featureMaxs[i], 0);
			featureMins[i] = max(featureMins[i], 0);
		}
	}
	free(line);
	fclose(fpIn);
}

void BaseModel::scaleFileData(const char* inputFileName, const char* spikeFileName, double lower, double upper, bool y_scaling, double y_lower, double y_upper) {
	int i, index;
	FILE *fpIn, *fpOut = NULL;
	double y_min = DBL_MAX;
	double y_max = -DBL_MAX;

	fpIn = fopen(inputFileName, "r");

	if (fpIn == NULL)
	{
		fprintf(stderr, "can't open file %s\n", inputFileName);
		exit(1);
	}

	int max_line_len = 1024;
	char *line = (char *)malloc(max_line_len * sizeof(char));

	fpOut = fopen(spikeFileName, "w");

	while (readline(fpIn, &line, &max_line_len) != NULL)
	{
		char *p = line;
		int next_index = 1;
		double target;
		double value;

		if (sscanf(p, "%lf", &target) != 1) {
			free(line);
			fclose(fpIn);
			fclose(fpOut);
			exit(-1);
		}
		output_target(fpOut, target, y_scaling, y_lower, y_upper, y_min, y_max);

		SKIP_TARGET

			while (sscanf(p, "%d:%lf", &index, &value) == 2)
			{
				for (i = next_index; i < index; i++)
					output(fpOut, i, 0, lower, upper);

				output(fpOut, index, value, lower, upper);

				SKIP_ELEMENT
					next_index = index + 1;
			}

		for (i = next_index; i <= featureMaxs.size() - 1; i++)
			output(fpOut, i, 0, lower, upper);

		fprintf(fpOut, "\n");
	}

	free(line);
	fclose(fpIn);
	fclose(fpOut);
}

void BaseModel::scaleData(std::map<long, double> &data, double lower, double upper) {
	for (auto &[index, value] : data) {
		// If featureMax of an index is 0, means that feature was not seen in training data, so the feature is scaled to 0.
		if (featureMaxs[index + 1] == 0) {
			value = 0;
			continue;
		}

		/* skip single-valued attribute */
		if (featureMaxs[index + 1] == featureMins[index + 1]) // Should this be here?
			continue;

		if (value <= featureMins[index + 1])
			value = lower;
		else if (value >= featureMaxs[index + 1])
			value = upper;
		else
			value = lower + (upper - lower) *
			(value - featureMins[index + 1]) /
			(featureMaxs[index + 1] - featureMins[index + 1]);
	}
}

/*
-------------------------------------------------------------------------------------
---------------------------------------SCALING---------------------------------------
-------------------------------------------------------------------------------------
*/

void BaseModel::output_target(FILE *fpOut, double value, bool y_scaling, double y_lower, double y_upper, double y_min, double y_max)
{
	if (y_scaling)
	{
		if (value == y_min)
			value = y_lower;
		else if (value == y_max)
			value = y_upper;
		else value = y_lower + (y_upper - y_lower) *
			(value - y_min) / (y_max - y_min);
	}
	fprintf(fpOut, "%.17g ", value);
}


void BaseModel::output(FILE *fpOut, int index, double value, double lower, double upper)
{
	/* skip single-valued attribute */
	if (featureMaxs[index] == featureMins[index])
		return;

	if (value <= featureMins[index])
		value = lower;
	else if (value >= featureMaxs[index])
		value = upper;
	else
		value = lower + (upper - lower) *
		(value - featureMins[index]) /
		(featureMaxs[index] - featureMins[index]);

	if (value != 0)
		fprintf(fpOut, "%d:%g ", index, value);
}