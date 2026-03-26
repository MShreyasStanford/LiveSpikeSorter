#ifndef DATA_BINNER_H_
#define DATA_BINNER_H_

#include <vector>
#include <map>
#include <mutex>
#include <unordered_set>
#include "../Helpers/FileWriter.h"

// DataBinner is an implementation of a Circular Buffer
class DataBinner {
public:
	DataBinner(int windowLength, int binLength, int windowOffset);
	~DataBinner();

	void insert(std::vector<long> times, std::vector<long> channels);
	void updateTime(long time);
	std::map<long, double> getDataWindow();
	void readInSpikes(const char* csvFilePath, const char* eventFilePath, const char* outputFilePath, FileWriter *m_fwOut,
	                  const std::unordered_set<long>* channelFilter = nullptr);

protected:
	char* strsep(char **stringp, const char *delim);

	int windowLength;
	int binLength;
	int windowOffset;
	int nBins;

	// Index of the bin containing the newest data.
	// Accordingly, the index of the bin containing the oldest data is currentIndex + 1
	int currentIndex; 

	// Time when DataBinner stops filling current bin and begins filling next bin
	long nextBinTime;

	std::vector<std::map<long, double>> bins;
	
	// Mutex to ensure insert() does not occur same time as getDataWindow()
	std::mutex guard;
};

#endif
