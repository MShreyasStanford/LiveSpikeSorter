#include "dataBinner.h"
#include <thread>
#include "Decoder.h"
#include "../Networking/NetworkHelpers.h"
#include "../Helpers/TimeHelpers.h"

DataBinner::DataBinner(int windowLength, int binLength, int windowOffset)
	: windowLength(windowLength)
	, binLength(binLength)
	, windowOffset(windowOffset)
	, nBins(windowLength / binLength + 1)
	, nextBinTime(binLength)
	, currentIndex(0)
{
	// Initialize bins with empty maps
	for (int i = 0; i < nBins; i++) {
		std::map<long, double> bin;
		bins.push_back(bin);
	}
};

DataBinner::~DataBinner() {};

void DataBinner::insert(std::vector<long> times, std::vector<long> channels) {
	std::lock_guard lk(guard);

	// Iterate over the spikes
	for (int i = 0; i < times.size(); i++) {
		// Update bins to new data, no assumptions on gaps or anything, as long as data comes in monotonically
		while (times[i] > nextBinTime) {
			// Set index to oldest bin
			currentIndex = (currentIndex + 1) % nBins;

			// Clear oldest bin for new data
			bins[currentIndex].clear();
			
			// Update next subbin time
			nextBinTime += binLength;
		}

		// Increment spike count of the channel by 1 
		// If the spike time received is too old so as to not make sense in circular buffer, ignore
		if ((nextBinTime - times[i]) / binLength < nBins) {
			// ex: d = (nextBinTime - times[i]) / binLength = how many indices behind currentIndex times[i] should be in
			//    Thus, we should place times[i] into (currentIndex - d) mod nBins
			//    However, in C++ -7 % 2 == -2, and so we adjust for this by doing (nBins + (x % nBins)) % nBins
			//    instead of just x % nBins
			bins[(nBins + (currentIndex - (nextBinTime - times[i]) / binLength) % nBins) % nBins][channels[i]]++;
		}
		else {
			std::cout << "Spike with spike time = " << times[i] << " discarded, as it is too old." << std::endl;
		}
	}
}

void DataBinner::updateTime(long time) 
{
	std::lock_guard lk(guard);
	while (time > nextBinTime) {
		currentIndex = (currentIndex + 1) % nBins;
		bins[currentIndex].clear();
		nextBinTime += binLength;
	}
}

// Return bin map that is the accumulation of all the subbin maps
std::map<long, double> DataBinner::getDataWindow() {
	std::lock_guard lk(guard);

	// Set window as copy of first bin 
	std::map<long, double> window = bins[0];

	// Iterate over bins
	for (int i = 1; i < nBins; i++) {
		// Iterate over bin's channel counts
		for (auto const& [channel, count] : bins[i]) {
			window[channel] += count;
		}
	}

	return window;
}


void DataBinner::readInSpikes(const char* csvFilePath, const char* eventFilePath, const char* outputFilePath, FileWriter *m_fwOut) {
	std::vector<long> times, channels;
	std::cout << "Binner reading in from " << csvFilePath << " and " << eventFilePath << std::endl;
	FILE *fpIn;
	FILE *fpLabel;
	FILE *fpOut;

	if ((fpIn = fopen(csvFilePath, "r")) == NULL)
		fprintf(stderr, "Can't open input file %s\n", csvFilePath);
	if ((fpLabel = fopen(eventFilePath, "r")) == NULL)
		fprintf(stderr, "Can't open input file %s\n", eventFilePath);
	fpOut = fopen(outputFilePath, "w");

	// Extract label and next label
	int eventLabel;
	int eventTime;
	fscanf(fpLabel, "%d %d", &eventTime, &eventLabel);

	long time, channel;
	float amplitude; // Thrown away

	while (fscanf(fpIn, "%20ld,%4ld,%f", &time, &channel, &amplitude) == 3)
	{
		// If a window's length of time has passed since the time event, record dataWindow and extract a new label
		if (eventTime + windowOffset < time) {
			if (times.size() > 0) {
				// Insert times and channels vectors into the data binner
				insert(times, channels);
				updateTime(eventTime);
				times.clear();
				channels.clear();

				std::map<long, double> window = getDataWindow();
				m_fwOut->WriteDecoderInput(eventLabel, window);
			}
			else {	
				std::cout << "Skipping event " << eventTime << std::endl;
			}
			// Extract the time of the next event and its label
			if (fscanf(fpLabel, "%d %d", &eventTime, &eventLabel) != 2)
				break;
		}

		// Insert time and channel
		times.push_back(time);
		channels.push_back(channel);
	}
	fclose(fpIn);
	fclose(fpLabel);
	fclose(fpOut);

	// Reset databinner for real time sorting
	for (int i = 0; i < nBins; i++) {
		bins[i].clear();
	}
	currentIndex = 0;
	nextBinTime = binLength;
}

/*-
 * Copyright (c) 1990, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
char* DataBinner::strsep(char **stringp, const char *delim) {
	char *s;
	const char *spanp;
	int c, sc;
	char *tok;

	if ((s = *stringp) == NULL)
		return (NULL);
	for (tok = s;;) {
		c = *s++;
		spanp = delim;
		do {
			if ((sc = *spanp++) == c) {
				if (c == 0)
					s = NULL;
				else
					s[-1] = 0;
				*stringp = s;
				return (tok);
			}
		} while (sc != 0);
	}
}