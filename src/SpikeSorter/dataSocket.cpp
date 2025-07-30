#if _WIN32 || _WIN64
#include <windows.h>
#include <stdlib.h>
#else
#include <unistd.h>
#endif

#include <sstream>
#include <errno.h>
#include <iostream>
#include <cmath>
#include <bitset>
#include <atomic>
#include "../../External/sglx/SglxCppClient.h"

#include "dataSocket.h"
#include "../Helpers/utils.h"

#include "../Helpers/Timer.h"

static std::mutex sglxMutex;

// ------------------------------------------------------------------------------
//
// Name			: StreamDataSocket::StreamDataSocket()
//
// Description  : Constructor, initialize retrieve buffer and connection settings
//
// ------------------------------------------------------------------------------
StreamDataSocket::StreamDataSocket(std::string accquisitionHost, uint16 accquisitionPort, int substream, long lMaxSize, long lMinSize, float fImecSampRate, float fNiqdSampRate, int downSampling)
	: m_sHost(accquisitionHost)
	, m_uPort(accquisitionPort)
	, m_iSubstream(substream)
	//, m_vImecChannels(channelMap, channelMap + lNChans) // Fill vector with values of *int array
	, m_vNidqChannels({8}) // Extracting only the digital line (the 9th line) from nidq
{
	static const char *ptLabel = { "StreamDataSocket::StreamDataSocket" };

	// Initialize the static data members inherited by parent DataSocket class
	m_lMaxSize = lMaxSize;
	m_lMinSize = lMinSize;
	//m_lNChans = lNChans;
	m_lDownsampling = 1; // = downSampling;
	m_fImecSampRate = fImecSampRate;
	m_fNidqSampRate = fNiqdSampRate;

	if (!connect()) {
		std::cout << "Error while trying to establish connection to " << accquisitionHost << ":" << accquisitionPort << "!" << std::endl; // TODO redundant amount of errors. Also need more errors for other parts
		_RUN_ERROR(ptLabel, "Couldn't establish connection");
	}
};


StreamDataSocket::~StreamDataSocket() {
	disconnect();
};


bool StreamDataSocket::connect() {

	//m_mSGlxMutex.lock();
	std::cout << "Trying to connect to " << getHost() << ":" << getPort() << "..." << std::endl;
	{
		std::unique_lock<std::mutex> lock(m_mSGlxMutex);
		if (!sglx_connect_std(S, m_sHost, m_uPort)) {
		error:
			printf("error [%s]\n", S.err.c_str());
		}
	}
			std::cout << "...Connection successful!" << std::endl;
	//m_mSGlxMutex.unlock();
	return true;
}


void StreamDataSocket::disconnect() {
	m_mSGlxMutex.lock();
	sglx_close(S);
	std::cout << "... Disconnected!" << std::endl;
	m_mSGlxMutex.unlock();
}


t_ull StreamDataSocket::getStreamSampleCt(int streamType, OSSSpecificParams osParams) {
	//m_mSGlxMutex.lock();
	std::unique_lock<std::mutex> lock(sglxMutex);
	t_ull sampleCt = (streamType == IMEC) ? sglx_getStreamSampleCount(S, streamType, osParams.substream) : sglx_getStreamSampleCount(S, streamType, 0);
	//m_mSGlxMutex.unlock();
	return sampleCt;
}

// sglx_fetch returns the sample count index of first sample in matrix, or zero if error. StreamDataSocket::fetch, however,
// returns the sample count index of the last sample in matrix.
t_ull StreamDataSocket::fetch(std::vector<short> &data, t_sglxconn &S, int streamType, int substream, t_ull startSamp, int maxSamps, const std::vector<int> &channelSubset) {
	//m_mSGlxMutex.lock();
	std::unique_lock<std::mutex> lock(sglxMutex);
	t_ull lLatestCt = sglx_fetch(data, S, streamType, substream, startSamp, maxSamps, channelSubset) + maxSamps;
	//m_mSGlxMutex.unlock();
	return lLatestCt;
}

// fetchLatest: lStreamSampleCt (the start of the fetch) can change
t_ull StreamDataSocket::fetchLatest(float *fData, OSSSpecificParams osParams, t_ull lStartCt) {
	Timer timer("fetchLatest()");
	t_ull lLatestCt = getStreamSampleCt(IMEC, osParams);

	// If default start value, fetch the min size. Otherwise, calculate the number of samples from lStartCt to current
	t_ull lToGet = lLatestCt - lStartCt;
	if (lStartCt == ULLONG_MAX) {
		lToGet = m_lMinSize;
	}

	// Limit the number of samples fetched to lMaxSize samples
	if (lToGet > m_lMaxSize) {
		lToGet = m_lMaxSize;
	}

	lStartCt = lLatestCt - lToGet;
	lLatestCt = fetch(m_sFetchBuffer, S, IMEC, osParams.substream, lStartCt, lToGet, osParams.vImecChannels);

	//Fill fData with m_sFetchBuffer's contents
	for (long lI = 0; lI < lToGet * osParams.lNChans; lI++) {
		fData[lI] = (float)m_sFetchBuffer[lI];
	}
	return lLatestCt;
}

t_ull StreamDataSocket::fetchLatest_TC(float *fData, OSSSpecificParams osParams, t_ull lStartCt) {
	t_ull lLatestCt = getStreamSampleCt(IMEC, osParams);

	// If default start value, fetch the min size. Otherwise, calculate the number of samples from lStartCt to current
	t_ull lToGet = lLatestCt - lStartCt;
	if (lStartCt == ULLONG_MAX) {
		lToGet = m_lMinSize;
		lStartCt = lLatestCt - m_lMinSize;
	}

	// Limit the number of samples fetched to lMaxSize samples
	if (lToGet > m_lMaxSize) {
		lToGet = m_lMaxSize;
		lStartCt = lLatestCt - m_lMaxSize;
	}

	lLatestCt = fetch(m_sFetchBuffer, S, IMEC, osParams.substream, lStartCt, lToGet, osParams.vImecChannels);

	//Fill fData with m_sFetchBuffer's contents
	for (long lI = 0; lI < lToGet * osParams.lNChans; lI++)
		fData[lI] = (float)m_sFetchBuffer[lI];

	//std::cout << "number of samples: " << lToGet << std::endl;
	//std::cout << "number of channels: " << m_lNChans << std::endl;

	return lLatestCt,lToGet;
}


// fetchFromPlace: lStreamSampleCt (the start of the fetch) is constant
t_ull StreamDataSocket::fetchFromPlace(float *fData, OSSSpecificParams osParams, t_ull lStartCt) {
	t_ull lLatestCt = getStreamSampleCt(IMEC, osParams);

	// Limit the fetch amount to the max size
	t_ull lToGet = min(m_lMaxSize, lLatestCt - lStartCt);

	lLatestCt = fetch(m_sFetchBuffer, S, IMEC, osParams.substream, lStartCt, lToGet, osParams.vImecChannels);

	//Fill data buffer with m_sFetchBuffer
	for (long lI = 0; lI < lToGet * osParams.lNChans; lI++)
		fData[lI] = (float)m_sFetchBuffer[lI];

	return lLatestCt;
}

t_ull StreamDataSocket::initNidqStream() {
	/* Wait some time before starting NIDQ stream fetching to avoid SpikeGLX errors.
	If you are receiving "[FETCH: Too late.]" error messages from
	the fetchEventInfo call, increase the time waited. */
	Sleep(1000); // TODO explore how low this value can go.
	//commented out for invalid argument
	//t_ull lLatestSampleCt = getStreamSampleCt(NIDQ);
	//re
	return 1;
}

// fetch the NIDQ data and extract stimulus event time and label (if they exist)
t_ull StreamDataSocket::fetchEventInfo(int &eventLabel, t_ull lStartCt, OSSSpecificParams osParams) {
	// Compute how many samples needed to get to present time
	//lToGet is always zero
	
	efficientWait(20);
	t_ull lToGet = getStreamSampleCt(NIDQ, osParams) - lStartCt;
	lToGet = getStreamSampleCt(NIDQ, osParams) - lStartCt;
	//std::cout << lToGet << "\n";
	// Fetch NIDQ data (substream is 0 by convention)
	
	//m_mSGlxMutex.lock();
	t_ull lLatestCt;
	{
		std::unique_lock<std::mutex> lock(sglxMutex);
		lLatestCt = sglx_fetchLatest(m_sNidqBuffer, S, NIDQ, 0, lToGet, m_vNidqChannels) + lToGet;
	}
	//m_mSGlxMutex.unlock();
	// Search the data for a non zero and break out if found, identifying the stimulus event label
	
	for (int i = 0; i < m_sNidqBuffer.size(); i++) {
		if (m_sNidqBuffer[i]==2||m_sNidqBuffer[i]==4) {
			eventLabel = m_sNidqBuffer[i];
			
			std::cout << eventLabel;
			efficientWait(100);
			//std::cout << "TRIAL!";
			
			break;
			
		}
	}
	
	return lLatestCt;
}

void StreamDataSocket::setDigitalOut(int signal) { 
	static const int nBits = sizeof(short) * 8; // number bits in a byte = 8
	bool hiLo;
	std::string line;
	std::cout << "IN DIGITAL OUT";
	std::cout << signal;
	if (signal==0)
	{
		std::cout << "Hello 4";
		sglx_setDigitalOut(S, 1, "PXI1Slot2/port1/line1");
		efficientWait(100);
		sglx_setDigitalOut(S, 0, "PXI1Slot2/port1/line1");

		}
	if (signal==1)
	{
		std::cout << "Hello 2";
		sglx_setDigitalOut(S, 1, "PXI1Slot2/port1/line2");
		efficientWait(100);
		sglx_setDigitalOut(S, 0, "PXI1Slot2/port1/line2");
		
	}
	// feedbackSignal == 0 signifies setting nidq stream out back to 0

	/*std::string binaryFeedbackSignal = std::bitset<nBits>(signal).to_string();
	std::cout << binaryFeedbackSignal<<"\n";
	efficientWait(1000);
	for (int i = 0; i < nBits; i++) {
		hiLo = binaryFeedbackSignal[i] == '1' ? true : false;
		line = "PXI1Slot2/port1/line" + std::to_string(i + 1); // TODO It appeared that the line was 1 indexed (why there's a plus 1), but double check
		sglx_setDigitalOut(S, hiLo, "PXI1Slot2/port1/line1");
		
		
	}*/
}


bool StreamDataSocket::isRunning() {
	bool running;

	std::unique_lock<std::mutex> lock(sglxMutex);
	sglx_isRunning(running, S);
	return running;
}


bool StreamDataSocket::startRun() {
	if (!isRunning()) {
		///m_mSGlxMutex.lock();
		{
			std::unique_lock<std::mutex> lock(sglxMutex);
			sglx_startRun(S);
		}
		//m_mSGlxMutex.unlock();
		std::cout << "New run started..." << std::endl;

		Sleep(5000); // Wait 5s for startup
	}
	else
		std::cout << "... run already started." << std::endl;
	return true;
}


bool StreamDataSocket::stopRun() {
	if (isRunning()) {
		//m_mSGlxMutex.lock();
		{
			std::unique_lock<std::mutex> lock(sglxMutex);
			sglx_stopRun(S);
		}
		//m_mSGlxMutex.unlock();
		std::cout << "Run stopped..." << std::endl;
	}
	else
		std::cout << "... no run started." << std::endl;
	return true;
}


void StreamDataSocket::waitUntil(t_ull lWaitUntilCt, OSSSpecificParams osParams) {

	t_ull lLatestCt = getStreamSampleCt(IMEC, osParams);
	if (lLatestCt < lWaitUntilCt) {
		int iWaitTime = (lWaitUntilCt - lLatestCt) / (m_fImecSampRate / 1000); // ms
		efficientWait(iWaitTime);
	}
}