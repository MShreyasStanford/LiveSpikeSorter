#include "mockStreamDataSocket.h"
#include <random>

MockStreamDataSocket::MockStreamDataSocket(std::string accquisitionHost, uint16 accquisitionPort, int substream, long lMaxSize, long lMinSize, float fImecSampRate, float fNiqdSampRate, int downSampling)
{
	m_lMaxSize = lMaxSize;
	m_lMinSize = lMinSize;
	m_lDownsampling = 1; // = downSampling;
	m_fImecSampRate = fImecSampRate;
	m_fNidqSampRate = fNiqdSampRate;

	m_lLatestCt = 0;
}

MockStreamDataSocket::~MockStreamDataSocket() {
	;
}

bool MockStreamDataSocket::startRun() {
	return true;
}

t_ull MockStreamDataSocket::getStreamSampleCt(int streamType) {
	return m_lLatestCt;
}

void MockStreamDataSocket::wait(t_ull lWaitUntilCt) {
	;
}

t_ull MockStreamDataSocket::fetchLatest_TC(float *fData, OSSSpecificParams osParams, t_ull lStartCt) {
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-20.0f, 20.0f);
	for (long i = 0; i < m_lMaxSize * osParams.lNChans; i++) {
		fData[i] = distribution(generator);
	}
	m_lLatestCt += m_lMaxSize;

	return m_lLatestCt, m_lMaxSize;
}