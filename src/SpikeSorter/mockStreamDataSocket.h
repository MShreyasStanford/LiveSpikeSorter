#include <string>

#include "dataSocket.h"

typedef unsigned long long t_ull;
typedef unsigned short  uint16;

class MockStreamDataSocket : public DataSocket
{
public:
	MockStreamDataSocket(std::string accquisitionHost, uint16 accquisitionPort, int substream, long lMaxSize, long lMinSize, float fImecSampRate, float fNidqSampRate, int downSampling);
	~MockStreamDataSocket();

	//bool   connect();
	//void   disconnect();

	t_ull   getStreamSampleCt(int streamType);
	//t_ull   fetchLatest(float *fData, t_ull lStartCt = ULLONG_MAX);
	t_ull	fetchLatest_TC(float *fData, OSSSpecificParams osParams, t_ull lStartCt = ULLONG_MAX);
	//t_ull   fetchFromPlace(float *fData, t_ull lStartCt);
	//t_ull	initNidqStream();
	//t_ull	fetchEventInfo(int &eventLabel, t_ull lStartCt);

	bool   startRun();

	void   wait(t_ull lWaitUntilCt);

protected:
	long m_lLatestCt;
};