#ifndef DATA_SOCKET_H_
#define DATA_SOCKET_H_

#include <mutex>

#include "../../External/sglx/SglxApi.h"

#include "../NetClient/NetClient.h"
#include "../NetClient/socket.h"
#include "../Helpers/TimeHelpers.h"

typedef unsigned long long t_ull;

enum streamTypes { NIDQ, OBX, IMEC }; // No OBX support currently

// Parameters specific to each online spike sorter if parallelized
struct OSSSpecificParams {
	long lNChans;
	std::vector<int> vImecChannels;
	int substream;
};

class DataSocket
{
public:
	virtual bool   connect() { return false; };
	virtual void   disconnect() {};

	virtual t_ull   getStreamSampleCt(int streamType, OSSSpecificParams osParams) = 0;
	virtual t_ull   fetchLatest(float *fData, OSSSpecificParams osParams, t_ull lStartCt = ULLONG_MAX) = 0;
	virtual t_ull	fetchLatest_TC(float *fData, OSSSpecificParams osParams, t_ull lStartCt = ULLONG_MAX) = 0;
	virtual t_ull   fetchFromPlace(float *fData, OSSSpecificParams osParams, t_ull lStartCt) = 0;
	virtual t_ull	initNidqStream() = 0;
	virtual t_ull	fetchEventInfo(int &eventLabel, t_ull lStartCt, OSSSpecificParams osParams) = 0;
	virtual void	setDigitalOut(int signal = 0) {};

	virtual bool   isRunning() { return true; };
	virtual bool   startRun() { return true; };
	virtual bool   stopRun() { return true; };

	virtual void   waitUntil(t_ull lCt, OSSSpecificParams osParams) {};

protected:
	// Short vector data buffers
	std::vector<short>	m_sFetchBuffer;
	std::vector<short>	m_sNidqBuffer;

	// Max and min window size
	t_ull       m_lMaxSize,
				m_lMinSize;

	long		//m_lNChans, // Number of channels in Imec Probe
				m_lDownsampling; // Downsampling not implemented for FileDataSocket and probably not functional in StreamDataSocket

	float		m_fImecSampRate,
				m_fNidqSampRate;
};


class StreamDataSocket : public DataSocket {
public:
	StreamDataSocket(std::string accquisitionHost, uint16 accquisitionPort, int substream, long lMaxSize, long lMinSize, float fImecSampRate, float fNidqSampRate, int downSampling);
	~StreamDataSocket();

	bool   connect();
	void   disconnect();

	t_ull   getStreamSampleCt(int streamType, OSSSpecificParams osParams);
	t_ull   fetchLatest(float *fData, OSSSpecificParams osParams, t_ull lStartCt = ULLONG_MAX);
	t_ull	fetchLatest_TC(float *fData, OSSSpecificParams osParams, t_ull lStartCt = ULLONG_MAX);
	t_ull   fetchFromPlace(float *fData, OSSSpecificParams osParams, t_ull lStartCt);
	t_ull	initNidqStream();
	t_ull	fetchEventInfo(int &eventLabel, t_ull lStartCt, OSSSpecificParams osParams);
	void	setDigitalOut(int signal = 0);

	bool   isRunning();
	bool   startRun();
	bool   stopRun();

	void   waitUntil(t_ull lWaitUntilCt, OSSSpecificParams osParams);

	std::string getHost() { return m_sHost; };
	uint16      getPort() { return m_uPort; };

protected:
	t_sglxconn  S;
	std::string m_sHost;
	uint16      m_uPort;

	/* The two integer values select a data stream.
	streamType: {0=nidq, 1=obx, 2=imec-probe}.
	substream:   {0=nidq (if streamType=0), 0+=which Onebox or imec probe}.
	Examples (streamType, substream):
	(0, 0) = nidq.	// for nidq, substream is arbitrary but zero by convention
	(1, 4) = obx4.
	(2, 7) = imec7. */
	int m_iSubstream;

	std::vector<int>	//m_vImecChannels,
						m_vNidqChannels;

	// Mutex needed in streamDataSocket because streamDataSocket's Imec and Nidq streams cpp api calls could conflict
	std::mutex	m_mSGlxMutex;

	t_ull	fetch(std::vector<short> &data, t_sglxconn &S, int streamType, int substream, t_ull startSamp, int maxSamps, const std::vector<int> &channelSubset);


};
#endif /* DATA_SOCKET_H_ */