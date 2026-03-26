#ifndef DECODER_H_
#define DECODER_H_

#include <mutex>
#include <vector>
#include <memory>
#include "BaseModel.h"
#include "dataBinner.h"
#include "SdmProcessor.h"
#include "../Networking/Sock.h"
#include "../Networking/sorterParameters.h"
#include "../Networking/onlineSpikesPayload.h"
#include "../Networking/inputParameters.h"
#include "../SpikeSorter/dataSocket.h"
#include "SKLearnModel.h"

class Decoder {
public:
	Decoder(
		std::vector<sockaddr_in> sorterImecAddrs, 
		std::vector<sockaddr_in> sorterNidqAddrs, 
		sockaddr_in guiAddr, 
		InputParameters params, 
		DataSocket** mNC);

	~Decoder();

private:
	// Thread functions
	std::thread eventReceiverThread;
	void spikeReceiver();
	void decode(long samplecount);

	// Variables for GUI
	sockaddr_in guiAddr;

	int16_t guiLabel; // guiLabel and eventLabel are different variables so reading in new eventStreamSampleCt doesn't change the true label on GUI
	long guiEventStreamSampleCt; // Same with eventStreamSampleCt
	int16_t predictLabel;
	int16_t nTrials;
	int16_t nCorrect;
	double confidence;

	long streamSampleCt;
	long recordingOffset;

	// Variables used if isDecoding is true
	std::unique_ptr<BaseModel> model;
	DataBinner fileDataBinner;
	DataBinner streamDataBinner;
	int windowLength;


	// Socket for communication with sorter and GUI
	sockaddr_in sorterNidqAddr;

	Sock m_imecSock; 
	Sock m_nidqSock;
	Sock m_sdmSock; // connected to stimulus display machine
	FragmentManager imecFm;
	FragmentManager nidqFm;

	std::vector<long> m_sdmActivitySubset;
	std::unique_ptr<SdmProcessor> m_sdmProcessor;
	int m_sdmTriggerBinMs;
	float m_sdmSamplingRateHz;

	bool isDecoding;
	bool readFromFile;
	bool isSendingFeedback;

	// File writing helper objects
	FileWriter 			m_fwOut;				// File writer to record spikes
	FileWriter 			m_fwEvent;				// File writer to record stimulus event times and labels
	FileWriter 			m_fwLog;				// File writer to record log
};

#endif /* DECODER_H_ */
