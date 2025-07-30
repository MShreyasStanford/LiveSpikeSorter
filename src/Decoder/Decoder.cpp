#include <thread>
#include "Decoder.h"
#include "../Networking/NetworkHelpers.h"
#include "../Helpers/TimeHelpers.h"
#include "SVMModel.h"
#include "RegressionModel.h"
#include "../Gui/gui.h"
#include <fstream>

typedef unsigned long long t_ull; // TODO for all stream counts use this type

Decoder::Decoder(std::vector<sockaddr_in> sorterImecAddrs, std::vector<sockaddr_in> sorterNidqAddrs, sockaddr_in guiAddr, InputParameters params, DataSocket** mNC)
	: m_imecSock(Sock::UDP)
	, m_nidqSock(Sock::UDP)
	, m_sdmSock(Sock::UDP)
	, imecFm(&m_imecSock)
	, nidqFm(&m_nidqSock)
	, sorterNidqAddr(sorterNidqAddr)
	, guiAddr(guiAddr)
	, fileDataBinner(params.iWindowLength, params.iBinLength, params.iWindowOffset)
	, streamDataBinner(params.iWindowLength, params.iBinLength, params.iWindowOffset)
	, windowLength(params.iWindowLength) // Should just access dataBinner's variable
	, isDecoding(params.bIsDecoding)
	, readFromFile(params.bReadFromFile)
	, isSendingFeedback(params.bIsSendingFeedback)
	, streamSampleCt(0)
	, guiEventStreamSampleCt(-1)
	, nTrials(0)
	, nCorrect(0)
	, predictLabel(-1)
	, guiLabel(-1)
{
	static const char *ptLabel = { "Decoder::Decoder" };

	// thread for assembling packets from sorter
	std::thread imecFmThread = imecFm.assemblerThread();
	imecFmThread.detach();

	// thread for listening for ACKs from GUI
	std::thread imecRetransThread = imecFm.retransmitterThread();
	imecRetransThread.detach();

	//  ---- Read in output and log file ----
	if (!isDecoding) {
		m_fwOut.FileLoad(params.sSpikesFile);
	}
	else {
		// Load file for storing binned spikes, to be used later by decoder
		std::string binnedSpikesFile = params.sDecoderWorkFolder + "binnedSpikes.txt";
		m_fwOut.FileLoad(binnedSpikesFile);

		// Read from spikeOutput.txt, eventfile.txt, and store results into binnedSpikes.txt
		fileDataBinner.readInSpikes(params.sSpikesFile.c_str(), params.sEventFile.c_str(), binnedSpikesFile.c_str(), &m_fwOut);
		m_fwOut.FileClose();

		// Initialize decoder model
		model = std::make_unique<SKLearnModel>(SKLearnModel::LogisticRegression);

		// Train the decoder model using binnedSpikes.txt, will produce scaledBinnedSpikes.txt as an intermediate file
		std::cout << "Training scikit-learn LogisticRegression model.." << std::endl;
		model->init(binnedSpikesFile, params.sDecoderWorkFolder);
		std::cout << "..trained model!" << std::endl;

		// Load the file where we will be writing decoder predictions
		m_fwOut.FileLoad(params.sDecoderWorkFolder + "predictions.txt");
	}

	m_fwLog.FileLoad(params.sLogFile);

	// Connect to sorter so it has the decoder addresses
	sendConnectMsg(&m_imecSock, sorterImecAddrs[params.uSelectedDevice], _DECODER_IMEC);
	sendConnectMsg(&m_imecSock, sorterNidqAddrs[params.uSelectedDevice], _DECODER_NIDQ);

	// Connect to stimulus display machine
	if (!m_sdmSock.connect(params.sdmIP, params.sdmPort))
		std::cerr << "Connection to stimulus display machine failed: " << m_sdmSock.errorReason() << std::endl;
	else {
		std::cout << "Successfully connected to stimulus display machine with address " << params.sdmIP << " at port " << params.sdmPort << std::endl;
	}

	// Start up eventReceiver and spikeReceiver (for exit protocol, could have spikeReceiver on a thread too
	spikeReceiver();
};

Decoder::~Decoder()
{
	m_fwOut.FileClose();
	m_fwEvent.FileClose();
	m_fwLog.FileClose();
};

void Decoder::decode(long samplecount) {
	std::map<long, double> dataWindow = streamDataBinner.getDataWindow();
	std::vector<double> probEstimates;

	// Use data window to decode
	probEstimates = model->predict(dataWindow, 0 , predictLabel);
	m_fwOut.WritePrediction(0, predictLabel, probEstimates, 0, samplecount + recordingOffset);
	std::cout << predictLabel << std::flush;

	// Send feedback
	if (isSendingFeedback) { // Could send other things besides the predict label if helpful.	
		m_nidqSock.sendData(&predictLabel, sizeof(int), sorterNidqAddr);
	}
}

void Decoder::spikeReceiver() {
	// Process spikes online and decode if needed
	OnlineSpikesPayload payload;
	std::string serializedPayload;
	char sdmBuf[1];

	// Receive SorterParams from main and redirect to GUI
	SorterParameters params = recvPayload<SorterParameters>(&imecFm);
	sendPayload(&imecFm, params, guiAddr);

	while (true) {
		payload = recvPayload<OnlineSpikesPayload>(&imecFm); // from sorter

		streamSampleCt = payload.streamSampleCt;
		recordingOffset = payload.recordingOffset;

		streamDataBinner.insert(payload.Times, payload.Templates);
		streamDataBinner.updateTime(streamSampleCt);

		// Send event time and label (for psth plots) and nTrials to gui 
		payload.eventStreamSampleCt = guiEventStreamSampleCt;
		payload.label = guiLabel;
		payload.nTrials = nTrials;

		if (isDecoding) {	// If decoding, send prediction variables 
			// Compute predictLabel and store it into payload
			decode(streamSampleCt);
			payload.predictLabel = predictLabel;

			// Send prediction to stimulus display machine
			sdmBuf[0] = (int8_t)predictLabel;
			m_sdmSock.sendData(sdmBuf, sizeof(int8_t));

			if (isSendingFeedback) { // Could send other things besides the predict label if helpful.	
				m_nidqSock.sendData(&predictLabel, sizeof(int), sorterNidqAddr);
			}
		}

		// Send spikesPayload to GUI
		sendPayload(&imecFm, payload, guiAddr);
	}
}
