#include <thread>
#include <atomic>
#include <unordered_set>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include "Decoder.h"
#include "ZScoreSdmProcessor.h"
#include "LogRegSdmProcessor.h"
#include "BinCountSdmProcessor.h"
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
	, m_sdmSock(params.sdmProcessorType == "bincounts" ? Sock::TCP : Sock::UDP)
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
	, m_sdmActivitySubset(params.vSdmActivitySubset)
	, m_sdmTriggerBinMs(params.sdmTriggerBinMs)
	, m_sdmSamplingRateHz(params.fImecSamplingRate)
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

	// Instantiate SDM processor by config
	if (params.sdmProcessorType == "bincounts") {
		m_sdmProcessor = std::make_unique<BinCountSdmProcessor>();
	} else if (params.sdmProcessorType == "logreg") {
		m_sdmProcessor = std::make_unique<LogRegSdmProcessor>();
	} else {
		m_sdmProcessor = std::make_unique<ZScoreSdmProcessor>();
	}
	m_sdmProcessor->init(params, m_sdmActivitySubset);

	// Connect to stimulus display machine
	const std::string sdmProto = (params.sdmProcessorType == "bincounts") ? "TCP" : "UDP";
	const std::string sdmIp = params.sdmIP.empty() ? std::string("192.168.1.1") : params.sdmIP;
	if (!m_sdmSock.connect(sdmIp, params.sdmPort))
		std::cerr << "SDM " << sdmProto << " setup failed: " << m_sdmSock.errorReason() << std::endl;
	else {
		std::cout << "SDM " << sdmProto << " destination set to " << sdmIp << ":" << params.sdmPort << std::endl;

		// BinCounts hello is deferred to spikeReceiver() where we know the
		// real template count from the sorter.  Send it now only for
		// non-bincounts processors (where sendHello is a no-op anyway).
		if (params.sdmProcessorType != "bincounts")
			m_sdmProcessor->sendHello(m_sdmSock);

		// For zscore/logreg, send the legacy 13-byte hello
		if (params.sdmProcessorType != "bincounts") {
			uint8_t sdmHello[13] = { 0 };
			sdmHello[0] = static_cast<uint8_t>(0);
			const float helloFloat = 0.0f;
			const uint64_t helloU64 = 0;
			std::memcpy(&sdmHello[1], &helloFloat, sizeof(float));
			std::memcpy(&sdmHello[5], &helloU64, sizeof(uint64_t));
			const uint sent = m_sdmSock.sendData(sdmHello, static_cast<uint>(sizeof(sdmHello)));
			if (sent == 0) {
				std::cerr << "SDM " << sdmProto << " hello send failed: " << m_sdmSock.errorReason() << std::endl;
			}
			else {
				std::cout << "SDM " << sdmProto << " hello sent (" << sent << " bytes)." << std::endl;
			}
		}
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

	std::unordered_set<long> sdmSubset;
	for (auto idx : m_sdmActivitySubset)
		sdmSubset.insert(idx);
	const bool useSdmSubset = !sdmSubset.empty();

	const long binSamples = std::max<long>(1, static_cast<long>(std::llround((static_cast<double>(m_sdmTriggerBinMs) / 1000.0) * static_cast<double>(m_sdmSamplingRateHz))));

	long currentBinIndex = -1;

	// Receive SorterParams from main and redirect to GUI
	SorterParameters sorterParams = recvPayload<SorterParameters>(&imecFm);
	sendPayload(&imecFm, sorterParams, guiAddr);

	// Now we know the real template count — update the processor and send hello
	m_sdmProcessor->setNumTemplates(sorterParams.m_lT);
	m_sdmProcessor->sendHello(m_sdmSock);

	while (true) {
		payload = recvPayload<OnlineSpikesPayload>(&imecFm); // from sorter

		streamSampleCt = payload.streamSampleCt;
		recordingOffset = payload.recordingOffset;

		streamDataBinner.insert(payload.Times, payload.Templates);
		streamDataBinner.updateTime(streamSampleCt);

		if (currentBinIndex < 0) {
			currentBinIndex = streamSampleCt / binSamples;
		}

		// Filter spikes to subset and feed to processor
		std::vector<long> sdmTimes;
		std::vector<long> sdmChannels;
		for (int i = 0; i < static_cast<int>(payload.Times.size()); i++) {
			if (!useSdmSubset || sdmSubset.find(payload.Templates[i]) != sdmSubset.end()) {
				sdmTimes.push_back(payload.Times[i]);
				sdmChannels.push_back(payload.Templates[i]);
			}
		}
		m_sdmProcessor->onSpikes(sdmTimes, sdmChannels, streamSampleCt);

		// Sliding-window processors send a packet every batch
		{
			const long long batchGlxSigned = static_cast<long long>(streamSampleCt) + static_cast<long long>(recordingOffset);
			const uint64_t batchGlxSampleCt = (batchGlxSigned > 0) ? static_cast<uint64_t>(batchGlxSigned) : 0ULL;
			m_sdmProcessor->onBatchComplete(m_sdmSock, batchGlxSampleCt, streamSampleCt);
		}

		// Process bin boundaries from spike times
		for (int i = 0; i < static_cast<int>(payload.Times.size()); i++) {
			const long binIdx = payload.Times[i] / binSamples;
			while (binIdx > currentBinIndex) {
				const long binEndSampleCt = (currentBinIndex + 1) * binSamples;
				const long long glxSampleCtSigned = static_cast<long long>(binEndSampleCt) + static_cast<long long>(recordingOffset);
				const uint64_t glxSampleCt = (glxSampleCtSigned > 0) ? static_cast<uint64_t>(glxSampleCtSigned) : 0ULL;

				m_sdmProcessor->sendPacket(m_sdmSock, glxSampleCt, binEndSampleCt);
				currentBinIndex++;
			}
		}

		// Flush remaining complete bins up to current stream time
		const long lastCompleteBinIndex = (streamSampleCt / binSamples) - 1;
		while (currentBinIndex <= lastCompleteBinIndex) {
			const long binEndSampleCt = (currentBinIndex + 1) * binSamples;
			const long long glxSampleCtSigned = static_cast<long long>(binEndSampleCt) + static_cast<long long>(recordingOffset);
			const uint64_t glxSampleCt = (glxSampleCtSigned > 0) ? static_cast<uint64_t>(glxSampleCtSigned) : 0ULL;

			m_sdmProcessor->sendPacket(m_sdmSock, glxSampleCt, binEndSampleCt);
			currentBinIndex++;
		}

		// Send event time and label (for psth plots) and nTrials to gui
		payload.eventStreamSampleCt = guiEventStreamSampleCt;
		payload.label = guiLabel;
		payload.nTrials = nTrials;

		if (isDecoding) {	// If decoding, send prediction variables
			// Compute predictLabel and store it into payload
			decode(streamSampleCt);
			payload.predictLabel = predictLabel;

			if (isSendingFeedback) { // Could send other things besides the predict label if helpful.
				m_nidqSock.sendData(&predictLabel, sizeof(int), sorterNidqAddr);
			}
		}

		// Send spikesPayload to GUI
		sendPayload(&imecFm, payload, guiAddr);
	}
}
