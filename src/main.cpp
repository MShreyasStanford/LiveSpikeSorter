#include <experimental/filesystem>

#include "SpikeSorter/OnlineSpikes.h"
#include "SpikeSorter/OnlineSpikesV2.h"
#include "SpikeSorter/ThresholdCrossing.h"
#include "Decoder/Decoder.h"
#include "Gui/Gui.h"

#include "Networking/inputParameters.h"
#include "Networking/NetworkHelpers.h"
#include "Networking/FragmentManager.h"

#include "SpikeSorter/dataSocket.h" // for decoder to access getStreamSampleCt()

#include <npp.h> // to get the number of GPU devices

#include <algorithm>

//Check if we are not in the .exe folder. A bit ugly, but it works
void exeFolderSetup() {
	bool bInProjFile = false;
	auto fPath = std::experimental::filesystem::current_path();

	for (auto & p : std::experimental::filesystem::directory_iterator(fPath)) {
		std::string test = p.path().string();
		if (test.substr(test.size() - 3) == "src") {
			bInProjFile = true;
			break;
		}
	}

	if (bInProjFile != true) {
		std::experimental::filesystem::current_path(fPath.parent_path().parent_path());
	}
}

void runOSSParallel(sockaddr_in mainAddr, InputParameters params, DataSocket* sharedSocket) {
	try {
		OnlineSpikesV2 oSpikeSorter(params, mainAddr, sharedSocket);
		oSpikeSorter.runSpikeSorting();
	}
	catch (const std::exception& e) {
		std::cerr << "Exception caught: " << e.what() << std::endl;
	}
}

void runSorter(sockaddr_in mainAddr, InputParameters params, DataSocket** &mNC) { // TODO does InputParameters need serialization now?
	// Initialize and run sorter object

	if (params.iSorterType == 0) {
		int nGPUsDetected;
		cudaGetDeviceCount(&nGPUsDetected);
		
		if (nGPUsDetected == 0) {
			throw std::runtime_error("No GPUs detected, cannot run OnlineSpikeSorter\n");
		}

		if (params.vSelectedDevices.size() == 0) {
			std::cout << "WARNING: No devices selected for OnlineSpikeSorter. Running on one GPU by default.\n";
			params.vSelectedDevices.push_back(0);
		}

		DataSocket* sharedSocket = new StreamDataSocket(params.sDataAccquisitionHost, 
			params.uDataAccquisitionPort, 
			params.iSubstream, 
			params.iMaxScanWindow, 
			params.iMinScanWindow, 
			params.fImecSamplingRate, 
			params.fNidqSamplingRate, 
			params.iDownsampling
		);

		// starts spikeglx
		sharedSocket->startRun();

		for (auto const& deviceIndex : params.vSelectedDevices) {
			InputParameters ossParams = params;
			ossParams.uSelectedDevice = deviceIndex;
			ossParams.sInputFolder = params.mapDeviceFilePaths[deviceIndex];
			ossParams.sOSSOutputFolder = params.mapOSSOutputFolders[deviceIndex];
			std::cout << "Setting OSS output folder to " << params.mapOSSOutputFolders[deviceIndex];
			std::thread ossThread = std::thread(runOSSParallel, mainAddr, ossParams, sharedSocket);
			ossThread.detach();
		}
	}
	else if (params.iSorterType == 1) {
		ThresholdCrossing tCrossing(params, mainAddr, mNC);
		tCrossing.runThresholdCrossing();
	}
	else {
		std::cout << "Sorter type " << params.iSorterType << " not supported." << std::endl;
	}
	
	// TODO Exit protocol
}

void runDecoder(Sock &mainServer, std::vector<sockaddr_in> sorterImecAddrs, std::vector<sockaddr_in> sorterNidqAddrs, InputParameters params, DataSocket** mNC) {
	// Blocks until receive connect message from gui, accquiring its address
	std::cout << "Main: Waiting to receive GUI addr to send to decoder..." << std::endl;
	sockaddr_in guiAddr = recvConnectMsg(&mainServer, _GUI);
	std::cout << "Main: GUI address received!" << std::endl;

	// Start up Decoder
	Decoder decoder(sorterImecAddrs, sorterNidqAddrs, guiAddr, params, mNC);
}

InputParameters parseCmdArgs(int argc, char* argv[]) {
	InputParameters cmdLineParams;

	int numSorters = -1;

	// set command line arguments as default GUI params
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "--n_gpus") {
			if (i + 1 >= argc) {
				std::cout << "Must supply integer after --n_gpus" << std::endl;
				exit(EXIT_SUCCESS);
			}

			int nGPUsDetected;
			cudaGetDeviceCount(&nGPUsDetected);

			numSorters = min(std::stoi(argv[i + 1]), nGPUsDetected);
			for (int j = 0; j < numSorters; j++) {
				cmdLineParams.vSelectedDevices.push_back(j);
			}

			if (numSorters < std::stoi(argv[i + 1]))
				std::cout << "WARNING: Requested " << argv[i + 1] << " GPUs/sorters, however, only " << nGPUsDetected 
				<< " GPUs detected on machine. Proceeding with only the first " << nGPUsDetected << " sorters passed in for arguments." << std::endl;
		}
		else if (arg == "--sdm_ip") {
			if (i + 1 >= argc) {
				std::cout << "Must supply IP address after --sdm_ip" << std::endl;
				exit(EXIT_SUCCESS);
			}
			
			cmdLineParams.sdmIP = argv[i + 1];
			std::cout << "Passed in stimulus display machine IP: " << cmdLineParams.sdmIP << std::endl;
		}
		else if (arg == "--sdm_port") {
			if (i + 1 >= argc) {
				std::cout << "Must supply port number after --sdm_port" << std::endl;
				exit(EXIT_SUCCESS);
			}

			cmdLineParams.sdmPort = std::stoi(argv[i + 1]);
			std::cout << "Passed in stimulus display machine port: " << cmdLineParams.sdmPort << std::endl;
		}
	}

	// second pass through for args that depend on some other args
	for (int i = 0; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--oss_input") {
			if (i + 1 >= argc) {
				std::cout << "Must supply " << numSorters << " paths after --oss_input" << std::endl;
				exit(EXIT_SUCCESS);
			}

			for (int j = 0; j < cmdLineParams.vSelectedDevices.size(); j++) {
				if (i + j + 1 >= argc) {
					std::cout << "Not enough parameters provided for --oss_input compared to --n_gpus. Setting paths to empty strings." << std::endl;
					cmdLineParams.mapDeviceFilePaths[cmdLineParams.vSelectedDevices[j]] = "";
					continue;
				}

				std::string val = argv[i + j + 1];
				if (val.length() >= 2 && val.substr(0, 2) == "--") {
					std::cout << "Not enough parameters provided for --oss_input compared to --n_gpus. Setting paths to empty strings." << std::endl;
					cmdLineParams.mapDeviceFilePaths[cmdLineParams.vSelectedDevices[j]] = "";
					continue;
				}

				cmdLineParams.mapDeviceFilePaths[cmdLineParams.vSelectedDevices[j]] = argv[i + j + 1];
			}
		}
		else if (arg == "--decoder_input") {
			if (i + 1 >= argc) {
				std::cout << "Must supply " << numSorters << " paths after --decoder_input" << std::endl;
				exit(EXIT_SUCCESS);
			}

			for (int j = 0; j < cmdLineParams.vSelectedDevices.size(); j++) {
				if (i + j + 1 >= argc) {
					std::cout << "Not enough parameters provided for --decoder_input compared to --n_gpus. Setting paths to empty strings." << std::endl;
					cmdLineParams.mapDecoderInputFolders[cmdLineParams.vSelectedDevices[j]] = "";
					continue;
				}

				std::string val = argv[i + j + 1];
				if (val.length() >= 2 && val.substr(0, 2) == "--") {
					std::cout << "Not enough parameters provided for --decoder_input compared to --n_gpus. Setting paths to empty strings." << std::endl;
					cmdLineParams.mapDecoderInputFolders[cmdLineParams.vSelectedDevices[j]] = "";
					continue;
				}

				cmdLineParams.mapDecoderInputFolders[cmdLineParams.vSelectedDevices[j]] = argv[i + j + 1];
			}
		}
		else if (arg == "--spikes_output") {
			if (i + 1 >= argc) {
				std::cout << "Must supply " << numSorters << " paths after --spikes_output" << std::endl;
				exit(EXIT_SUCCESS);
			}

			for (int j = 0; j < cmdLineParams.vSelectedDevices.size(); j++) {
				if (i + j + 1 >= argc) {
					std::cout << "Not enough parameters provided for --spikes_output compared to --n_gpus. Setting paths to empty strings." << std::endl;
					cmdLineParams.mapSpikeFiles[cmdLineParams.vSelectedDevices[j]] = "";
					continue;
				}

				std::string val = argv[i + j + 1];
				if (val.length() >= 2 && val.substr(0, 2) == "--") {
					std::cout << "Not enough parameters provided for --spikes_output compared to --n_gpus. Setting paths to empty strings." << std::endl;
					cmdLineParams.mapSpikeFiles[cmdLineParams.vSelectedDevices[j]] = "";
					continue;
				}

				cmdLineParams.mapSpikeFiles[cmdLineParams.vSelectedDevices[j]] = argv[i + j + 1];
			}
		}
		else if (arg == "--cuda_output_dir") {
			if (i + 1 >= argc) {
				std::cout << "Must supply " << numSorters << " paths after --cuda_output_dir" << std::endl;
				exit(EXIT_SUCCESS);
			}

			for (int j = 0; j < cmdLineParams.vSelectedDevices.size(); j++) {
				if (i + j + 1 >= argc) {
					std::cout << "Not enough parameters provided for --cuda_output_dir compared to --n_gpus. Setting paths to empty strings." << std::endl;
					cmdLineParams.mapOSSOutputFolders[cmdLineParams.vSelectedDevices[j]] = "";
					continue;
				}

				std::string val = argv[i + j + 1];
				if (val.length() >= 2 && val.substr(0, 2) == "--") {
					std::cout << "Not enough parameters provided for --cuda_output_dir compared to --n_gpus. Setting paths to empty strings." << std::endl;
					cmdLineParams.mapOSSOutputFolders[cmdLineParams.vSelectedDevices[j]] = "";
					continue;
				}

				cmdLineParams.mapOSSOutputFolders[cmdLineParams.vSelectedDevices[j]] = argv[i + j + 1];
			}
		}
	}
	return cmdLineParams;
}
// ------------------------------------------------------------------------------
//
// Name			: main()
//
// Description  : The entry point. Starts up the the spike sorter, the decoder, 
// and GUI and provides them with all of the inputs needed to start processing 
// and interacting with one another.
//
// ------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
	exeFolderSetup(); // TODO check if this is necessary

	// Parse command line arguments and then autopopulate the input GUI with them
	InputParameters cmdLineParams = parseCmdArgs(argc, argv);
	Gui gui(cmdLineParams);

	// Gather the final input arguments after the user is finished with the input GUI
	InputParameters params = gui.gatherInputParameters();

	// Start the main server and perform handshakes with spikesorter
	std::cout << "Starting Main Server." << std::endl;
	sockaddr_in mainAddr = getSetupAddr(_LOCAL_HOST, gui.getMasterPort());

	Sock mainServer(Sock::UDP);
	if (!mainServer.bind(gui.getMasterPort())) {
		// If crashing here, change the value for master port
		std::string err = mainServer.errorReason();
		fprintf(stderr, "Error binding: %s\n", err.c_str());
		return 1;
	}

	std::cout << "Main Server socket bound to port " << gui.getMasterPort()
		<< " and listening for clients." << std::endl;

	std::cout << "Main Server is expecting the spike sorter(s) to connect now." << std::endl;

	// Start up spikeSorter
	std::cout << "Starting up spikeSorter" << std::endl;
	DataSocket** mNC = new DataSocket*();
	std::thread sorterThread = std::thread(runSorter, mainAddr, params, std::ref(mNC));

	// Blocks until receive connect message from every spikesorter, accquiring their addresses
	std::vector<sockaddr_in> sorterImecAddrs;
	std::vector<sockaddr_in> sorterNidqAddrs;
	int numSorters = params.iSorterType == 0 ? params.vSelectedDevices.size() : 1;

	for (int i = 0; i < numSorters; i++) {
		sockaddr_in sorterImecAddr = recvConnectMsg(&mainServer, _SPIKE_SORTER_IMEC);
		sockaddr_in sorterNidqAddr = recvConnectMsg(&mainServer, _SPIKE_SORTER_NIDQ);
		std::cout << "Main Server received a spike sorter connection from " << inet_ntoa((sorterImecAddr).sin_addr) <<
			" port " << ntohs((sorterImecAddr).sin_port) << std::endl;
		sorterImecAddrs.push_back(sorterImecAddr);
		sorterNidqAddrs.push_back(sorterNidqAddr);
	}


	// Startup decoder
	std::vector<std::thread> decoderThreads;

	// one decoder per spike sorter, one spike sorter per spikeglx substream
	for (auto const& deviceIndex : params.vSelectedDevices) {
		InputParameters decoderParams = params;
		decoderParams.uSelectedDevice = deviceIndex;
		decoderParams.sInputFolder = params.mapDeviceFilePaths[deviceIndex];
		decoderParams.sSpikesFile = params.mapSpikeFiles[deviceIndex];
		decoderParams.sDecoderWorkFolder = params.mapDecoderInputFolders[deviceIndex];

		decoderThreads.push_back(std::thread(runDecoder, mainServer, sorterImecAddrs, sorterNidqAddrs, decoderParams, mNC));
	}

	// Start up GUI
	gui.plotOutputs(mainAddr, params.iMaxScanWindow, params.iAvgWindowTime, params.bIsDecoding);

	// Wait for all sorter thread to finish
	bool finishedMsg;
	mainServer.recvData(&finishedMsg, sizeof(bool));

	// Join all threads
	sorterThread.join();

	for (int i = 0; i < numSorters; i++) {
		decoderThreads[i].join();
	}
	
	return 0;
}