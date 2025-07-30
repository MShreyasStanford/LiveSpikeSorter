#include <npp.h>

#include <ImGUI/imgui.h>
#include <ImGUI/imgui_stdlib.h>

#include "../Networking/NetworkHelpers.h"
#include "../Helpers/GuiHelpers.h"
#include "InputGui.h"
#include "tinyfiledialogs/tinyfiledialogs.h"
#include <algorithm>

InputGUI::InputGUI(InputParameters cmdLineParams)
	: computerJob(_GUI),
	masterHost(_LOCAL_HOST),
	masterPort(_DEFAULT_MASTER_PORT),
	iCCalc(150),
	iLCalc(50),
	iMCalc(82)
{
	/* -------- single‑value fields -------- */
	Params.sInputFolder =
		cmdLineParams.sInputFolder.empty()
		? std::string(_FILE_PATH) + _FOLDER "input\\"
		: cmdLineParams.sInputFolder;

	Params.sImecFile =
		cmdLineParams.sImecFile.empty()
		? std::string(_FILE_PATH) + _FOLDER "input\\data.bin"
		: cmdLineParams.sImecFile;

	Params.sNidqFile =
		cmdLineParams.sNidqFile.empty()
		? std::string(_FILE_PATH) + "nidq.bin"
		: cmdLineParams.sNidqFile;

	Params.sEventFile =
		cmdLineParams.sEventFile.empty()
		? std::string(_FILE_PATH) + "eventFile.txt"
		: cmdLineParams.sEventFile;

	Params.sDecoderInputFolder = cmdLineParams.sDecoderInputFolder;

	Params.sDecoderWorkFolder =
		cmdLineParams.sDecoderWorkFolder.empty()
		? std::string(_FILE_PATH) + _FOLDER "output\\"
		: cmdLineParams.sDecoderWorkFolder;

	Params.sSpikesFile =
		cmdLineParams.sSpikesFile.empty()
		? Params.sDecoderWorkFolder + "spikeOutput.txt"
		: cmdLineParams.sSpikesFile;

	Params.sLogFile =
		cmdLineParams.sLogFile.empty()
		? Params.sDecoderWorkFolder + "log.txt"
		: cmdLineParams.sLogFile;

	Params.sdmIP = cmdLineParams.sdmIP;
	Params.sdmPort = cmdLineParams.sdmPort;

	/* -------- device selection -------- */
	Params.vSelectedDevices = cmdLineParams.vSelectedDevices;
	if (Params.vSelectedDevices.empty())
		Params.vSelectedDevices.push_back(0);

	/* -------- per‑device maps -------- */
	Params.mapDeviceFilePaths = cmdLineParams.mapDeviceFilePaths;
	Params.mapDecoderInputFolders = cmdLineParams.mapDecoderInputFolders;
	Params.mapSpikeFiles = cmdLineParams.mapSpikeFiles;
	Params.mapOSSOutputFolders = cmdLineParams.mapOSSOutputFolders;

	/* Fill in missing keys (if any) */
	auto ensure_key = [&](auto& m, const std::string& def_val)
	{
		for (int dev : Params.vSelectedDevices)
			if (m.find(dev) == m.end())
				m[dev] = def_val;
	};

	ensure_key(Params.mapDeviceFilePaths, Params.sInputFolder);
	ensure_key(Params.mapDecoderInputFolders, Params.sDecoderInputFolder);
	ensure_key(Params.mapSpikeFiles, Params.sSpikesFile);
	ensure_key(Params.mapOSSOutputFolders, Params.sOSSOutputFolder);

	// TODO: remove the marked entries from InputParameters and remove any dependencies on them
	/* -------- fixed defaults -------- */
	Params.sDataAccquisitionHost = _LOCAL_HOST;
	Params.uDataAccquisitionPort = 4142;
	Params.fImecSamplingRate = 30000.f;
	Params.fNidqSamplingRate = 25000.f;
	Params.fThresholdStd = 3.f;
	Params.iSubstream = 0;
	Params.iNidqRefreshRate = 1;
	Params.iMinScanWindow = 82;
	Params.iMaxScanWindow = 1500;
	Params.dTau = 0.12; // remove, no longer needed
	Params.dThreshold = 1.0; // remove, no longer needed
	Params.dRatioToMax = 0.9; // remove, no longer needed
	Params.iConvolutionTimes = 2; // remove, no longer needed
	Params.iDownsampling = 1; // remove, no longer needed
	Params.iMaxIts = 150; // remove, no longer needed
	Params.iTimeBehind = 0;
	Params.iAvgWindowTime = 5000;
	Params.iRedundancy = 7;
	Params.iSorterType = 0;
	Params.iNumTemplates = 300;

	Params.bSmallskip = false;
	Params.bReadFromFile = false;
	Params.bIsDecoding = false;
	Params.bIsSendingFeedback = false;

	/* Decoder window/bin */
	Params.iWindowLength = 3000;
	Params.iBinLength = 30;
	Params.iWindowOffset = 4500;
}


InputGUI::~InputGUI() {
}


//A function to calculate the amount of memory that is used. THIS IS DEPRECATED AND IS NO LONGER ACCURATE.
// TODO: Do the math for this for OnlineSpikesV2.cu, nearly all memory allocated for it should be found in OnlineSpikesV2MemoryList.h.
// If someone actually wants to go and make this super extensible, I recommend using the X-macro defined in OnlineSpikesV2MemoryList.h
// to have an automatic memory calculator --- the calculation code will automatically update to reflect the addition or removal of memory
// fields added to OnlineSpikesV2MemoryList.h. Look at OnlineSpikesV2::allocateMemory() to get an idea on how one would do this
double InputGUI::CalcBytes(long L, long C, long M, long MaxWin, long MinWin, long SpikesExpected) {
	double Bytes = 0;
	unsigned long long N = MaxWin + MinWin;

	Bytes += C * M * L * 2 * sizeof(float);				//m_gfD and m_gfD2
	Bytes += C * N * 2 * sizeof(float);					//m_gfV and m_gfY
	Bytes += N * L * sizeof(float);						//m_gfX
	Bytes += (N + 2 * M) * L * sizeof(float);			//m_gfU
	Bytes += C * C * sizeof(float);						//m_gfW
	Bytes += MaxWin * C * sizeof(float);				//m_gfYW
	Bytes += C * sizeof(float);							//m_gfDC

	//GPUhelpers
	Bytes += 2 * (L * (N / 1024) + 1) * sizeof(float);	//my maxkernel
	Bytes += N * sizeof(float);							//myDCremover
	Bytes += C * sizeof(float);

	//cg
	Bytes += C * N * SpikesExpected * sizeof(float);	//m_cgfA
	Bytes += 2 * SpikesExpected * sizeof(float);		//m_fcgs and m_fcgd
	Bytes += N * C * sizeof(float);						//m_fcgq

	//Convert to MB
	Bytes = Bytes / 1'000'000;

	return Bytes;
}

void InputGUI::InputTextWithFileDialog(
	const char* label,
	std::string* str,
	const char* buttonLabel,
	const char *initDirectory,
	int numFilterPatterns,
	const char *filterPatterns[],
	int allowMultipleSelect,
	bool folderSelect = false
) {
	ImGui::PushItemWidth(733);
	ImGui::InputText(label, str); 
	ImGui::PopItemWidth();

	ImGui::SameLine();
	if (ImGui::Button(buttonLabel)) {
		char * selection;
		if (!folderSelect) // File Select
			selection = tinyfd_openFileDialog("Select File", initDirectory, numFilterPatterns, filterPatterns, NULL, allowMultipleSelect);
		else // Folder select
			selection = tinyfd_selectFolderDialog("Select Folder", initDirectory);
		
		if (selection != NULL && !folderSelect) 
			*str = selection;
		else if (selection != NULL && folderSelect)
			*str = std::strcat(selection, "\\");
	}
}

std::string InputGUI::getDeviceInfo(int deviceNumber) {
	std::string output;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, deviceNumber);
	output = "Device Number: " + std::to_string(deviceNumber) + "\n";
	output += "Device name: " + std::string(prop.name) + "\n";
	output += "Memory Clock Rate (KHz): " + std::to_string(prop.memoryClockRate) + "\n";
	output += "Memory Bus Width (bits): " + std::to_string(prop.memoryBusWidth) + "\n";
	float peakMemoryBandwidth = 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6;
	output += "Peak Memory Bandwidth (GB/s): " + std::to_string(peakMemoryBandwidth) + "\n";
	return output;
}

void InputGUI::gatherNetworkParameters() {
	
}

void InputGUI::gatherDataAccquisitionParameters() {
	ImGui::Text("Input Folder:"); ImGui::SameLine();
	HelpMarker("Folder containing the templates.npy, whitening_mat.npy, and channel_map.npy files.");
	InputTextWithFileDialog("##Folder", &Params.sInputFolder, "Select##folderButton", Params.sInputFolder.c_str(), NULL, NULL, NULL, true);

	ImGui::Text("Imec Sampling Rate (Hz):");
	ImGui::InputFloat("##ImecSampRate", &Params.fImecSamplingRate, 100, 1000, "%.6f");

	ImGui::Text("Nidq Sampling Rate (Hz):");
	ImGui::InputFloat("##NidqSampRate", &Params.fNidqSamplingRate, 100, 1000, "%.6f");

	ImGui::Text("Nidq Event Stimulus Check refresh rate (ms):"); ImGui::SameLine();
	HelpMarker("The program uses the Nidq digital line to signal event stimuli. This value dictates how long the program waits each time before checking the Nidq stream.");
	ImGui::InputInt("##NidqRefreshRate", &Params.iNidqRefreshRate, 1, 10);

	ImGui::Checkbox("Read from binary file instead of streaming neural data in real time via SpikeGLX:", &Params.bReadFromFile);

	if (Params.bReadFromFile) {
		ImGui::Text("Binary imec stream file:");
		InputTextWithFileDialog("##imecStream", &Params.sImecFile, "Select##imecStreamButton", Params.sInputFolder.c_str(), 1, binFilterPattern, 0);

		ImGui::Text("Binary nidq stream file:");
		InputTextWithFileDialog("##nidqStream", &Params.sNidqFile, "Select##nidqStreamButton", Params.sInputFolder.c_str(), 1, binFilterPattern, 0);
	}
	else {
		int temp1 = masterPort;
		ImGui::Text("Master Port:");
		ImGui::SameLine(); HelpMarker("The port number that the main server will bind to and listen to clients (the spike sorter/decoder) on");
		ImGui::InputInt("##MasterPort", &temp1, 1, 100, 0);
		masterPort = temp1;

		ImGui::Text("Data Acquission Machine Host number:");
		ImGui::SameLine(); HelpMarker("The IP address of the machine running SpikeGLX. If SpikeGLX is running on the SpikeSorter machine, leave host and port numbers as is.");
		ImGui::InputText("##Host", &Params.sDataAccquisitionHost, ImGuiInputTextFlags_CharsDecimal, 0, 0);

		int temp2 = Params.uDataAccquisitionPort;
		ImGui::Text("Data Accquistion Machine Port:");
		ImGui::InputInt("##Port", &temp2, 1, 100, 0);
		Params.uDataAccquisitionPort = temp2;

		ImGui::Text("Substream:"); ImGui::SameLine();
		HelpMarker("What imec-probe are you using? E.g. if using imec4, set substream as 4");
		ImGui::InputInt("##Substream", &Params.iSubstream);
	}
}

void InputGUI::gatherGPUParameters() {
	int nGPUsDetected;
	cudaGetDeviceCount(&nGPUsDetected);
	ImGui::Text("Select GPU to use: ");

	std::vector<uint16> newSelectedDevices;
	for (int i = 0; i < nGPUsDetected; i++) {
		bool isSelected = std::find(Params.vSelectedDevices.begin(), Params.vSelectedDevices.end(), i) != Params.vSelectedDevices.end();
		if (ImGui::Selectable(getDeviceInfo(i).c_str(), isSelected)) {
			if (!isSelected) {
				newSelectedDevices.push_back(i);
			}
		}
		else if (isSelected) {
			newSelectedDevices.push_back(i);
		}
	}

	Params.vSelectedDevices = newSelectedDevices;
}

void InputGUI::gatherSpikeSorterParameters() {
	ImGui::Text("Minimum Scanning Window:");
	ImGui::SameLine(); HelpMarker("The minimum scan window is the amount of samples the program takes from the previous batch. Recommended that it is the same as the amount of samples in a template.");
	ImGui::InputInt("##Minscanwin", &Params.iMinScanWindow, 1, 25);

	ImGui::Text("Maximum Scanning Window:");
	ImGui::SameLine(); HelpMarker("The maximum scan window is the amount of samples the program takes. The total size of the batch is Maximum + Minimum scan window.");
	ImGui::InputInt("##Maxscanwin", &Params.iMaxScanWindow, 50, 250);

	ImGui::Text("#Redundancy");
	ImGui::SameLine(); HelpMarker("The amount of extra memory redundancy is introduced. This number x maximum amount of iterations/batch is the amount of maximum spikes that is allowed.");
	ImGui::InputInt("##Redundancy", &Params.iRedundancy, 1, 5);

	ImGui::Text("Time allowed to be behind");
	ImGui::SameLine(); HelpMarker("The time (in ms) the onlinesorter is allowed to run behind. Normally the program, when it has a long batch, skips the next one. If this is set to a non-zero value the program instead of skipping a batch tries to keep up. Some batches are easier than others so there is a chance that it gets back to online. If it does get longer than the value specified it will start skipping batches. When a value of  > 100'000 gets specified it will always try to keep up and not skip any values.");
	ImGui::InputInt("##TimeBehind", &Params.iTimeBehind, 100, 500);

	ImGui::Text("Spike Rate Average Time");
	ImGui::SameLine(); HelpMarker("The amount of milliseconds the average spike rate is taken over. Has to be an integer");
	ImGui::InputInt("##SpikeRateAverageTime", &Params.iAvgWindowTime);

	ImGui::Checkbox("Small Skip", &Params.bSmallskip);
	ImGui::SameLine();	HelpMarker("If selected and time behind != 0, the program will do many small skips instead of one big one to the front.");

	/*if (ImGui::CollapsingHeader("GPU memory calculator")) {

		ImGui::TextWrapped("A calculator, which can be used to calculate the amount of memory that is currently needed. It is dependent on the amount of Templates, Channels, Amount of iterations/batch, maximum and minimum scan window and the redundancy.");

		//Inputs
		ImGui::InputInt("#Channels", &iCCalc, 1, 5);
		ImGui::InputInt("#Templates", &iLCalc, 1, 5);
		ImGui::InputInt("Samples/Template", &iMCalc, 1, 5);

		//Calculate Amount of bytes needed
		double Bytes = CalcBytes(iLCalc, iCCalc, iMCalc, Params.iMaxScanWindow, Params.iMinScanWindow, Params.iRedundancy * Params.iMaxIts);

		//Convert to GB's and add extra 10% for overhead
		double ShowBytes = 1.1 * Bytes / 1'000.;

		ImGui::Text("Minimum amount of memory needed: %.2f GB", ShowBytes);
	}*/
}

void InputGUI::gatherThresholdCrossingParameters() {

	ImGui::Text("Minimum Scanning Window:");
	ImGui::SameLine(); HelpMarker("The minimum scan window is the amount of samples the program takes from the previous batch. Recommended that it is the same as the amount of samples in a template.");
	ImGui::InputInt("##Minscanwin", &Params.iMinScanWindow, 1, 25);

	ImGui::Text("Maximum Scanning Window:");
	ImGui::SameLine(); HelpMarker("The maximum scan window is the amount of samples the program takes. The total size of the batch is Maximum + Minimum scan window.");
	ImGui::InputInt("##Maxscanwin", &Params.iMaxScanWindow, 50, 250);

	ImGui::Text("Number of Templates:");
	ImGui::SameLine(); HelpMarker("The number of templates to run threshold crossing on, starting from template #0.");
	ImGui::InputInt("##NumTemplates", &Params.iNumTemplates, 1, 10);

	ImGui::Text("Threshold in standard deviations:");
	ImGui::SameLine(); HelpMarker("The number of standard deviations away from the mean that we should denote as the threshold for spikes.");
	ImGui::InputFloat("##ThresholdStd", &Params.fThresholdStd, 0.1,1.0);
}

void  InputGUI::gatherSorterParameters() {
	int prevSorterType = Params.iSorterType;

	ImGui::Text("Spike sorting type:");
	ImGui::SameLine(); HelpMarker("Which strategy to use for determining spike rates, either Spike Sorter (which uses KiloSort algorithm), or Threshold Crossing (which determines spikes based on distance in standard deviations away from mean.");
	ImGui::RadioButton("Spike Sorter", &Params.iSorterType, 0); ImGui::SameLine();
	ImGui::RadioButton("Threshold Crossing", &Params.iSorterType, 1);

	if (Params.iSorterType == 0) {
		if (prevSorterType != Params.iSorterType) {
			Params.iMinScanWindow = 82;
		}
		gatherSpikeSorterParameters();
	}
	else if (Params.iSorterType == 1) {
		if (prevSorterType != Params.iSorterType) {
			Params.iMinScanWindow = 1500;
		}
		gatherThresholdCrossingParameters();
	}
}

void InputGUI::gatherParallelizedOSSInputs() {
	if (Params.vSelectedDevices.size() > 1) {
		for (size_t i = 0; i < Params.vSelectedDevices.size(); ++i) {
			std::string label = "Input Folder for Device " + std::to_string(Params.vSelectedDevices[i]) + ": ";
			ImGui::Text(label.c_str()); ImGui::SameLine();
			std::string &filePath = Params.mapDeviceFilePaths[Params.vSelectedDevices[i]];  // Ensure this map exists to store file paths per device

			std::string uniqueLabel = "##Folder" + std::to_string(i);
			std::string buttonLabel = "Select##FolderButton" + std::to_string(i);

			ImGui::PushItemWidth(400);
			ImGui::InputText(uniqueLabel.c_str(), &filePath);
			ImGui::PopItemWidth();
			ImGui::SameLine();
			if (ImGui::Button(buttonLabel.c_str())) {
				const char* selection = tinyfd_selectFolderDialog("Select Folder", filePath.c_str());
				if (selection != nullptr) {
					filePath = std::string(selection) + "\\";
				}
			}
			HelpMarker("Folder containing the templates.npy, whitening_mat.npy, and channel_map.npy files.");
		}
	}
}
void InputGUI::gatherDecoderParameters() {
	/* Currently log file is not used
	ImGui::Text("Log file:");
	ImGui::InputText("##LogFile", &Params.logFile);
	*/
	ImGui::Checkbox("Use real-time decoding.", &Params.bIsDecoding);
	ImGui::SameLine(); HelpMarker("To use the decoder, first run the spikesorter for 10-20 minutes with this unchecked to produce a file called spike_output.txt. Then, run the spikesorter again with this box checked. The decoder will use spike_output as training data to train a real time classifier.");
	if (!Params.bIsDecoding) {
		ImGui::Text("Spike Sorter output data file:");
		ImGui::SameLine(); HelpMarker("Once written, use this file as the decoder training data file.");
		ImGui::InputText("##SpikeFile", &Params.sSpikesFile);
	}
	else {
		ImGui::Text("Directory:");
		ImGui::SameLine(); HelpMarker("Set the base directory. This will auto-populate the training data file, output folder, and event file paths.");
		InputTextWithFileDialog("##Directory", &Params.sDecoderInputFolder, "Select##DirectoryButton", Params.sDecoderWorkFolder.c_str(), NULL, NULL, NULL, true);

		if (!Params.sDecoderInputFolder.empty()) {
			Params.sSpikesFile = Params.sDecoderInputFolder + "/spikeOutput.txt";
			Params.sDecoderWorkFolder = Params.sDecoderInputFolder + "/";
			Params.sEventFile = Params.sDecoderInputFolder + "/eventfile.txt";
		}

		ImGui::Text("Decoder training data file:");
		ImGui::SameLine(); HelpMarker("Choose a Spike Sorter output data file to use as decoder training data.");
		InputTextWithFileDialog("##dataFile", &Params.sSpikesFile, "Select##datafileButton", Params.sDecoderWorkFolder.c_str(), 1, txtFilterPattern, 0);

		ImGui::Text("Decoder output folder:");
		ImGui::SameLine(); HelpMarker("Folder where the Decoder will write output files (and temporary files).");
		InputTextWithFileDialog("##workFolder", &Params.sDecoderWorkFolder, "Select##workFolderButton", Params.sDecoderWorkFolder.c_str(), NULL, NULL, NULL, true);

		ImGui::Text("Stimulus Display training event file:");
		InputTextWithFileDialog("##EventFile", &Params.sEventFile, "Select##eventFileButton", _FILE_PATH, 1, txtFilterPattern, 0);

		ImGui::Text("Decoder window length:");
		ImGui::SameLine(); HelpMarker("The decoder window length determines how much time the decoder trains and predicts on. (9000 streamSampleCounts / 30000 samples per second = 300 ms)");
		ImGui::InputInt("##DecodeWindowLength", &Params.iWindowLength);

		ImGui::Text("Decoder bin length:");
		ImGui::SameLine(); HelpMarker("Every binLength stream sample counts, the windowLength long window shifts by binLength stream sample counts. (1500 streamSampleCounts / 30000 samples per second = 50 ms)");
		ImGui::InputInt("##DecodeBinLength", &Params.iBinLength);

		ImGui::Text("Decoder training window offset:");
		ImGui::SameLine(); HelpMarker("The decoder training window offset determines the number of samples offset from the event time to train on. The effective window becomes [eventTime + offset - windowLength, eventTime + offset]  (4500 streamSampleCounts / 30000 samples per second = 150 ms)");
		ImGui::InputInt("##DecodeWindowOffset", &Params.iWindowOffset);

		ImGui::Checkbox("Send feedback", &Params.bIsSendingFeedback);
	}
}


void InputGUI::gatherInputParameters(bool &finished, bool &isNetworking)
{
	// Input gpu, decoder, sorter parameters
	ImGui::SetNextWindowSize({ 800,800 });
	ImGui::Begin("OnlineSorter Input Interface", NULL, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
	ImGui::PushItemWidth(780); // Set widget (text inputs) width

	if (ImGui::CollapsingHeader("Data Accquisition Parameters")) 
		gatherDataAccquisitionParameters();

	if (ImGui::CollapsingHeader("GPU Parameters"))
		gatherGPUParameters();

	if (ImGui::CollapsingHeader("Sorter Parameters"))
		gatherSorterParameters();

	if (Params.vSelectedDevices.size() > 1 && Params.iSorterType == 0) {
		if (ImGui::CollapsingHeader("Parallelized Online Sorter Parameters"))
			gatherParallelizedOSSInputs();
	}

	if (ImGui::CollapsingHeader("Decoder Parameters"))
		gatherDecoderParameters();

	if (ImGui::Button("Start Online Sorting", { 785,75 }))
		finished = true;

	ImGui::End();
};

