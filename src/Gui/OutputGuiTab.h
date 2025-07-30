#ifndef OUTPUT_GUI_H_
#define OUTPUT_GUI_H_

#include <vector>
#include <mutex>
#include <thread>

#include "../Networking/Sock.h"
#include "ImGui/imgui.h"
#include "../Networking/FragmentManager.h"

class OutputGuiTab; // TODO eventually separate neuron and OutputGUI into two separate files
class Neuron;

class Neuron {
public:
	Neuron(int number);
	~Neuron();

	int m_inumber;
	float m_SpikeRate;
	int m_iYchanPos;
	int m_iChannumber;

	std::mutex spikeTimeMutex;
	std::mutex spikeRateMutex;

	std::vector<long> m_vSpikeTime;
	std::vector<float> m_vfSpikeRate;
	std::vector<float> m_vfSpikeAmplitude;

	void AddSpike(long time);
	void AddSpikeRate();
	void AddSpikeAmplitude(float amp);
	float GetSpikeRate();
	void SetChanNum(int Channum);
	void Update(OutputGuiTab* GUI);
	long GetTotSpikeCount();
	void CalcSpikeRate(long *streamSampleCt, long TimeWindow, float SamplingRate);
	void SelectNeuron();
	void DeselectNeuron();
	bool IsSelected();
	void plotFR(float sampRate, int binSize);
	void plotAmplitude();
	void plotISI();
	void plotPSTH(std::vector<long> eventTimes, std::vector<int16_t> eventLabels, float sampRate, int bins, int binSize, int binSizeCts, int rangeCts, int negRange, int negRangeCts, int16_t label);
	void plotPSTHs(std::vector<long> eventCts, std::vector<int16_t> eventLabels, float sampRate);
	void plotAutoCorr();
	void plotCrossCorr(int History, OutputGuiTab* GUI);

	// This is to only show the data (so that you can more easily run it multithreaded)
	void plotCrossCorr(int History, std::vector<std::vector<float>>& crosscorr, std::vector<int>& NeuronNums);

	std::vector<float> m_vfSpikesPerBin;
	std::vector<float> AutoCorrelation(int start = 0, int end = 0);
	std::vector<int> CalcNeuronNums(std::vector<double> Channums, long lT, int ChanRange = 10);
	std::vector<std::vector<float>> CorrelateWithNeighbors(Neuron **Neurons, std::vector<int>& NeuronNums);

	void CorrelateWithNeighborsMult(Neuron **Neurons, std::vector<int> &NeuronNums, bool *done, std::vector<std::vector<float>> *output, int start = 0);

	static std::vector<float> Correlate(std::vector<float> const & f, std::vector<float> const & g, int start = 0);

	void CheckAndAddBin(const int BinSize, const int streamSampleCt);
	void AddBin(float BinVal);

	std::mutex ampMutex;              // guard m_vfSpikeAmplitude
	std::mutex binMutex;
protected:
	bool m_bSelected;
};

class OutputGuiTab {

public:
	//constructor and destructor
	OutputGuiTab(std::string tabName); // Default constructor
	~OutputGuiTab();

	//functions
	void setupOutput(sockaddr_in mainAddr, long m_lMaxScanWind, long m_lSpikeRateWindow, bool isDecoding);
	void Render(const ImVec2 windowCenter);

	//Sorter Settings
	bool isDecoding;
	long m_lT;
	long m_lM;
	long m_lN;
	long m_lC;
	long m_lAvgWindowTime;
	float m_fSampRate;
	int m_iMinNeuronIndex;
	std::string tabName;

	//sorter objects
	std::vector<double> m_dChanpos;

	// neuron indices
	std::vector<int> m_vNeuronIndices;
	//classes
	Neuron** m_bNeurons;

	std::vector<long> eventTimes;
	std::vector<int16_t> eventLabels;

private:

	//main thread function
	void UpdateEvents();

	//Functions
	void DrawImGUI(const ImVec2 windowCenter);
	void plotRaster(const ImVec2 windowCenter, bool &showRaster);
	void displayNeuronInfo(const ImVec2 windowCenter, bool &showNeuronInfo);
	void plotProcessTimes(const ImVec2 windowCenter, bool &showProcessTimes);
	void plotVRMS(const ImVec2 windowCenter, bool &showVRMS);
	void plotP2P(const ImVec2 windowCenter, bool &showP2P);
	void displayTrialInfo(const ImVec2 windowCenter, bool &showTrialInfo);
	void setMaxScanWindow(long m_lMaxScanWindow, float m_fSampRate);

	// Network socket used to receive payloads from the decoder
	Sock sock;
	FragmentManager fm;

	//Helpers
	bool m_bUpdating;

	//Extra threads
	std::thread m_tCalcing;

	// Value to keep track of current streamSampleCount (based on received payloads)
	long streamSampleCt;

	// guards for all the little history vectors and trial info
	std::mutex trialMutex;
	std::mutex eventMutex;
	std::mutex vrmsMutex;
	std::mutex p2pMutex;

	// Prediction variables
	int16_t predictLabel;
	int16_t label;
	int16_t nTrials;
	int16_t nCorrect;
	double confidence;

	// Object to prevent docking bugs
	ImGuiWindowClass plotWindowClass;

	// Realtime statistics variables
	std::vector<double> m_vdVRMS;
	std::vector<float>  m_vfP2P;
	std::vector<long>	m_vProcessTimes;
	int					maxScanWindow;

	std::mutex processingTimeMutex;
};



#endif