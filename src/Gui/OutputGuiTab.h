#ifndef OUTPUT_GUI_H_
#define OUTPUT_GUI_H_

#include <vector>
#include <deque>
#include <utility>
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
	void AddMatchScore(float score, long timeSamples, float sampRate);
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
	void plotMatchScore(float trainingBaseline);
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

	// --- Matching-pursuit match score, 30 s rolling median -----------------
	std::mutex          scoreMutex;
	std::deque<std::pair<long, float>> m_dqScoreWin;
	long                m_lLastScoreSampleCt = -1;
	long                m_lFirstScoreCt      = -1;  // first spike, so partial windows are suppressed
	long                m_lLastScoreSpikeCt  = -1;  // most recent scored spike
	std::vector<float>  m_vfScoreMedian;            // 30 s rolling median, one per second
	std::vector<float>  m_vfScoreTimeSec;           // matching timestamps
	float               m_fScoreCurrent  = 0.0f;    // latest rolling median

	float CurrentMatchScore(long nowSampleCt, float sampRate);
	int                 m_iMatchHistorySec = 300;    // seconds of history shown
protected:
	bool m_bSelected;
};

class OutputGuiTab {

public:
	//constructor and destructor
	OutputGuiTab(std::string tabName, std::string ossInputDir = "", float retrainThresholdUm = 0.0f);
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

	// Shared toggle for showing subset neurons in raster plot
	bool m_bFittoActive = false;
	std::vector<float> m_vfMatchBaseline;

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
	float matchPctForNeuron(int idx) const;
	void plotProcessTimes(const ImVec2 windowCenter, bool &showProcessTimes);
	void plotVRMS(const ImVec2 windowCenter, bool &showVRMS);
	void plotP2P(const ImVec2 windowCenter, bool &showP2P);
	void plotDriftTrace(const ImVec2 windowCenter, bool& showDrift);
	void plotMatchScores(const ImVec2 windowCenter, bool& showMatchScores);
	void loadDriftReference();
	void loadMatchBaseline();
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

	// Drift trace variables
	std::vector <float> m_vfDriftDepth;
	std::vector <float> m_vfDriftTimeSec;
	std::mutex driftMutex;
	long m_lastDriftUpdateCt = -1;

	// Offline Kilosort drift for the training recording, overlaid on the live trace
	std::string         m_sOssInputDir;
	std::vector<float>  m_vfKsDrift;
	std::vector<float>  m_vfKsDriftTimeSec;
	bool                m_bDriftRefLoaded = false;
	float               m_fKsTrainSec     = 0.0f;  // training duration, seconds
	bool                m_bDriftFollow    = true;  // auto-scroll x to the live trace

	// Drift retrain threshold
	float               m_fRetrainThresholdUm = 0.0f;

	// Population match-quality trace: median over units of (current / own baseline).
	std::mutex          popMatchMutex;
	std::vector<float>  m_vfPopMatchTimeSec;
	std::vector<float>  m_vfPopMatchPct;         
	int                 m_iPopContributing = 0;   // units with a frozen baseline
	int                 m_iPopBelow85      = 0;   // of those, how many under 85%
	int                 m_iPopSilent       = 0;   // have a baseline + fired before, no spike in the last window
	long                m_lLastPopSampleCt = -1;
	int                 m_iMatchHistorySec = 600; // seconds of history shown
	void                updatePopulationMatch();
};



#endif