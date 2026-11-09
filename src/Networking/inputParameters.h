#ifndef INPUTPARAMETERS_H
#define INPUTPARAMETERS_H

typedef unsigned short  uint16;

#include <string>
#include <vector>
#include <map>
#include <limits>

struct InputParameters {
	std::string					sInputFolder{},
								sImecFile{},
								sNidqFile{},
								sSpikesFile{},
								sLogFile{},
								sEventFile{},
								sDataAccquisitionHost{},
								sDecoderWorkFolder{},
								sDecoderInputFolder{},
								sOSSOutputFolder{},
								sdmIP{},
								sdmProcessorType{ "zscore" },
								sSpikeStreamAddr{};   // "host:port" for the public UDP spike stream; empty = off

	uint16						uDataAccquisitionPort{ 0 },
								uSelectedDevice{ 0 };

	std::vector<uint16>			vSelectedDevices,
								vChannelSubset;
	std::vector<long>			vSdmActivitySubset{};

	std::map<uint16, std::string>	mapDeviceFilePaths,
								mapOSSOutputFolders,
								mapDecoderInputFolders,
								mapSpikeFiles;

	// imec probe substream: substream N = imecN
	std::map<uint16, int>		mapDeviceSubstreams;

	double						dTau,
								dThreshold,
								dRatioToMax;

	int							sdmTriggerBinMs{ 50 };

	// Settings owned by processor
	std::map<std::string, std::string>	mapSdmParams;

	float						fImecSamplingRate,
								fNidqSamplingRate,
								fThresholdStd;

	int							iSubstream,
								iNidqRefreshRate,
								iMinScanWindow,
								iMaxScanWindow,
								iConvolutionTimes,
								iDownsampling,
								iMaxIts,
								iTimeBehind{ 0 },     // Allowed lag
								iAvgWindowTime,
								iRedundancy,
								iWindowLength,
								iBinLength,
								iWindowOffset,
								iSorterType,
								iNumTemplates;

	bool						bReadFromFile,
								bIsDecoding,
								bIsSendingFeedback,
								bSmallskip{ false };  // when lag exceeds iTimeBehind: skip only to the lag limit
	uint16_t					sdmPort{};

	// Skip the ImGui input window and use CLI-populated params directly
	bool						bSkipInputGui{ false };

	// For (live) (rigid) drift estimation
	bool						bDriftEstimation{ false };
	float						fDriftWindowSeconds{ 10.0f };
	float						fDriftMaxShiftUm{ 50.0f };
	// Suggestion threshold (for retraining)
	float						fDriftRetrainThresholdUm{ 50.0f };
};
#endif