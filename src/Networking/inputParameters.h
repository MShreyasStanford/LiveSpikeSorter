#ifndef INPUTPARAMETERS_H
#define INPUTPARAMETERS_H

typedef unsigned short  uint16;

#include <vector>
#include <map>

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
								sdmIP{};

	uint16						uDataAccquisitionPort,
								uSelectedDevice;

	std::vector<uint16>			vSelectedDevices,
								vChannelSubset;

	std::map<uint16, std::string>	mapDeviceFilePaths,
								mapOSSOutputFolders,
								mapDecoderInputFolders,
								mapSpikeFiles;

	double						dTau,
								dThreshold,
								dRatioToMax;

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
								iTimeBehind,
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
								bSmallskip;

	uint16_t					sdmPort{};
};
#endif