#include "SorterHelpers.h"

#include <stdio.h>
#include <string.h>
#include <regex>

#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#if _WIN32 || _WIN64
#include <windows.h>
#endif


// ------------------------------------------------------------------------------
//
// Name			: getTemplates()
//
// Description  : Get Template File Location
//
// ------------------------------------------------------------------------------
cnpy::NpyArray getTemplates( std::string templatesFile ) {
	cnpy::NpyArray aTemplates;
	try
	{	
		//Load data
		aTemplates = cnpy::npy_load(templatesFile);

		// ---- Check Template ----
		if (aTemplates.word_size != sizeof(float))
			throw std::runtime_error("loadTemplateFile: Wrong number format");
		if ( aTemplates.shape.size() != 3 && aTemplates.shape.size() != 2 )
			throw std::runtime_error("loadTemplateFile: Wrong dimensionality");
	} catch (std::exception& e) {
		std::cout << "Error while trying to load " << templatesFile << ": " << e.what() << std::endl;
		exit(0);
	}
	return aTemplates;
};


// ------------------------------------------------------------------------------
//
// Name			: getWhitening()
//
// Description  : Get Whitening file location
//
// ------------------------------------------------------------------------------
cnpy::NpyArray getWhitening( std::string whiteningFile) {
	cnpy::NpyArray aWhitening;
	try
	{

		aWhitening = cnpy::npy_load( whiteningFile );
		// ---- Check Template
		//if ( aWhitening.word_size != sizeof(double) )
		//	throw std::runtime_error("loadWhiteningFile: Wrong number format");
		if ( aWhitening.shape.size() != 2 )
			throw std::runtime_error("loadWhiteningFile: Wrong dimensionality");
		//if ( aWhitening.shape[0] != lC || aWhitening.shape[1] != lC ) // Uncomment when move to OnlineSpikes
			//throw std::runtime_error("loadWhiteningFile: Wrong size");
	} catch (std::exception& e) {
		std::cout << "Error while trying to load " << whiteningFile << ": " << e.what() << std::endl;
		exit(1);
	}
	return aWhitening;
};


// ------------------------------------------------------------------------------
//
// Name			: getChannelMap()
//
// Description  : Get Channel Map File Location
//
// ------------------------------------------------------------------------------
cnpy::NpyArray getChannelMap( std::string channelMapFile ) {
	cnpy::NpyArray aChannelMap;
	try
	{
		aChannelMap = cnpy::npy_load(channelMapFile);

		// ---- Check Template ----
		if ( aChannelMap.word_size != sizeof(int) )
			throw std::runtime_error("loadChannelMapFile: Wrong number format");
		if (aChannelMap.shape.size() > 1 && aChannelMap.shape[aChannelMap.shape.size() - 1] > 1)
			//std::cout << aChannelMap.shape.size() << "," << aChannelMap.shape[aChannelMap.shape.size() - 1] << std::endl;
			throw std::runtime_error("loadChannelMapFile: Wrong dimensionality");;

	} catch (std::exception& e) {
		std::cout << "Error while trying to load " << channelMapFile << ": " << e.what() << std::endl;
		exit(1);
	}
	return aChannelMap;
};

cnpy::NpyArray getTemplateMap(std::string templateMapFile) {
	cnpy::NpyArray aTemplateMap;
	try
	{
		aTemplateMap = cnpy::npy_load(templateMapFile);

		if (aTemplateMap.word_size != sizeof(int))
			throw std::runtime_error("templateMapFile: Wrong number format");
		if (aTemplateMap.shape.size() > 1 && aTemplateMap.shape[aTemplateMap.shape.size() - 1] > 1)
			throw std::runtime_error("templateMapFile: Wrong dimensionality");
	}
	catch (std::exception& e) {
		std::cout << "Error while trying to load " << templateMapFile << ": " << e.what() << std::endl;
		exit(EXIT_SUCCESS);
	}
	return aTemplateMap;
}

// ------------------------------------------------------------------------------
//
// Name			: InputAgent::setDevice()
//
// Description  : Set GPU Device for calculation
//
// ------------------------------------------------------------------------------
void setDevice( int gpuNumber, myCudnnConvolution *mCC ) {
	static const char *ptLabel = {"InputAgent::setDevice"};
	
	if (!mCC->selectDevice(gpuNumber)) {
		std::cout << "Error while trying to select device " << gpuNumber << std::endl;
		exit(1);
	}
}
