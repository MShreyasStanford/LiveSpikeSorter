#ifndef SORTERHELPERS_H_
#define SORTERHELPERS_H_

#include <iostream>
#include <string>

#include "../External/CNPY/cnpy.h"
#include "dataSocket.h"
#include "myCudnnConvolution.h"

// =================================================================
// Retrieval functions
// =================================================================
cnpy::NpyArray getTemplates(std::string templatesFile);
cnpy::NpyArray getWhitening(std::string whiteningFile);
cnpy::NpyArray getChannelMap(std::string channelMapFile);
cnpy::NpyArray getTemplateMap(std::string templateMapFile);
void           setDevice(int gpuNumber, myCudnnConvolution *mCC);

#endif /* SORTERHELPERS_H_ */


