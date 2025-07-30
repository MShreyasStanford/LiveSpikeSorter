#pragma once

#include "OutputGuiTab.h"
#include "../Networking/inputParameters.h"
#include <vector>

class OutputGui
{
public:
	OutputGui(InputParameters cmdLineParams);
	~OutputGui();
	void setupOutput(sockaddr_in mainAddr, long maxScanWind, long spikeRateWindow, bool isDecoding);
	void Render(const ImVec2 windowCenter);
private:
	std::vector<OutputGuiTab*> tabs;

};

