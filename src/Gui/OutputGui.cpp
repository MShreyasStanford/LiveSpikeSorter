#include "OutputGui.h"
#include <sstream>

OutputGui::OutputGui(InputParameters cmdLineParams)
{
	std::map<uint16, std::string>* filePaths = &cmdLineParams.mapDeviceFilePaths;
	for (auto itr = filePaths->begin(); itr != filePaths->end(); ++itr) {
		std::stringstream ss;
		ss << "Tab: " << itr->second;
		tabs.push_back(new OutputGuiTab(ss.str()));
	}
}

OutputGui::~OutputGui()
{
	for (const auto &tab : tabs) {
		delete tab;
	}
}

void OutputGui::setupOutput(sockaddr_in mainAddr, long maxScanWind, long spikeRateWindow, bool isDecoding)
{
	for (const auto &tab : tabs) {
		tab->setupOutput(mainAddr, maxScanWind, spikeRateWindow, isDecoding);
	}
}

void OutputGui::Render(const ImVec2 windowCenter)
{
	ImGuiWindowFlags windowFlags =
		ImGuiWindowFlags_NoDocking;       // optional: disallow moving the window

	ImGui::Begin("Main Output GUI", nullptr, windowFlags);
	if (ImGui::BeginTabBar("##FixedTabs")) {
		for (auto tab : tabs) {
			if (ImGui::BeginTabItem(tab->tabName.c_str())) {
				tab->Render(windowCenter);
				ImGui::EndTabItem();
			}
		}
		ImGui::EndTabBar();
	}
	ImGui::End();
}