#include <ImGUI/imgui.h>
#include <ImGUI/imgui_stdlib.h>

#include "GuiHelpers.h"


void HelpMarker(const char* desc) {
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered()) {
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}

void clampedInputInt(const char* label, int* v, int low, int high) {
	ImGui::InputInt(label, v, 1, 5, 0);
	if (*v < low)
		*v = low;
	if (*v > high)
		*v = high;
}

void setHistogramBins(int &bins) {
	ImGui::SetNextItemWidth(100);
	ImGui::SliderInt("##Bins", &bins, 25, 100);
	ImGui::SameLine(); ImGui::Text("nBins");

}

void setRange(int &range, int low, int high) {
	ImGui::SetNextItemWidth(100);
	ImGui::SliderInt("##Range", &range, low, high);
	ImGui::SameLine(); ImGui::Text("Range");
}


