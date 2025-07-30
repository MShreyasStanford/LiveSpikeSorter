#ifndef GUIHELPERS_H_
#define GUIHELPERS_H_

// Helper function that creates little question mark with info when hovered over
void HelpMarker(const char* desc);

// Helper function that sets a permissble range for user int inputs. TODO if needed make function generic
void clampedInputInt(const char* label, int* v, int low, int high);

// Helper functions to set the bin type/count for ImGui histograms.
void setHistogramBins(int &bins);
void setRange(int &range, int low, int high);


#endif
