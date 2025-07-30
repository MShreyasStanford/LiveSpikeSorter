#ifndef GUI_H_
#define GUI_H_

#include "../Networking/inputParameters.h"

#include "../Networking/Sock.h"
#include "InputGui.h"
#include "OutputGui.h"

#include "ImGui/imgui_impl_win32.h"
#include "ImGui/imgui_impl_dx12.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include <tchar.h>


struct FrameContext
{
	ID3D12CommandAllocator* CommandAllocator;
	UINT64                  FenceValue;
};


// Data // TODO see what can be moved into GUI
static int const						NUM_FRAMES_IN_FLIGHT = 3;
static FrameContext						g_frameContext[NUM_FRAMES_IN_FLIGHT] = {};
static UINT								g_frameIndex = 0;

static int const						NUM_BACK_BUFFERS = 3;
static ID3D12Device*					g_pd3dDevice = NULL;
static ID3D12DescriptorHeap*			g_pd3dRtvDescHeap = NULL;
static ID3D12DescriptorHeap*			g_pd3dSrvDescHeap = NULL;
static ID3D12CommandQueue*				g_pd3dCommandQueue = NULL;
static ID3D12GraphicsCommandList*		g_pd3dCommandList = NULL;
static ID3D12Fence*						g_fence = NULL;
static HANDLE							g_fenceEvent = NULL;
static UINT64							g_fenceLastSignaledValue = 0;
static IDXGISwapChain3*					g_pSwapChain = NULL;
static HANDLE							g_hSwapChainWaitableObject = NULL;
static ID3D12Resource*					g_mainRenderTargetResource[NUM_BACK_BUFFERS] = {};
static D3D12_CPU_DESCRIPTOR_HANDLE		g_mainRenderTargetDescriptor[NUM_BACK_BUFFERS] = {};

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
void WaitForLastSubmittedFrame();
FrameContext* WaitForNextFrameResources();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

static WNDCLASSEX wc;
static HWND hwnd;

class Gui {
private:
	InputGUI inputGUI;
	OutputGui outputGUI;

	// State
	bool finished;
	ImVec4 clear_color;
	ImGuiIO& io;

	// Helper functions
	void beginFrame();
	void endFrame(ImVec4 &clear_color, ImGuiIO &io);
	
public:
	Gui(InputParameters cmdLineParams);
	~Gui();

	InputParameters gatherInputParameters();
	void plotOutputs(sockaddr_in mainAddr, long m_lMaxScanWind, long m_lSpikeRateWindow, bool isDecoding);

	// Getters
	int &getComputerJob() { return inputGUI.getComputerJob(); }
	std::string &getMasterHost() { return inputGUI.getMasterHost(); }
	uint16 &getMasterPort() { return inputGUI.getMasterPort(); }

};

#endif // INPUT_GUI