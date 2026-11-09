#include <ImGUI/imgui.h>
#include <ImGUI/imgui_stdlib.h>
#include <ImGUI/implot.h>
#include <ImGUI/implot_internal.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")

static void LoadIconTexture();
static void BuildDockLayout(ImGuiID dockspace_id);

ImGuiID g_spikeStatsDockNode = 0; 
ImGuiID g_rasterDockNode = 0;

#include "../Helpers/TimeHelpers.h"
#include "../Helpers/GuiHelpers.h"
#include "../Networking/sorterParameters.h"
#include "../Networking/onlineSpikesPayload.h"
#include "../Networking/NetworkHelpers.h"
#include "Gui.h"


#ifdef _DEBUG
#define DX12_ENABLE_DEBUG_LAYER
#endif

#ifdef DX12_ENABLE_DEBUG_LAYER
#include <dxgidebug.h>
#pragma comment(lib, "dxguid.lib")
#endif


LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// GUI font scale knobs
static constexpr float GUI_FONT_SIZE           = 30.0f; // What size font renders at (default is 26.0f)
static constexpr float GUI_FONT_GLOBAL_SCALE   = 1.0f;  // Multiplier to upscale GUI_FONT_SIZE
static constexpr float GUI_UI_SCALE            = 0.7f;  // Multiplies every style metric (padding, spacing, scrollbar width, frame heights)

Gui::Gui(InputParameters cmdLineParams):
	finished(false),
	clear_color(ImVec4(0.45f, 0.55f, 0.60f, 1.00f)),
	io((IMGUI_CHECKVERSION(), ImGui::CreateContext(), ImPlot::CreateContext(), ImGui::GetIO())),
	inputGUI(cmdLineParams),
	outputGUI(cmdLineParams) // Setup Dear ImGui context  
{
	// Create application window
		//ImGui_ImplWin32_EnableDpiAwareness();
	wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, _T("ImGui Example"), NULL };
	::RegisterClassEx(&wc);
	hwnd = ::CreateWindow(wc.lpszClassName, _T("Live Spikes"), WS_OVERLAPPEDWINDOW, 100, 100, 1280, 800, NULL, NULL, wc.hInstance, NULL);

	// Initialize Direct3D
	if (!CreateDeviceD3D(hwnd))
	{
		CleanupDeviceD3D();
		::UnregisterClass(wc.lpszClassName, wc.hInstance);
		exit(1);
	}

	// Show the window maximized
	::ShowWindow(hwnd, SW_SHOWMAXIMIZED);
	::UpdateWindow(hwnd);

	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
	//io.ConfigViewportsNoAutoMerge = true;
	//io.ConfigViewportsNoTaskBarIcon = true;

	// Setup Dear ImGui style
	// TODO: add custom dark style for LSS icon visually compatible with built-in Darkmode
	// ImGui::StyleColorsDark();
	ImGui::StyleColorsLight();

	// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
	ImGuiStyle& style = ImGui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style.WindowRounding = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}

	ImFontConfig fontCfg;
	fontCfg.SizePixels = GUI_FONT_SIZE;
	ImFont* uiFont = nullptr;
	const char* fontPath = "C:\\Windows\\Fonts\\segoeui.ttf";
	if (GetFileAttributesA(fontPath) != INVALID_FILE_ATTRIBUTES)
		uiFont = io.Fonts->AddFontFromFileTTF(fontPath, GUI_FONT_SIZE);
	if (!uiFont)
		io.Fonts->AddFontDefault(&fontCfg);   // bult-in font as fallback
	io.FontGlobalScale = GUI_FONT_GLOBAL_SCALE;
	style.ScaleAllSizes(GUI_UI_SCALE);

	// Setup Platform/Renderer backends
	ImGui_ImplWin32_Init(hwnd);
	ImGui_ImplDX12_Init(g_pd3dDevice, NUM_FRAMES_IN_FLIGHT,
		DXGI_FORMAT_R8G8B8A8_UNORM, g_pd3dSrvDescHeap,
		g_pd3dSrvDescHeap->GetCPUDescriptorHandleForHeapStart(),
		g_pd3dSrvDescHeap->GetGPUDescriptorHandleForHeapStart());

	LoadIconTexture();
};

Gui::~Gui() {
	WaitForLastSubmittedFrame();

	if (g_iconTexture) { g_iconTexture->Release(); g_iconTexture = nullptr; }

	// Cleanup
	ImGui_ImplDX12_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImPlot::DestroyContext();
	ImGui::DestroyContext();

	CleanupDeviceD3D();
	::DestroyWindow(hwnd);
	::UnregisterClass(wc.lpszClassName, wc.hInstance);
};


InputParameters Gui::gatherInputParameters()
{
	// Launched via the Python launcher (--no_input_gui): skip the ImGui input
	// window entirely and use the CLI-populated params as-is.
	if (inputGUI.getInputParameters().bSkipInputGui)
		return inputGUI.getInputParameters();

	// Some state
	bool isNetworking = true;

	// Set window to be in the middle
	ImVec2 windowCenter = ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
	ImGui::SetNextWindowPos(windowCenter, ImGuiCond_Always, ImVec2(0.5f, 0.5f));

	// InputOutput Main loop
	while (!finished)
	{
		// Handle poll events and start frame
		beginFrame();

		// InputGUI function
		inputGUI.gatherInputParameters(finished, isNetworking);

		endFrame(clear_color, io);
	}

	ImGui::SetNextWindowPos(windowCenter, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
	beginFrame();
	ImGui::Begin("Loading Window");
	ImGui::Text("Loading");
	ImGui::End();
	endFrame(clear_color, io);


	return inputGUI.getInputParameters();
}

void Gui::plotOutputs(sockaddr_in mainAddr, long m_lMaxScanWind, long m_lSpikeRateWindow, bool isDecoding)
{
	// For setting the window positions
	const ImVec2 windowCenter = ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
	
	outputGUI.setupOutput(mainAddr, m_lMaxScanWind, m_lSpikeRateWindow, isDecoding);

	// Pin ini next to the exe so layout persists regardless of working directory
	static char s_iniPath[MAX_PATH];
	{
		wchar_t buf[MAX_PATH];
		GetModuleFileNameW(NULL, buf, MAX_PATH);
		WideCharToMultiByte(CP_UTF8, 0, buf, -1, s_iniPath, MAX_PATH, NULL, NULL);
		char* slash = strrchr(s_iniPath, '\\');
		if (slash) strcpy(slash + 1, "imgui_lss.ini");
	}
	io.IniFilename = s_iniPath;

	// OutputGUI's Main loop
	while (true)
	{
		// Skip GPU work when the OS window is minimized — avoids DXGI waitable stall
		if (::IsIconic(hwnd)) {
			::Sleep(16);
			MSG msg;
			while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
				::TranslateMessage(&msg);
				::DispatchMessage(&msg);
				if (msg.message == WM_QUIT) exit(0);
			}
			continue;
		}

		// Handle poll events and start frame
		beginFrame();

		// Enable the dockspace
		ImGuiID dockspace_id = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

		// Build default layout once, only when no saved ini exists
		static bool layoutApplied = false;
		if (!layoutApplied)
		{
			ImGuiDockNode* root = ImGui::DockBuilderGetNode(dockspace_id);
			if (root != nullptr)
			{
				BuildDockLayout(dockspace_id);
				layoutApplied = true;
			}
		}

		// Recover dock node IDs when layout restored from ini
		if (g_rasterDockNode == 0)
		{
			ImGuiWindow* w = ImGui::FindWindowByName("Raster Plot");
			if (w && w->DockId != 0) g_rasterDockNode = w->DockId;
		}
		if (g_spikeStatsDockNode == 0)
		{
			for (int i = 0; i < 1024; ++i)
			{
				std::string name = "Neuron " + std::to_string(i) + " Stats";
				ImGuiWindow* w = ImGui::FindWindowByName(name.c_str());
				if (w && w->DockId != 0) { g_spikeStatsDockNode = w->DockId; break; }
			}
		}


		// outputGUI function
		outputGUI.Render(windowCenter);

		RenderIconWindow();

		endFrame(clear_color, io);
	}
}

// Helper functions
void Gui::beginFrame() {
	// Poll and handle messages (inputs, window resize, etc.)
		// See the WndProc() function below for our to dispatch events to the Win32 backend.
	MSG msg;
	while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
	{
		::TranslateMessage(&msg);
		::DispatchMessage(&msg);
		if (msg.message == WM_QUIT) {
			std::cout << "Exited out" << std::endl;
			exit(0);
		}
	}

	// Start the Dear ImGui frame
	ImGui_ImplDX12_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();
}

void Gui::endFrame(ImVec4 &clear_color, ImGuiIO &io) {
	// Rendering
	ImGui::Render();

	FrameContext* frameCtx = WaitForNextFrameResources();
	UINT backBufferIdx = g_pSwapChain->GetCurrentBackBufferIndex();
	frameCtx->CommandAllocator->Reset();

	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	barrier.Transition.pResource = g_mainRenderTargetResource[backBufferIdx];
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
	g_pd3dCommandList->Reset(frameCtx->CommandAllocator, NULL);
	g_pd3dCommandList->ResourceBarrier(1, &barrier);

	// Render Dear ImGui graphics
	const float clear_color_with_alpha[4] = { clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w };
	g_pd3dCommandList->ClearRenderTargetView(g_mainRenderTargetDescriptor[backBufferIdx], clear_color_with_alpha, 0, NULL);
	g_pd3dCommandList->OMSetRenderTargets(1, &g_mainRenderTargetDescriptor[backBufferIdx], FALSE, NULL);
	g_pd3dCommandList->SetDescriptorHeaps(1, &g_pd3dSrvDescHeap);
	ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), g_pd3dCommandList);
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
	g_pd3dCommandList->ResourceBarrier(1, &barrier);
	g_pd3dCommandList->Close();

	g_pd3dCommandQueue->ExecuteCommandLists(1, (ID3D12CommandList* const*)&g_pd3dCommandList);

	// Update and Render additional Platform Windows
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault(NULL, (void*)g_pd3dCommandList);
	}

	g_pSwapChain->Present(1, 0); // Present with vsync
	//g_pSwapChain->Present(0, 0); // Present without vsync

	UINT64 fenceValue = g_fenceLastSignaledValue + 1;
	g_pd3dCommandQueue->Signal(g_fence, fenceValue);
	g_fenceLastSignaledValue = fenceValue;
	frameCtx->FenceValue = fenceValue;
}

// ---------------------------------------------------------------------------
// Icon helpers
// ---------------------------------------------------------------------------

// First-run default layout configuration
// These values only apply when no imgui_lss.ini exists next to the exe.
// Delete imgui_lss.ini to force a reset to these defaults.
//
// Layout produced by the values below:
//
//  |<----------- LEFT_COL (55%) --------->|<---- right (45%) ---->|
//  +--------------------------------------+-----------------------+
//  |                                      |                       |
//  |         Raster Plot                  |   Processing Times    |
//  |       (LEFT_RASTER: 75% tall)        | (RIGHT_TOP: 50% tall) |
//  |                                      +-----------------------+
//  +-------------------+------------------+                       |
//  |   LSS Icon        |   Neurons        |   Neuron N Stats      |
//  | (ICON_DISPLAY_W)  |   (remainder)    | (RIGHT_TOP: 50% tall) |
//  +-------------------+------------------+-----------------------+
//
// To move a window to a different slot, swap its string in BuildDockLayout below.
// =============================================================================

// Left column width as a fraction of the total screen width (0–1)
static constexpr float LAYOUT_LEFT_COL_FRAC    = 0.55f;
// Raster plot height as a fraction of the left column height (0–1)
static constexpr float LAYOUT_LEFT_RASTER_FRAC = 0.75f;
// Processing-time panel height as a fraction of the right column height (0–1)
static constexpr float LAYOUT_RIGHT_TOP_FRAC   = 0.50f;
// Icon panel width in pixels — drives the icon/neurons split in the bottom strip
static constexpr float ICON_DISPLAY_W          = 300.0f;
// Icon panel height in pixels (floating fallback only; docked height follows the split)
static constexpr float ICON_DISPLAY_H          = 300.0f;
// Vertical placement of the icon in its panel (0-1 where 0.5 is center)
static constexpr float ICON_VERTICAL_FRAC      = 0.99f;

static void LoadIconTexture()
{
	// Path: <repo>\src\Gui\LSS_icon.jpg
	wchar_t exePath[MAX_PATH];
	GetModuleFileNameW(NULL, exePath, MAX_PATH);
	std::wstring path(exePath);
	for (int i = 0; i < 3; ++i)
		path = path.substr(0, path.find_last_of(L'\\'));
	path += L"\\src\\Gui\\LSS_icon.jpg";

	CoInitializeEx(NULL, COINIT_MULTITHREADED);

	IWICImagingFactory* wicFactory = nullptr;
	if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
		IID_PPV_ARGS(&wicFactory)))) return;

	IWICBitmapDecoder* decoder = nullptr;
	if (FAILED(wicFactory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
		WICDecodeMetadataCacheOnLoad, &decoder))) { wicFactory->Release(); return; }

	IWICBitmapFrameDecode* frame = nullptr;
	decoder->GetFrame(0, &frame);

	IWICFormatConverter* converter = nullptr;
	wicFactory->CreateFormatConverter(&converter);
	converter->Initialize(frame, GUID_WICPixelFormat32bppRGBA,
		WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);

	UINT w, h;
	converter->GetSize(&w, &h);
	g_iconWidth = (int)w; g_iconHeight = (int)h;

	UINT srcPitch = w * 4;
	std::vector<BYTE> pixels(srcPitch * h);
	converter->CopyPixels(nullptr, srcPitch, srcPitch * h, pixels.data());
	converter->Release(); frame->Release(); decoder->Release(); wicFactory->Release();

	D3D12_RESOURCE_DESC texDesc = {};
	texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	texDesc.Width = w; texDesc.Height = h;
	texDesc.DepthOrArraySize = 1; texDesc.MipLevels = 1;
	texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	texDesc.SampleDesc.Count = 1; texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	D3D12_HEAP_PROPERTIES defaultHeap = { D3D12_HEAP_TYPE_DEFAULT };
	if (FAILED(g_pd3dDevice->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
		&texDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_iconTexture))))
	{
		std::cerr << "[Icon] CreateCommittedResource (texture) failed\n";
		return;
	}

	UINT64 uploadPitch = (srcPitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u)
		& ~(UINT64)(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1u);
	D3D12_RESOURCE_DESC uploadDesc = {};
	uploadDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	uploadDesc.Width = uploadPitch * h; uploadDesc.Height = 1;
	uploadDesc.DepthOrArraySize = 1; uploadDesc.MipLevels = 1;
	uploadDesc.Format = DXGI_FORMAT_UNKNOWN; uploadDesc.SampleDesc.Count = 1;
	uploadDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	D3D12_HEAP_PROPERTIES uploadHeap = { D3D12_HEAP_TYPE_UPLOAD };
	ID3D12Resource* uploadBuf = nullptr;
	if (FAILED(g_pd3dDevice->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE,
		&uploadDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf))))
	{
		std::cerr << "[Icon] CreateCommittedResource (upload) failed\n";
		g_iconTexture->Release(); g_iconTexture = nullptr;
		return;
	}

	void* mapped = nullptr; D3D12_RANGE readRange = { 0, 0 };
	if (FAILED(uploadBuf->Map(0, &readRange, &mapped)))
	{
		std::cerr << "[Icon] Map failed\n";
		uploadBuf->Release(); g_iconTexture->Release(); g_iconTexture = nullptr;
		return;
	}
	for (UINT y = 0; y < h; ++y)
		memcpy((BYTE*)mapped + y * uploadPitch, pixels.data() + y * srcPitch, srcPitch);
	uploadBuf->Unmap(0, nullptr);

	ID3D12CommandAllocator* cmdAlloc = nullptr;
	ID3D12GraphicsCommandList* cmdList = nullptr;
	if (FAILED(g_pd3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAlloc))))
	{
		std::cerr << "[Icon] CreateCommandAllocator failed\n";
		uploadBuf->Release(); g_iconTexture->Release(); g_iconTexture = nullptr;
		return;
	}
	if (FAILED(g_pd3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
		cmdAlloc, nullptr, IID_PPV_ARGS(&cmdList))))
	{
		std::cerr << "[Icon] CreateCommandList failed\n";
		cmdAlloc->Release(); uploadBuf->Release();
		g_iconTexture->Release(); g_iconTexture = nullptr;
		return;
	}

	D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
	dst.pResource = g_iconTexture; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
	src.pResource = uploadBuf; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
	src.PlacedFootprint.Footprint = { DXGI_FORMAT_R8G8B8A8_UNORM, w, h, 1, (UINT)uploadPitch };
	cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

	D3D12_RESOURCE_BARRIER barrier = {};
	barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barrier.Transition.pResource = g_iconTexture;
	barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
	cmdList->ResourceBarrier(1, &barrier);
	cmdList->Close();

	ID3D12CommandList* lists[] = { cmdList };
	g_pd3dCommandQueue->ExecuteCommandLists(1, lists);
	ID3D12Fence* fence = nullptr; HANDLE evt = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	g_pd3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
	g_pd3dCommandQueue->Signal(fence, 1);
	if (fence->GetCompletedValue() < 1) { fence->SetEventOnCompletion(1, evt); WaitForSingleObject(evt, INFINITE); }
	CloseHandle(evt); fence->Release(); cmdList->Release(); cmdAlloc->Release(); uploadBuf->Release();

	SIZE_T incr = g_pd3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CPU_DESCRIPTOR_HANDLE srvCpu = g_pd3dSrvDescHeap->GetCPUDescriptorHandleForHeapStart();
	srvCpu.ptr += incr;
	g_iconSrvGpuHandle = g_pd3dSrvDescHeap->GetGPUDescriptorHandleForHeapStart();
	g_iconSrvGpuHandle.ptr += incr;
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels = 1;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	g_pd3dDevice->CreateShaderResourceView(g_iconTexture, &srvDesc, srvCpu);
}

static void BuildDockLayout(ImGuiID dockspace_id)
{
	ImGuiViewport* vp = ImGui::GetMainViewport();
	ImGui::DockBuilderRemoveNode(dockspace_id);
	ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
	ImGui::DockBuilderSetNodeSize(dockspace_id, vp->Size);

	// Step 1: split screen left / right
	ImGuiID left, right;
	ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, LAYOUT_LEFT_COL_FRAC, &left, &right);

	// Step 2: split left column top (raster) / bottom strip (icon + neurons)
	ImGuiID top_left, bot_left;
	ImGui::DockBuilderSplitNode(left, ImGuiDir_Up, LAYOUT_LEFT_RASTER_FRAC, &top_left, &bot_left);

	// Step 3: split bottom strip left (icon) / right (neurons)
	ImGuiID icon_node, neurons_node;
	float iconRatio = ImClamp(ICON_DISPLAY_W / (vp->Size.x * LAYOUT_LEFT_COL_FRAC), 0.1f, 0.5f);
	ImGui::DockBuilderSplitNode(bot_left, ImGuiDir_Left, iconRatio, &icon_node, &neurons_node);

	// Step 4: split right column top (processing times) / bottom (spike stats)
	ImGuiID top_right, bot_right;
	ImGui::DockBuilderSplitNode(right, ImGuiDir_Up, LAYOUT_RIGHT_TOP_FRAC, &top_right, &bot_right);

	// Assign windows to slots — change the string to move a window elsewhere
	ImGui::DockBuilderDockWindow("Raster Plot",                  top_left);
	ImGui::DockBuilderDockWindow("Plot Menu",                    top_left);
	ImGui::DockBuilderDockWindow("LSS Icon",                     icon_node);
	ImGui::DockBuilderDockWindow("Neurons",                      neurons_node);
	ImGui::DockBuilderDockWindow("Processing time distribution", top_right);
	ImGui::DockBuilderDockWindow("Drift trace",                  top_right);
	ImGui::DockBuilderDockWindow("Match quality",                top_right);
	ImGui::DockBuilderFinish(dockspace_id);

	g_rasterDockNode = top_left;
	g_spikeStatsDockNode = bot_right;
}

void Gui::RenderIconWindow()
{
	if (!g_iconTexture) return;

	const float aspect = (float)g_iconWidth / (float)g_iconHeight;
	ImGui::SetNextWindowSize(ImVec2(ICON_DISPLAY_W, ICON_DISPLAY_H), ImGuiCond_FirstUseEver);
	
	ImGuiWindowClass iconClass;
	iconClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar;
	ImGui::SetNextWindowClass(&iconClass);
	
	ImGui::Begin("LSS Icon", nullptr,
		ImGuiWindowFlags_NoTitleBar        |
		ImGuiWindowFlags_NoScrollbar       |
		ImGuiWindowFlags_NoScrollWithMouse |
		ImGuiWindowFlags_NoBringToFrontOnFocus);

	ImVec2 start = ImGui::GetCursorPos();
	ImVec2 avail = ImGui::GetContentRegionAvail();
	float drawW = avail.x, drawH = drawW / aspect;
	if (drawH > avail.y) { drawH = avail.y; drawW = drawH * aspect; }
	//ImGui::SetCursorPos(ImVec2((avail.x - drawW) * 0.5f, (avail.y - drawH) * 0.5f));
	ImGui::SetCursorPos(ImVec2(start.x + (avail.x - drawW) * 0.5f, start.y + (avail.y - drawH) * ICON_VERTICAL_FRAC));

	ImGui::Image((ImTextureID)g_iconSrvGpuHandle.ptr, ImVec2(drawW, drawH));
	ImGui::End();
}

// ---------------------------------------------------------------------------

bool CreateDeviceD3D(HWND hWnd)
{
	// Setup swap chain
	DXGI_SWAP_CHAIN_DESC1 sd;
	{
		ZeroMemory(&sd, sizeof(sd));
		sd.BufferCount = NUM_BACK_BUFFERS;
		sd.Width = 0;
		sd.Height = 0;
		sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
		sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		sd.SampleDesc.Count = 1;
		sd.SampleDesc.Quality = 0;
		sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		sd.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
		sd.Scaling = DXGI_SCALING_STRETCH;
		sd.Stereo = FALSE;
	}

	// [DEBUG] Enable debug interface
#ifdef DX12_ENABLE_DEBUG_LAYER
	ID3D12Debug* pdx12Debug = NULL;
	if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pdx12Debug))))
		pdx12Debug->EnableDebugLayer();
#endif

	// Create device
	D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
	if (D3D12CreateDevice(NULL, featureLevel, IID_PPV_ARGS(&g_pd3dDevice)) != S_OK)
		return false;

	// [DEBUG] Setup debug interface to break on any warnings/errors
#ifdef DX12_ENABLE_DEBUG_LAYER
	if (pdx12Debug != NULL)
	{
		ID3D12InfoQueue* pInfoQueue = NULL;
		g_pd3dDevice->QueryInterface(IID_PPV_ARGS(&pInfoQueue));
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
		pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, true);
		pInfoQueue->Release();
		pdx12Debug->Release();
	}
#endif

	{
		D3D12_DESCRIPTOR_HEAP_DESC desc = {};
		desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
		desc.NumDescriptors = NUM_BACK_BUFFERS;
		desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		desc.NodeMask = 1;
		if (g_pd3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&g_pd3dRtvDescHeap)) != S_OK)
			return false;

		SIZE_T rtvDescriptorSize = g_pd3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = g_pd3dRtvDescHeap->GetCPUDescriptorHandleForHeapStart();
		for (UINT i = 0; i < NUM_BACK_BUFFERS; i++)
		{
			g_mainRenderTargetDescriptor[i] = rtvHandle;
			rtvHandle.ptr += rtvDescriptorSize;
		}
	}

	{
		D3D12_DESCRIPTOR_HEAP_DESC desc = {};
		desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		desc.NumDescriptors = 2; // slot 0: ImGui font, slot 1: icon texture
		desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		if (g_pd3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&g_pd3dSrvDescHeap)) != S_OK)
			return false;
	}

	{
		D3D12_COMMAND_QUEUE_DESC desc = {};
		desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
		desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
		desc.NodeMask = 1;
		if (g_pd3dDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(&g_pd3dCommandQueue)) != S_OK)
			return false;
	}

	for (UINT i = 0; i < NUM_FRAMES_IN_FLIGHT; i++)
		if (g_pd3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_frameContext[i].CommandAllocator)) != S_OK)
			return false;

	if (g_pd3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_frameContext[0].CommandAllocator, NULL, IID_PPV_ARGS(&g_pd3dCommandList)) != S_OK ||
		g_pd3dCommandList->Close() != S_OK)
		return false;

	if (g_pd3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)) != S_OK)
		return false;

	g_fenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	if (g_fenceEvent == NULL)
		return false;

	{
		IDXGIFactory4* dxgiFactory = NULL;
		IDXGISwapChain1* swapChain1 = NULL;
		if (CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)) != S_OK)
			return false;
		if (dxgiFactory->CreateSwapChainForHwnd(g_pd3dCommandQueue, hWnd, &sd, NULL, NULL, &swapChain1) != S_OK)
			return false;
		if (swapChain1->QueryInterface(IID_PPV_ARGS(&g_pSwapChain)) != S_OK)
			return false;
		swapChain1->Release();
		dxgiFactory->Release();
		g_pSwapChain->SetMaximumFrameLatency(NUM_BACK_BUFFERS);
		g_hSwapChainWaitableObject = g_pSwapChain->GetFrameLatencyWaitableObject();
	}

	CreateRenderTarget();
	return true;
}

void CleanupDeviceD3D()
{
	CleanupRenderTarget();
	if (g_pSwapChain) { g_pSwapChain->SetFullscreenState(false, NULL); g_pSwapChain->Release(); g_pSwapChain = NULL; }
	if (g_hSwapChainWaitableObject != NULL) { CloseHandle(g_hSwapChainWaitableObject); }
	for (UINT i = 0; i < NUM_FRAMES_IN_FLIGHT; i++)
		if (g_frameContext[i].CommandAllocator) { g_frameContext[i].CommandAllocator->Release(); g_frameContext[i].CommandAllocator = NULL; }
	if (g_pd3dCommandQueue) { g_pd3dCommandQueue->Release(); g_pd3dCommandQueue = NULL; }
	if (g_pd3dCommandList) { g_pd3dCommandList->Release(); g_pd3dCommandList = NULL; }
	if (g_pd3dRtvDescHeap) { g_pd3dRtvDescHeap->Release(); g_pd3dRtvDescHeap = NULL; }
	if (g_pd3dSrvDescHeap) { g_pd3dSrvDescHeap->Release(); g_pd3dSrvDescHeap = NULL; }
	if (g_fence) { g_fence->Release(); g_fence = NULL; }
	if (g_fenceEvent) { CloseHandle(g_fenceEvent); g_fenceEvent = NULL; }
	if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }

#ifdef DX12_ENABLE_DEBUG_LAYER
	IDXGIDebug1* pDebug = NULL;
	if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&pDebug))))
	{
		pDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_SUMMARY);
		pDebug->Release();
	}
#endif
}

void CreateRenderTarget()
{
	for (UINT i = 0; i < NUM_BACK_BUFFERS; i++)
	{
		ID3D12Resource* pBackBuffer = NULL;
		g_pSwapChain->GetBuffer(i, IID_PPV_ARGS(&pBackBuffer));
		g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, g_mainRenderTargetDescriptor[i]);
		g_mainRenderTargetResource[i] = pBackBuffer;
	}
}

void CleanupRenderTarget()
{
	WaitForLastSubmittedFrame();

	for (UINT i = 0; i < NUM_BACK_BUFFERS; i++)
		if (g_mainRenderTargetResource[i]) { g_mainRenderTargetResource[i]->Release(); g_mainRenderTargetResource[i] = NULL; }
}

void WaitForLastSubmittedFrame()
{
	FrameContext* frameCtx = &g_frameContext[g_frameIndex % NUM_FRAMES_IN_FLIGHT];

	UINT64 fenceValue = frameCtx->FenceValue;
	if (fenceValue == 0)
		return; // No fence was signaled

	frameCtx->FenceValue = 0;
	if (g_fence->GetCompletedValue() >= fenceValue)
		return;

	g_fence->SetEventOnCompletion(fenceValue, g_fenceEvent);
	WaitForSingleObject(g_fenceEvent, INFINITE);
}

FrameContext* WaitForNextFrameResources()
{
	UINT nextFrameIndex = g_frameIndex + 1;
	g_frameIndex = nextFrameIndex;

	HANDLE waitableObjects[] = { g_hSwapChainWaitableObject, NULL };
	DWORD numWaitableObjects = 1;

	FrameContext* frameCtx = &g_frameContext[nextFrameIndex % NUM_FRAMES_IN_FLIGHT];
	UINT64 fenceValue = frameCtx->FenceValue;
	if (fenceValue != 0) // means no fence was signaled
	{
		frameCtx->FenceValue = 0;
		g_fence->SetEventOnCompletion(fenceValue, g_fenceEvent);
		waitableObjects[1] = g_fenceEvent;
		numWaitableObjects = 2;
	}

	WaitForMultipleObjects(numWaitableObjects, waitableObjects, TRUE, INFINITE);

	return frameCtx;
}

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
		return true;

	switch (msg)
	{
	case WM_SIZE:
		if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
		{
			WaitForLastSubmittedFrame();
			CleanupRenderTarget();
			HRESULT result = g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
			assert(SUCCEEDED(result) && "Failed to resize swapchain.");
			CreateRenderTarget();
		}
		return 0;
	case WM_SYSCOMMAND:
		if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
			return 0;
		break;
	case WM_DESTROY:
		::PostQuitMessage(0);
		return 0;
	}
	return ::DefWindowProc(hWnd, msg, wParam, lParam);
}
