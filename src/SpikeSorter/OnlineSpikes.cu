//============================================================================
// Name        : CudaOMP.cpp
// Author      : Sebastian Weingärtner & Bram Simons
// Version     : 0.5
// Copyright   :
// Description : A program to sort spikes online with predetermined templates
//============================================================================


#include "../Helpers/Utils.h"

#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <iterator>
#include <chrono>
#include <queue>
#include <typeinfo>
#include <vector>
#include <thread>
#include "../Helpers/Timer.h"
#include <fstream>

#ifdef WINDOWS
#include <windows.h>
#include <stdlib.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cufft.h>

#include "OnlineSpikes.h"

#include "../Helpers/TimeHelpers.h"
#include "../Networking/onlineSpikesPayload.h"
#include "../Networking/NetworkHelpers.h"
#include "../Networking/FragmentManager.h"
#include "../NetClient/NetClient.h"
#include "myCudnnConvolution.h"
#include "SorterHelpers.h"

#include <atomic>
#include <mutex>

#ifdef WINDOWS
#ifndef _WIN_CLOCK
#define _WIN_CLOCK

#endif
#endif

// Index macro for flattened 3D NpyArrays received from .data(), which are row-major
#define NPY_INDEX_ROW_MAJOR_3D(i, j, k, a, b, c) ((i) * (b) * (c) + (j) * (c) + (k))
#define CUDA_INDEX_COL_MAJOR_3D(i, j, k, a, b, c) ((k) * (a) * (b) + (j) * (a) + (i))

static std::atomic<bool> sentSorterParams = false;
static std::mutex sentSorterParamsMtx;
 
// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::OnlineSpikes()
//
// Description  : Constructor, for initialization
//
// ------------------------------------------------------------------------------
OnlineSpikes::OnlineSpikes(InputParameters Params, sockaddr_in mainAddr, DataSocket* mNC)
// General Processing Settings
				: m_lMinWindow			(Params.iMinScanWindow)					// Minimum amount of samples/window
				, m_lMaxWindow			(Params.iMaxScanWindow)					// Maximum amount of samples/window
				, m_lMaxIterProcessing	(Params.iMaxIts)							// Maximum number of iterations per Data slice for the pseudoinverse
				, m_lMaxIterPinv		(30)							// Maximum number of iterations for the cg_pinv, used to be 50
				, m_dTauProcessing		(Params.dTau)					// Thresholding for adding spike per Data slice. Higher --> less spikes
				, m_dThresProcessing	(Params.dThreshold) 					// Thresholding for first spike per Data slice. Higher --> less spikes
				, m_dRatToMax           (Params.dRatioToMax)							// If non-zero it tries to find responses of m_fRatToMax * Max_Response outside neighboorhood thereby having to do less convolutions
				, m_lRatToMaxTimes      (Params.iConvolutionTimes)					// How many times does it do the above?
				, m_lRedundancy			(Params.iRedundancy)
				, m_lTimeBehind         (Params.iTimeBehind)							// Time (in ms) that it is allowed to lack behind the online recording, set to 0 if you want it to skip batches if it is behind. If it is larger than 100`000 ms it is assumed you don't want any skipping at all (only necessary if onlie sorter is slower).
				, m_lDownsampling		(Params.iDownsampling)						// Temporal Downsampling
				, m_fSamplingRate		(Params.fImecSamplingRate / m_lDownsampling)	// Imec Sampling rate in Hz
				, m_iNidqRefreshRate    (Params.iNidqRefreshRate)

// Non essential value initializations
				, m_lN					(m_lMinWindow + m_lMaxWindow) // Rename m_lN and m_lMaxWindow
				, m_lM					(             0 )
				, m_lT					(             0 )
				, m_lC					(             0 )
				, m_lW					(             0 )
				, m_lM_ReadIn			(			  0	)
				, m_lNInds				(             0 )
				, m_dVRMS				(			  0 )
				, m_fP2P				(             0 )
				, m_lCtDC				(			  0 )
				, m_lSkipCounter		(             0 )
				, m_lSpikeRateWindow    (Params.iAvgWindowTime) // TODO figure out what this is
				, m_bIsSendingFeedback   (Params.bIsSendingFeedback)
				, m_bSmallskip          (Params.bSmallskip)
				, m_uSelectedDevice     (Params.uSelectedDevice)

// Non essential pointer initializations
				, m_lInds				(		   NULL )
				, m_fX					(		   NULL )
				, m_fY					(		   NULL )
				, m_fD					(		   NULL )
				, m_fD2					(		   NULL )
				, m_fW					(		   NULL )
				, m_gfD					(		   NULL )
				, m_gfU					(		   NULL )
				, m_gfV					(		   NULL )
				, m_gfD2				(          NULL )
				, m_gfY                 (          NULL )
				, m_gfX					(          NULL )
				, m_gfYW                (          NULL )
				, m_gfW					(		   NULL )
				, m_fcgA				(          NULL )
				, m_fcgd				(          NULL )
				, m_fcgq				(          NULL )
				, m_fcgs				(          NULL )
				, m_gNpp8MaxBufffer		(          NULL )
				, m_gNpp8MinBufffer		(          NULL )
				, m_gfValBuf			(		   NULL )
				, m_giIndBuf			(          NULL )
				, m_gAx					(          NULL )

// Initialize Objects
				, m_mCC					(0				)	// Initialize convolution object
				, m_imecSock			(Sock::UDP		)
				, m_nidqSock			(Sock::UDP		)
				, imecFm				(&m_imecSock	)
				, nidqFm				(&m_nidqSock	)	
				, m_cuBLAS				(				)	// Initialize cuBLAS context
				, m_DCRem				(               )

// Store shared DataSocket for parallelization
				, m_mNC					(mNC			)

// For visualization debugging purposes
				, m_sPythonVerifierDir  ("C:\\SGL_DATA\\05_31\\cuda_output")
{
	static const char *ptLabel = { "OnlineSpikes::OnlineSpikes" };
	
	// ---- Read in CUDA device ----
	setDevice(Params.uSelectedDevice, &m_mCC);

	std::cout << "OSS started with device number " << Params.uSelectedDevice << " and input directory "
		<< Params.sInputFolder << std::endl;

	// If not correctly found
	if (cublasCreate(&m_cuBLAS) != CUBLAS_STATUS_SUCCESS)
		_RUN_ERROR(ptLabel, "createCuBLAS: Failed to initialize cuBLAS handle");

	if (cusolverDnCreate(&m_cuSolver) != CUBLAS_STATUS_SUCCESS)
		_RUN_ERROR(ptLabel, "Failed to initialized cusolver handle");

	// Template Read in
	cnpy::NpyArray aTemplates = getTemplates(Params.sInputFolder + "templates.npy");

	//Get sizes
	m_lT = aTemplates.shape[0];

	//Read as: if 3D array/matrix set size of channels to that shape, if not 3D array set channels = 1
	m_lC = (aTemplates.shape.size() == 3) ? aTemplates.shape[2] : 1;

	//Samples per Template without undersampling
	m_lM_ReadIn = aTemplates.shape[1];

	//Round down to crop template at the end
	m_lM = m_lM_ReadIn / m_lDownsampling;

	// ---- Format Template Dimensions
	float *fD = aTemplates.data<float>();							// m_lT x m_lM_ReadIn x m_lC; is in row major i.e. lexicographical
	float *fD3;

	/* Construct A. Allocate the matrices on pinned memory for faster copying to device */
	_CUDA_CALL(cudaMallocHost((void**)&m_fD, m_lC * m_lM * m_lT * sizeof(float))); // m_lM x m_lT x m_lC, 
	_CUDA_CALL(cudaMallocHost((void**)&m_fD2, m_lC * m_lM * m_lT * sizeof(float))); // m_lC x m_lM x m_lT // for device copy
	_CUDA_CALL(cudaMallocHost((void**)&fD3, m_lC * m_lM * m_lT * sizeof(float))); // m_lC x m_lM x m_lT // for device copy - inverted for correlation as convolution

	/*-------------------------------
	All matrices are stored in a (1D) list, the matrix is flattened, indexing is from top to bottom (column major)
	if A = T x M x C matrix than
	A[x,y,z] = A[y + x * T + M * T * z]					|T x M x C|

	Transpose over front view:
	A[x,y,z]^T = A[y,x,z] = A[x + y * M + T * M * z]    |T x M x C --> M x T x C|

	Transpose over top view:
	A[x,y,z]^T = A[z,y,x] = A[y + z * C + M * C * x]    |T x M x C --> C x M x T|

	Transpose over side view:
	A[x,y,z]^T = A[x,z,y] = A[z + x * T + C * T * y]    |T x M x C --> T x C x M|
	----------------------------------*/

	//lI = x, lJ = y, lK = z
	for (long sampleInd = 0; sampleInd < m_lM; sampleInd++)           //m_lM = #Samples/template
		for (long templateInd = 0; templateInd < m_lT; templateInd++)       //m_lT = #Templates
			for (long chanInd = 0; chanInd < m_lC; chanInd++) { //m_lC = #Channels

				//Temporary float, which is later put into memory
				float fDTemp = 0.;

				//This simplifies when there is no downsampling (m_lDownsampling = 1), when downsampling != 1 , this 
				//reduces the amount of elements in the matrix
				for (long lU = 0; lU < m_lDownsampling; lU++)
					//fDTemp += fD[templateInd + (sampleInd * m_lDownsampling + lU) * m_lT + chanInd * m_lT * m_lM_ReadIn];
					fDTemp += fD[NPY_INDEX_ROW_MAJOR_3D(templateInd, sampleInd * m_lDownsampling + lU, chanInd, m_lT, m_lM_ReadIn, m_lC)];
					//fDTemp += fD[chanInd + (sampleInd * m_lDownsampling + lU) * m_lC + templateInd * m_lC * m_lM_ReadIn];

				//m_fD  stores the transpose, so that it gets as  |M x T x C|
				//m_fD2 stores the transpose, so that it gets as  |C x M x T|
				m_fD[CUDA_INDEX_COL_MAJOR_3D(sampleInd, templateInd, chanInd, m_lM, m_lT, m_lC)]
					= m_fD2[CUDA_INDEX_COL_MAJOR_3D(chanInd, sampleInd, templateInd, m_lC, m_lM, m_lT)]
					= fDTemp / ((float)m_lDownsampling);
				//m_fD[sampleInd + templateInd * m_lM + chanInd * m_lT * m_lM] = m_fD2[chanInd + sampleInd * m_lC + templateInd * m_lC * m_lM]
				//	= fDTemp / ((float)m_lDownsampling);

			}

	// Flip templates to get correlation values with convolution
	for (long lJ = 0; lJ < m_lT; lJ++) {
		for (long lI = 0; lI < m_lM * m_lC; lI++) {
			fD3[lJ * m_lM * m_lC + m_lM * m_lC - lI - 1] = m_fD2[lI + lJ * m_lM * m_lC];
		}
	}

	FindChanNumbers(m_fD2, m_dChanActivated, "Max");

	_DEBUG_PUT_0(ptLabel, "The following data was loaded successfully: " << m_lT << " Templates with " << m_lM << " Samples on " << m_lC << " channels");

	// ---- Whitening Matrix Read in ----
	cnpy::NpyArray aWhitening = getWhitening(Params.sInputFolder + "whiteningMat.npy");
	if (aWhitening.shape[0] != m_lC || aWhitening.shape[1] != m_lC)
		_RUN_ERROR(ptLabel, "loadWhiteningFile: Incorrect whitening size");

	//Whitening matrix data this is a double?
	float *dT = aWhitening.data<float>();

	//Allocate space
	m_fW = (float *)malloc(m_lC * m_lC * sizeof(float));

	//Copy to memory
	for (long lI = 0; lI < m_lC * m_lC; lI++)
		m_fW[lI] = (float)dT[lI];

	_DEBUG_PUT_0(ptLabel, "The following data was loaded successfully: Whitening matrix " << m_lC << " x " << m_lC);

	// =================================================================
	// Initialize Parameters
	// =================================================================
	// ---- CPU Variables. Pinned memory (memory malloced by cudaMallocHost) speeds up host<-->device data transfer
	_CUDA_CALL(cudaMallocHost((void **)&m_fX, m_lN * m_lT * sizeof(float)));
	_CUDA_CALL(cudaMallocHost((void **)&m_fY, m_lN * m_lC * sizeof(float))); 

	m_lInds = (long *)malloc(m_lN * m_lM * sizeof(long));
	m_lNInds = 0;

	// ---- Device Variables
	_CUDA_CALL(cudaMalloc((void**)&m_gfD2, m_lT * m_lM * m_lC * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_gfD, m_lT * m_lM * m_lC * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_gfU, (m_lN + 2 * (m_lM - 1))* m_lT * sizeof(float))); // Allocate extra space for padding
	_CUDA_CALL(cudaMalloc((void**)&m_gfV, m_lN * m_lC * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_gfY, m_lN * m_lC * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_gfX, m_lN * m_lT * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_gfW, m_lC * m_lC * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_gfYW, m_lN * m_lC * 2.0 * sizeof(float)  ));
	_CUDA_CALL(cudaMalloc((void**)&m_gfDC, m_lC * 1 * sizeof(float)));

	//cg arrays
	_CUDA_CALL(cudaMalloc((void**)&m_fcgA, m_lN * m_lC * m_lMaxIterProcessing * m_lRedundancy * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_fcgd, m_lMaxIterProcessing * m_lRedundancy * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_fcgq, m_lN * m_lC * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_fcgs, m_lMaxIterProcessing * m_lRedundancy * sizeof(float)));
	_CUDA_CALL(cudaMalloc((void**)&m_gAx, m_lN * m_lC * m_lRedundancy * sizeof(float)));

	//Set to zeros
	_CUDA_CALL(cudaMemset(m_fcgA, 0, m_lN * m_lC * m_lMaxIterProcessing * m_lRedundancy * sizeof(float)));
	_CUDA_CALL(cudaMemset(m_fcgd, 0, m_lMaxIterProcessing * m_lRedundancy * sizeof(float)));
	_CUDA_CALL(cudaMemset(m_fcgq, 0, m_lN * m_lC * sizeof(float)));
	_CUDA_CALL(cudaMemset(m_fcgs, 0, m_lMaxIterProcessing * m_lRedundancy * sizeof(float)));


	//Initialize to zeros
	_CUDA_CALL(cudaMemsetAsync(m_gfDC, 0, m_lC * 1 * sizeof(float)));
	_CUDA_CALL(cudaMemsetAsync(m_gfX, 0, m_lN * m_lT * sizeof(float)));


	//Copy fD3 to m_gfD, m_fD2 to m_gfD2 and m_fW to m_gfW
	_CUDA_CALL(cudaMemcpyAsync(m_gfD, fD3, m_lT * m_lM * m_lC * sizeof(float), cudaMemcpyHostToDevice));
	_CUDA_CALL(cudaMemcpyAsync(m_gfD2, m_fD2, m_lT * m_lM * m_lC * sizeof(float), cudaMemcpyHostToDevice));
	_CUDA_CALL(cudaMemcpyAsync(m_gfW, m_fW, m_lC * m_lC * sizeof(float), cudaMemcpyHostToDevice));

	//Wait for the Async copies to be complete
	_CUDA_CALL(cudaDeviceSynchronize());

	//Free/clean fD3 data
	cudaFreeHost(fD3);
	free(m_fW);
	cudaFreeHost(m_fD2);

	// ---- Allocate CUDNN workspace
	m_mCC.setSizes(m_lN, m_lM, m_lC, m_lT);

	// ---- Allocate DC remover sizes
	m_DCRem.InitArrays(m_lC, m_lN);

	int iMaxBufferSize;
	nppsMaxIndxGetBufferSize_32f(m_lN * m_lT, &iMaxBufferSize);

	int iMinBufferSize;
	nppsMaxIndxGetBufferSize_32f(m_lN * m_lT, &iMinBufferSize);

	_CUDA_CALL(cudaMalloc(&m_gNpp8MaxBufffer, iMaxBufferSize));
	_CUDA_CALL(cudaMalloc(&m_gNpp8MinBufffer, iMinBufferSize));
	_CUDA_CALL(cudaMalloc(&m_gfValBuf, sizeof(float)));
	_CUDA_CALL(cudaMalloc(&m_giIndBuf, sizeof(int)));


	// TODO move this code up

	// Connect to the Main Server (so mainAddr can give decoder the sorter's addresses)
	sendConnectMsg(&m_imecSock, mainAddr, _SPIKE_SORTER_IMEC);
	sendConnectMsg(&m_nidqSock, mainAddr, _SPIKE_SORTER_NIDQ);

	// Receive connection from decoder to accquire decoder's addresses
	m_decoderImecAddr = recvConnectMsg(&m_imecSock, _DECODER_IMEC);
	m_decoderNidqAddr = recvConnectMsg(&m_nidqSock, _DECODER_NIDQ);

	// Start up assembler to handle larger packets
	std::thread imecFmThread = imecFm.assemblerThread();
	imecFmThread.detach();

	std::thread imecRetransThread = imecFm.retransmitterThread();
	imecRetransThread.detach();


	// --- Channel map read in ---
	cnpy::NpyArray aChannelMap = getChannelMap(Params.sInputFolder + "channelMap.npy");

	// Check if size is correct
	if (aChannelMap.shape[0] != m_lC)
		_RUN_ERROR(ptLabel, "loadChannelMap: Wrong size, size should be: " + std::to_string(m_lC) + ", but is: " + std::to_string(aChannelMap.shape[0]));

	// Set pointer to the data in the npy file
	int *channelMap = aChannelMap.data<int>();

	std::stringstream ss;
	ss << "Channel map for device #" << Params.uSelectedDevice << " = [";
	for (int i = 0; i < m_lC; i++) {
		m_vChannelMap.push_back(channelMap[i]);
		ss << channelMap[i] << " ";
	}
	ss << "]" << std::endl;
	std::cout << ss.str();

	// ---- Template Map Read in ----
	cnpy::NpyArray aTemplateMap = getTemplateMap(Params.sInputFolder + "templateMap.npy");
	if (aTemplateMap.shape[0] != m_lT)
		_RUN_ERROR(ptLabel, "templateMapFile: Incorrect template map size");

	std::stringstream sss;
	sss << "Device " << Params.uSelectedDevice << " template map = [";
	int minTemplateIndex = INT_MAX;
	int maxTemplateIndex = INT_MIN;
	for (int i = 0; i < m_lT; i++) {
		m_vTemplateMap.push_back(aTemplateMap.data<int>()[i]);
		if (aTemplateMap.data<int>()[i] > maxTemplateIndex) maxTemplateIndex = aTemplateMap.data<int>()[i];
		if (aTemplateMap.data<int>()[i] < minTemplateIndex) minTemplateIndex = aTemplateMap.data<int>()[i];
		sss << aTemplateMap.data<int>()[i] << " ";
	}
	sss << "]" << std::endl;
	std::cout << sss.str();

	// Send SorterParams to Decoder // TODO Evaluate if extracting sorterParams from OnlineSpikes.cu is necessary
	SorterParameters sorterParams = GetSorterParams();

	// can only call this once, since there's only one instance of Decoder, which only uses	params that are
	// agnostic to which OSS params it gets
	
//	{
//		std::unique_lock<std::mutex> lock(sentSorterParamsMtx);
//
//		if (!sentSorterParams) {
			sorterParams.m_lT = maxTemplateIndex + 1; // temp fix for GUI
			sendPayload(&imecFm, sorterParams, m_decoderImecAddr);
//			sentSorterParams = true;
//		}
//	}
	//m_mNC = new StreamDataSocket(Params.sDataAccquisitionHost, Params.uDataAccquisitionPort, Params.iSubstream, Params.iMaxScanWindow, Params.iMinScanWindow, Params.fImecSamplingRate, Params.fNidqSamplingRate, Params.iDownsampling); // TODO move stuff here into StreamDataSocket constructor

	
	
	//03_09 commented, don't need to process nidq
	//m_nidqProcessor = std::thread(&OnlineSpikes::processNidqStream, this);
};


// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::~OnlineSpikes()
//
// Description  : Destructor for freeing pointers
//
// ------------------------------------------------------------------------------
OnlineSpikes::~OnlineSpikes()
{
	static const char *ptLabel = { "OnlineSpikes::~OnlineSpikes" };

	// =================================================================
	// Free Parameter Pointers
	// =================================================================

	_CUDA_CALL(cudaFreeHost(m_fX)); // Call cudaFreeHost for cudaMallocHost'd data
	_CUDA_CALL(cudaFreeHost(m_fY));
	free(m_fW);
	free(m_lInds);
	
	//Free gpu componenets
	_CUDA_CALL(cudaFree(m_gfD));
	_CUDA_CALL(cudaFree(m_gfD2));
	_CUDA_CALL(cudaFree(m_gfU));
	_CUDA_CALL(cudaFree(m_gfV));
	_CUDA_CALL(cudaFree(m_gfY));
	_CUDA_CALL(cudaFree(m_gfX));
	_CUDA_CALL(cudaFree(m_gfW));
	_CUDA_CALL(cudaFree(m_gfYW));
	_CUDA_CALL(cudaFree(m_gfDC));

	//cg arrays
	_CUDA_CALL(cudaFree(m_fcgA));
	_CUDA_CALL(cudaFree(m_fcgd));
	_CUDA_CALL(cudaFree(m_fcgq));
	_CUDA_CALL(cudaFree(m_fcgs));
	_CUDA_CALL(cudaFree(m_gAx));


	_CUDA_CALL(cudaFree(m_gNpp8MaxBufffer));
	_CUDA_CALL(cudaFree(m_gNpp8MinBufffer));
	_CUDA_CALL(cudaFree(m_gfValBuf));
	_CUDA_CALL(cudaFree(m_giIndBuf));


	if (cublasDestroy(m_cuBLAS) != CUBLAS_STATUS_SUCCESS) {
		_RUN_ERROR(ptLabel, "Unable to destroy cublas handle.");
	}
	if (cusolverDnDestroy(m_cuSolver) != CUBLAS_STATUS_SUCCESS) {
		_RUN_ERROR(ptLabel, "Unable to destroy cusolver handle.");
	}
	m_nidqProcessor.join();
};

void OnlineSpikes::processNidqStream() {
	
	t_ull eventSampleCt;
	int eventLabel = -1;
	int predictLabel = -1;
 
	t_ull lLatestSampleCt = m_mNC->initNidqStream();
	
	while (true) {
		// Wait m_iNidqRefreshRate ms to pass so fetchEventInfo call has data to fetch (if no data is fetched SpikeGLX complains)
		efficientWait(m_iNidqRefreshRate);
		
		if (m_bIsSendingFeedback) {
			// Send predictLabel as feedback over the NIDQ stream
			m_nidqSock.recvData(&predictLabel, sizeof(int));
			//predictLabel comes out as decimal number; need to convert to hex to get intended result
			std::stringstream ss;
			ss << std::hex << predictLabel;
			std::string res(ss.str());

			//std::cout <<  res.back()<<"\n";
			int test = res.back() - '0';
			std::cout << "TEST INT" << test;
			//predict Label here is not what we expect
			m_mNC->setDigitalOut(test);


			// Wait some time
			efficientWait(200); // TODO figure out how small the wait value can be! (how long do we need the signal up for psychtoolbox to read the feedback signal)

			// Bring the digitalOut back to 0.
			m_mNC->setDigitalOut();
		}
		/*
		// Fetch stimulus event info, setting sampleCt to current sample count. If a stimulus event occurs, also reset eventSampleCt and eventLabel 
		//ISSUE HERE COMMENTING OUT FIXES THE thing
		lLatestSampleCt = m_mNC->fetchEventInfo(eventLabel, lLatestSampleCt);
		
		if (eventLabel != -1) {
			// Send event signal to decoder
			eventSampleCt = m_mNC->getStreamSampleCt(IMEC);
			
			//std::cout << eventSampleCt << std::endl; // TODO examine latency

			m_nidqSock.sendData(&eventSampleCt, sizeof(t_ull), m_decoderNidqAddr);
			m_nidqSock.sendData(&eventLabel, sizeof(int), m_decoderNidqAddr);

			// Reset eventLabel
			eventLabel = -1;

			

			// Should sleep some amount here (if Psychtoolbox is imprecise about sending over signals)
		}*/
	}
}
void OnlineSpikes::runSpikeSorting() {
	static const char *ptLabel = { "OnlineSpikes::runSpikeSorting" };

	long 	lProcessedCt, // Most recent stream sample count that has been processed
			lLatestCt,	// Most recent stream sample count
			lAllowedCt;
	bool bSkip = false;
	m_lSkipCounter = 0;

	// Start up the spikeGLX run
	//m_mNC->startRun();

	// =================================================================
	// Get Initial Data.
	// =================================================================
	// params specific to this sorter for parallelization purposes
	OSSSpecificParams osParams = {
		m_lC,
		m_vChannelMap
	};

	std::cout << "osParams.m_lC = " << m_lC << std::endl;
	std::cout << "osParams.m_vChannelMap = [";
	for (auto const& channel : m_vChannelMap)
		std::cout << channel << " ";
	std::cout << "]" << std::endl;

	// Wait for minimum processing window of data to be ready
	lLatestCt = m_mNC->getStreamSampleCt(IMEC, osParams);
	m_mNC->waitUntil(lLatestCt + m_lMinWindow, osParams);

	// Update stream sample count and fill m_fY with m_lMinScanWin amount of data
	lLatestCt = m_mNC->fetchLatest(m_fY, osParams);

	// Set window size
	m_lW = m_lMinWindow;

	// Copy data from cpu --> gpu
	_CUDA_CALL(cudaMemcpy(m_gfYW, m_fY, m_lC * m_lW * sizeof(float), cudaMemcpyHostToDevice));

	// Whiten: m_gfY = m_gfW * m_gfYW
	WhitenOnGPU(m_cuBLAS, m_gfW, m_gfYW, m_gfY, m_lW, m_lC);

	// Remove DC (0th frequency)
	m_lCtDC = m_DCRem.RemoveDC(m_cuBLAS, m_gfY, m_gfDC, m_lW, m_lC, m_lCtDC);

	cudaDeviceSynchronize();


	// Update processed stream sample count
	lProcessedCt = lLatestCt;

	// =================================================================
	// Data Processing
	// =================================================================

	// Init timespec variables to track processing time 
	struct timespec tBatchBefore, tBatchAfter;

	// Vectors to store the spike times, templates, and amplitudes to be sent to the Decoder
	std::vector<long> vTimes;
	std::vector<long> vTemplates;
	std::vector<float> vAmplitudes;

	//Main loop, runs while not finished
	while (true) {
		// Wait for the minimum data window if we're ahead
		m_mNC->waitUntil(lProcessedCt + m_lMinWindow, osParams);

		clock_gettime(tBatchBefore);

		//preimplementation below
		lLatestCt = m_mNC->getStreamSampleCt(IMEC, osParams);
		//lLatestCt = m_mNC->fetchLatest(m_fY + m_lMinWindow * m_lC, osParams, lProcessedCt);

		//moved 508-> 498
		//lLatestCt = m_mNC->fetchLatest(m_fY + m_lMinWindow * m_lC, lProcessedCt);
		//check if allowed count < processed count, move pointer of m_fy; make sure you fetch right amount
		//Calculate time allowed to be behind (divide by 1000 as it is in ms)


		lAllowedCt = lLatestCt - m_lTimeBehind * m_fSamplingRate / 1'000;
		std::cout << "Device " << m_uSelectedDevice << " calling fetch with latestCt - processedCt = " << lLatestCt - lProcessedCt << std::endl;
		
		//----------------Check what to do with data/batch-------------------
		// We're on time, so we fetch the latest batch of data
		if ((lLatestCt - lProcessedCt) <= m_lMaxWindow) {
			//std::cout << "Case 1\n";
			lLatestCt = m_mNC->fetchLatest(m_fY, osParams, lProcessedCt);

			// timing issues, such as if command line takes over main thread of execution
			// causes lLatestCt to be very large, making lLatestCt - lProcessedCt exceed
			// m_lMaxWindow despite the conditional, thus the need for the min
			m_lW = min((lLatestCt - lProcessedCt) * (lLatestCt - lProcessedCt >= 0), m_lMaxWindow) + m_lMinWindow;
			//m_lW = max(lLatestCt - lProcessedCt, m_lMinWindow);
			//m_lW = lLatestCt - lProcessedCt + m_lMinWindow;

			bSkip = false;
		} // We're behind, but being behind is not tolerated (m_lTimeBehind == 0) so skip enough data to fetch most recent m_lMaxWindow batch
		else if (m_lTimeBehind == 0) {
			std::cout << "Case 2\n";
			//lLatestCt = m_mNC->fetchLatest(m_fY + m_lMinWindow * m_lC, osParams, lProcessedCt);
			lLatestCt = m_mNC->fetchLatest(m_fY + m_lMinWindow * m_lC, osParams, lProcessedCt);
			m_lW = m_lMaxWindow;
			bSkip = true;	
		} // We are behind, but we think we can catch up, thus take from the place we are at now	
		else if (lProcessedCt >= lAllowedCt) {
			std::cout << "Case 3\n";
			lLatestCt = m_mNC->fetchFromPlace(m_fY + m_lMinWindow * m_lC, osParams, lProcessedCt);
			m_lW = m_lMaxWindow + m_lMinWindow;
			bSkip = false;
		} // We are too much behind, 2 options, small or big skip. 
		// Smallskip: we fetch from the sample count we are allowed to be
		else if (m_bSmallskip == true) {
			std::cout << "Case 4\n";
			lLatestCt = m_mNC->fetchFromPlace(m_fY + m_lMinWindow * m_lC, osParams, lAllowedCt);
			m_lW = m_lMaxWindow;
			bSkip = true;
		} // Bigskip: we fetch the most recent m_lMaxWindow batch.
		else {  // m_bSmallskip == false 
			std::cout << "Case 5\n";
			lLatestCt = m_mNC->fetchLatest(m_fY + m_lMinWindow * m_lC, osParams, lProcessedCt);
			m_lW = m_lMaxWindow;
			bSkip = true;
		}

		// Copy Data from cpu --> gpu
		if (bSkip) { 
			// Skip the last minScanWindow of previous batch (the first m_lMinWindow * m_lC bits)
			_CUDA_CALL(cudaMemcpy(m_gfYW, m_fY + m_lMinWindow * m_lC, m_lC * m_lW * sizeof(float), cudaMemcpyHostToDevice));

			// Increment skip counter
			m_lSkipCounter++;
		}
		else { 
			_CUDA_CALL(cudaMemcpy(m_gfYW, m_fY, m_lC * m_lW * sizeof(float), cudaMemcpyHostToDevice));
		}
		
		_CUDA_CALL(cudaDeviceSynchronize());

		/*
		std::cout << "COPYING TO DEBUG with C = " << m_lC << " and W = " << m_lW << std::endl;
		_CUDA_CALL(cudaMemcpy(y, m_gfYW, m_lC * m_lW * sizeof(float), cudaMemcpyDeviceToHost));
		_CUDA_CALL(cudaDeviceSynchronize());
		std::cout << "Finished copying!" << std::endl;
		outf << "Sample " << lProcessedCt << "\n";
		outf << "Channels " << m_lC << "\n";
		outf << "Window " << m_lW << "\n";
		outf << "ChannelMap ";
		for (auto const& channel : m_vChannelMap)
			outf << channel << " ";
		outf << "\n";
		for (int i = 0; i < m_lC * m_lW; i++) {
			//std::cout << "Writing " << y[i] << std::endl;
			outf << y[i] << " ";
		}
		outf << "\n";
		*/
		// Calculate Peak to Peak
		P2P_calc(m_gfYW, m_lW * m_lC, &m_fP2P);	

		// Whiten the data (on gpu)
		WhitenOnGPU(m_cuBLAS, m_gfW, m_gfYW, m_gfY, m_lW, m_lC);

		_CUDA_CALL(cudaDeviceSynchronize());

		// Move Data as Fifo so that we keep last minScanWindow of this batch (not whitened nor dc filtered)
		std::thread FIFO(&OnlineSpikes::MoveFifo, this);

		// Remove DC (0th frequency)
		m_lCtDC = m_DCRem.RemoveDC(m_cuBLAS, m_gfY, m_gfDC, m_lW, m_lC, m_lCtDC);

		_CUDA_CALL(cudaDeviceSynchronize());

		/*
		_CUDA_CALL(cudaMemcpy(y, m_gfYW, m_lC * m_lW * sizeof(float), cudaMemcpyDeviceToHost));

		outf << "Preprocessed" << "\n";
		for (int i = 0; i < m_lC * m_lW; i++) {
			outf << y[i] << " ";
		}
		outf << "\n";*/

		// processData returns m_lNInds (the number of spikes found) and m_lInds (the spikes' indices)

		m_lNInds = processData(m_gfY, m_lW, m_lInds);
		std::cout << "m_lNInds = " << m_lNInds << std::endl;

		_CUDA_CALL(cudaMemsetAsync(m_fcgA, 0, m_lNInds * m_lN * m_lC * sizeof(float)));

		// Fill vectors (rename function eventually) 
		saveSpikes(m_lNInds, m_lInds, lLatestCt - m_lW + 1, m_lW - m_lMinWindow, vTimes, vTemplates, vAmplitudes);

		// Calculate time taken to process data
		clock_gettime(tBatchAfter);
		long processTime = GetTimeDiff(tBatchAfter, tBatchBefore);

		// Send OnlineSpikesPayload to Decoder
		OnlineSpikesPayload payload = { 0,
										lLatestCt,
										vTimes,
										vTemplates,
										vAmplitudes,
										m_dVRMS,
										m_fP2P,
										processTime };
		sendPayload(&imecFm, payload, m_decoderImecAddr);
		

		//update stream sample count
		lProcessedCt = lLatestCt;

		// Clear out vectors for next iteration
		vTimes.clear();
		vTemplates.clear();
		vAmplitudes.clear();

		// Move Data as Fifo so that we keep last minscanwindow of this batch
		FIFO.join();
	}
}

template<typename T>
void printDeviceVector(const T* d_vec, int size, const std::string& vecName) {
	std::vector<T> h_vec(size);
	cudaMemcpy(h_vec.data(), d_vec, size * sizeof(T), cudaMemcpyDeviceToHost);
	std::cout << vecName << " contents: ";
	for (const auto& elem : h_vec) {
		std::cout << elem << " ";
	}
	std::cout << std::endl;
}

bool isNonZero(const float* d_vec, int size) {
	std::vector<float> h_vec(size);
	cudaMemcpy(h_vec.data(), d_vec, size * sizeof(float), cudaMemcpyDeviceToHost);

	for (const auto& elem : h_vec) {
		if (elem != 0.0) return true;
	}
	return false;
}

__global__ void computeReciprocalSqrt(float* input, float* output, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		output[idx] = (input[idx] > 0) ? 1.0f / sqrtf(input[idx]) : 0.0f;
	}
}

template<typename T>
void prettyPrintgA(T* gA, long lda, long ldb, int max_rows = 250, int max_cols = 10) {
	// Allocate host memory
	T* hA = new T[lda * ldb];

	// Copy data from device to host
	cudaMemcpy(hA, gA, lda * ldb * sizeof(T), cudaMemcpyDeviceToHost);

	// Determine how many rows and columns to print
	int rows_to_print = min(static_cast<long>(max_rows), lda);
	int cols_to_print = min(static_cast<long>(max_cols), ldb);

	std::cout << "Matrix gA (showing up to " << rows_to_print << "x" << cols_to_print << " elements):" << std::endl;

	for (int i = 0; i < rows_to_print; ++i) {
		for (int j = 0; j < cols_to_print; ++j) {
			if (hA[j* lda + i] != 0.0) {
				std::cout << "Index (" << i << ", " << j << "): ";
				std::cout << std::setw(12) << std::setprecision(4) << std::fixed << hA[j * lda + i] << "\n";
			}
		}
		if (cols_to_print < ldb) std::cout << "...";
		std::cout << std::endl;
	}

	if (rows_to_print < lda) {
		std::cout << "..." << std::endl;
	}

	std::cout << "Matrix dimensions: " << lda << " x " << ldb << std::endl;

	// Free host memory
	delete[] hA;
}

/*
__global__ void computeColL2Norm(const float* d_matrix, float* d_norms, int num_rows, int num_cols) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < num_cols) {
		float sum = 0.0f;

		// Compute the sum of squares for the current col
		for (int row = 0; row < num_rows; row++) {
			float val = d_matrix[row * num_cols + col];
			sum += val * val;
		}

		// Store the computed L2 norm for the current row
		d_norms[col] = sqrtf(sum);
	}
}

__global__ void normalizeCols(float* d_matrix, const float* d_norms, int num_rows, int num_cols) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < num_cols) {
		float norm = d_norms[col];

		// Normalize each element in the row by the computed norm
		for (int row = 0; row < num_rows; row++) {
			d_matrix[row * num_cols + col] /= norm;
		}
	}
}
*/
/*
void OnlineSpikes::normalize_cols(float *d_matrix, int num_rows, int num_cols) {
	static const char *ptLabel = { "OnlineSpikes::normalize" };

	// First, compute the norm of each row
	float *d_norms;
	_CUDA_CALL(cudaMalloc(&d_norms, num_cols * sizeof(float)));

	int blockSize = 256;
	int gridSize = (num_rows + blockSize - 1) / blockSize;
	computeColL2Norm<<<gridSize, blockSize>>> (d_matrix, d_norms, num_rows, num_cols);
	cudaDeviceSynchronize();
	normalizeCols<<<gridSize, blockSize>>> (d_matrix, d_norms, num_rows, num_cols);
	cudaDeviceSynchronize();
	cudaFree(d_norms);
}*/

long OnlineSpikes::processData(float *fY, long lW, long* lInds) {
	static const char *ptLabel = { "OnlineSpikes::processData" }; _UNUSED(*ptLabel);
	// fY = Measured data (with or without offset)
	// lW = #scans in window
	// lInds = list of identified events (which is empty now, but will be filled by function)

	// =================================================================
	// Local variables
	// =================================================================
	register long lI2;

	double dL2_Y, dL2_Old, dL2_New;

	long lNInds = 0;
	float l2R_temp = 0;

	// =================================================================
	// Main Optimization Loop.
	// =================================================================

	//Initialize m_fX with zeros
	memset(m_fX, 0, lW * m_lT * sizeof(float));

	// should be channel major
	_CUDA_CALL(cudaMemcpyAsync(m_gfV, fY, lW * m_lC * sizeof(float), cudaMemcpyDeviceToDevice));

	if (cublasSnrm2(m_cuBLAS, lW * m_lC, m_gfV, 1, &l2R_temp) != CUBLAS_STATUS_SUCCESS) {
		_RUN_ERROR(ptLabel, "Failed to calculate new L2 norm");
	}

	_CUDA_CALL(cudaDeviceSynchronize());

	// Initial L_2^2 
	dL2_Y = dL2_Old = dL2_New = l2R_temp * l2R_temp;// * l2R_temp;
	//	std::cout << "Initial residual norm squared = " << dL2_Y << std::endl;

	// Here we do the main calculation (OMP)
	for (long lIter = 0; lIter < m_lMaxIterProcessing; lIter++) {
		/*----------------------------------------------------------
		Here we do the Convolution to find where we have a high response
		m_gfV = Y (measured data/signal) , m_gfD = A (templates matrix, but flipped), m_gfU = output (response)
		This can also be seen as the matrix-vector product as we have toeplitz structure A*y
		------------------------------------------------------------*/
	
		// m_gfU has length (W + 2 * (M - 1)) * T, the 2 * (M-1) term is just for padding, so think of as W * T
		// m_gfV has length  W * C
		// m_gfD has length T * M * C
		_CUDA_CALL(cudaMemset(m_gfU, 0, (m_lN + 2 * (m_lM - 1))* m_lT * sizeof(float)));
		_CUDA_CALL(cudaDeviceSynchronize());
		{
			Timer timer("Convolution");
			m_mCC.fwdConvolve(m_gfV, m_gfD, m_gfU, lW, m_lM, m_lC, m_lT);
		}

		_CUDA_CALL(cudaDeviceSynchronize());

		// Add largest Correlation
		float fMaxU{}; // Copy maximum value to allow Threshold check
		
		//Because of cuda convolution we made U bigger, here we ignore the padding
		float *gfU_pad = (m_gfU + (m_lM - 1) * m_lT); // padded only the start of U with (M-1)*T entries?

		/*
		{ // normalize each channel in output of convolve to norm 1	
			Timer timer("Correlation normalization");
			normalize_cols(gfU_pad, lW, m_lT);
		}*/

		//Initialize maximum value (value doesn't matter)
		int iMaxU{};

		//Find maximum with kernel
		{
			Timer timer("Max");
			FindMax(gfU_pad, lW * m_lT, &iMaxU, &fMaxU);
		}

		//Increase lInds
		lInds[lNInds++] = iMaxU;

		std::cout << "Spike found at template " << iMaxU % m_lT << " and sample " << iMaxU / m_lT << " with correlation " << fMaxU << std::endl;

		//The amount of spikes/columns we add later in the pinv
		long lAmountToAdd = 1;

		// Threshold Exit. There are no spikes to find because the max amplitude was smaller than the threshold.
		if (fabs(fMaxU) < m_dThresProcessing)	 {
			std::cout << "THRESHOLD EXITING" << std::endl;

			//Clean gotten X and decrement lNinds
			lNInds--;
			m_fX[lInds[lNInds]] = 0.f;
			break;
		}

		//If m_dRatToMax is not zero and not many iteration have been done, use the 'trick'
		if (m_dRatToMax != 0 && (lIter < m_lRatToMaxTimes)) {

			//Remember values
			int iCurMaxIndex = iMaxU;
			int iCurMaxIndexOld = iMaxU;
			float fCurMaxValue = fMaxU;

			//Set area around found spike to zero
			float *gfU_temp = gfU_pad + (iMaxU / m_lT) * m_lT - min(iMaxU / m_lT, m_lM / 2) * m_lT;
			_CUDA_CALL(cudaMemset(gfU_temp, 0, min(m_lM, lW - iMaxU / m_lT + m_lM / 2 + 1)  * m_lT * sizeof(float)));


			//Run Loop
			while (fCurMaxValue > m_dRatToMax * fMaxU) {

				//Find the maximum value
				FindMax(gfU_pad, m_lT * lW, &iCurMaxIndex, &fCurMaxValue);

				//Check if it is not to small or on the edge
				if ((fCurMaxValue < m_dRatToMax * fMaxU) && (std::abs(iCurMaxIndex / m_lT - iCurMaxIndexOld / m_lT) <= m_lM / 2 + 1)) {
					break;
				}

				//increment lInds
				lInds[lNInds++] = iCurMaxIndex;
				lAmountToAdd++;

				//Remember maximum
				iCurMaxIndexOld = iCurMaxIndex;

				//Set parts around found to zero
				float *gfU_temp = gfU_pad + (iCurMaxIndex / m_lT) * m_lT - min(iCurMaxIndex / m_lT, m_lM / 2) * m_lT;
				_CUDA_CALL(cudaMemset(gfU_temp, 0, min(m_lM, lW - iCurMaxIndex / m_lT + m_lM / 2 + 1)  * m_lT * sizeof(float)));

				//sync device
				_CUDA_CALL(cudaDeviceSynchronize());
			}
		}

		_CUDA_CALL(cudaMemcpyAsync(m_gfV, fY, lW * m_lC * sizeof(float), cudaMemcpyDeviceToDevice));

		/*---------------------------------------------------
		X = pinv(A) * y is calculated on cpu?
		m_fD2 =  template matrix (A)   c x m x l
		m_fV =   signal space data (y)
		m_fX =   spike space data (X) (output)
		lInds =  list of identified events (indices with high response)
		lNInds = # Identified events
		Residual: V = Y - A * X
		Also calculates new residual --- so m_gfV gets updated
		-----------------------------------------------------*/
		//printDeviceVector(m_gfV, 25, "m_gfV before");
		{
			Timer timer("Pseudoinverse");
			std::cout << "Adding " << lAmountToAdd << " spikes for pseudoinverse." << std::endl;
			cu_cgPseudoInverse(m_gfD2, lInds, m_gfV, m_gfX, lW, m_lM, m_lT, m_lC, lNInds, lAmountToAdd);
		}
		_CUDA_CALL(cudaDeviceSynchronize());

		//printDeviceVector(m_gfV, 25, "m_gfV after");
		// m_gfV should now contain the new residual, and m_gfX should now contain an approximation of A^+ y

		for (size_t i = 0; i < lNInds; i++) {
			//std::cout << "Copying index " << lInds[i] << " of X from GPU to CPU" << std::endl;
			_CUDA_CALL(cudaMemcpyAsync(&(m_fX[lInds[i]]), &(m_gfX[i]), sizeof(float), cudaMemcpyDeviceToHost));
		}	
		_CUDA_CALL(cudaDeviceSynchronize());

//		std::cout << "m_fX = [";
	//	for (size_t i = 0; i < lW * m_lT; i++) std::cout << m_fX[i] << " ";
		//std::cout << "]" << std::endl;


		// Copy spike vector gpu --> cpu
		//_CUDA_CALL(cudaMemcpyAsync(m_fX, m_gfX, lNInds * sizeof(float), cudaMemcpyDeviceToHost));

		//Save old L_2^2
		dL2_Old = dL2_New;

		//Calculate new L_2^2
		if (cublasSnrm2(m_cuBLAS, lW * m_lC, m_gfV, 1, &l2R_temp) != CUBLAS_STATUS_SUCCESS)
			_RUN_ERROR(ptLabel, "Failed to calculate new L2 norm");

		_CUDA_CALL(cudaDeviceSynchronize());

		//Update L_2^2
		dL2_New = l2R_temp * l2R_temp;

		if (lIter == 0) {
			dL2_Y = l2R_temp * l2R_temp;
		}

		cudaDeviceSynchronize();

		//for (long lI = 0; lI < lNInds; lI++)
		//	m_fX[lInds[lI]] = m_fX[lI];

		//Halt Criteria. No more spikes to find given threshold parameter
		std::cout << "LHS = " << (dL2_Y - dL2_Old) / (lNInds - 1.) * m_dTauProcessing << std::endl;
		std::cout << "RHS = " << (dL2_Old - dL2_New) << std::endl;
		std::cout << "Residual norm squared = " << dL2_New << std::endl;

		if ((lNInds > 1) && ((dL2_Y - dL2_Old) / (lNInds - 1.) * m_dTauProcessing > (dL2_Old - dL2_New))) {
			std::cout << "Halt criteria, iterations = " << lIter + 1 << std::endl;
			m_dVRMS = l2R_temp / (std::sqrt(lW * m_lC));
			break;
		}
	}

	return lNInds;
}

void OnlineSpikes::test_convolution()
{
	// m_gfU has length (W + 2 * (M - 1)) * T, the 2 * (M-1) term is just for padding, so think of as W * T
	// m_gfV has length  W * C
	// m_gfD has length T * M * C

	static const char *ptLabel = { "OnlineSpikes::test_convolution" };

	/*------------- Load Y ---------------*/
	std::cout << "Loading and processing Y" << std::endl;
	cnpy::NpyArray npy_Y = cnpy::npy_load("C:\\Users\\Spike\ Sorter\\source\\repos\\OnlineSpikes_v2\\src\\Python\\cg_learning_Y.npy");
	double* c_Y = npy_Y.data<double>();
	long numRows = npy_Y.shape[0];
	std::cout << "npy_Y.shape = [";
	for (const auto &dimLen : npy_Y.shape)
		std::cout << dimLen << " ";
	std::cout << "]" << std::endl;
	std::cout << "Read in y with length " << numRows << std::endl;

	m_lN = numRows / m_lC;

	float* c_Y_casted = (float*)malloc(numRows * sizeof(float));
	std::cout << "c_Y (head) = [ ";
	for (size_t i = 0; i < numRows; i++) {
		if (i < 10)
			std::cout << c_Y[i] << " ";
		c_Y_casted[i] = static_cast<float>(c_Y[i]);
	}
	std::cout << "]" << std::endl;

	float *g_Y;
	_CUDA_CALL(cudaMalloc((void**)&g_Y, numRows * sizeof(float)));
	cudaMemcpyAsync(g_Y, c_Y_casted, numRows * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	printDeviceVector(g_Y, 20, "g_Y head");

	/* Want output to be of size W * T */
	float* g_U;
	_CUDA_CALL(cudaMalloc((void**)&g_U, (m_lN + 2 * (m_lM - 1)) * m_lT * sizeof(float)));
	_CUDA_CALL(cudaMemsetAsync(g_U, 0, (m_lN + 2 * (m_lM - 1)) * m_lT * sizeof(float)));
	
	m_mCC.fwdConvolve(g_Y, m_gfD, m_gfU, m_lW, m_lM, m_lC, m_lT);
	printDeviceVector(m_gfU, 50, "m_gfU head");

	float fMaxU{}; // Copy maximum value to allow Threshold check

	//Initialize maximum value (value doesn't matter)
	int iMaxU{};

	//Find maximum with kernel
	FindMax(g_U, (m_lN + 2 * (m_lM - 1)) * m_lT * sizeof(float), &iMaxU, &fMaxU);
	std::cout << "Max U = " << fMaxU << std::endl;
}

void OnlineSpikes::test_cgPseudoInverse() {
	long lIter = 0;

	//Parameter
	double tau = 0.01;    //Halt criteria;

	//parameters to make sure cublasSgemm works as intended
	const double fAlpha_blas = 1.0;
	const double fAlpha_blas_negative = -1.0;
	const double fBeta_blas = 0.0;

	double falpha_L2, fbeta_L2;

	static const char *ptLabel = { "OnlineSpikes::test_cgPseudoInverse" };

	/* -------------- Load A ---------------*/
	std::cout << "Loading and processing A" << std::endl;
	cnpy::NpyArray npy_A = cnpy::npy_load("C:\\Users\\Spike\ Sorter\\source\\repos\\OnlineSpikes_v2\\src\\Python\\cg_learning_A.npy");
	double* c_A = npy_A.data<double>();
	long numRows = npy_A.shape[0];
	long numCols = npy_A.shape[1];

	std::vector<double> c_A_transposed(numRows * numCols);

	for (size_t i = 0; i < numRows; i++) {
		for (size_t j = 0; j < numCols; j++) {
			c_A_transposed[j * numRows + i] = c_A[i * numCols + j];
		}
	}

	// Allocate GPU memory
	double *g_A;
	_CUDA_CALL(cudaMalloc((void**)&g_A, numRows * numCols * sizeof(double)));

	// Copy the entire array to the GPU in one call
	_CUDA_CALL(cudaMemcpy(g_A, c_A_transposed.data(), numRows * numCols * sizeof(double), cudaMemcpyHostToDevice));

	prettyPrintgA(g_A, numRows, numCols);

	/*------------- Load Y ---------------*/
	std::cout << "Loading and processing Y" << std::endl;
	cnpy::NpyArray npy_Y = cnpy::npy_load("C:\\Users\\Spike\ Sorter\\source\\repos\\OnlineSpikes_v2\\src\\Python\\cg_learning_Y.npy");	
	double* c_Y = npy_Y.data<double>();

	std::cout << "c_Y = [";
	for (size_t i = 0; i < numRows; i++)
		std::cout << c_Y[i] << " ";
	std::cout << "]" << std::endl;

	double *g_Y;
	_CUDA_CALL(cudaMalloc((void**)&g_Y, numRows * sizeof(double)));
	cudaMemcpyAsync(g_Y, c_Y, numRows * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	printDeviceVector(g_Y, numRows, "g_Y");

	double* g_X;
	_CUDA_CALL(cudaMalloc((void**)&g_X, numCols * sizeof(double)));
	_CUDA_CALL(cudaMemsetAsync(g_X, 0, numCols * sizeof(double)));


	//----------------------------------- Fill Matrix/Vectors ---------------------

	_CUDA_CALL(cudaDeviceSynchronize());

	// r = A^t y
	double *g_r;
	_CUDA_CALL(cudaMalloc((void**)&g_r, numCols * sizeof(double)));
	_CUDA_CALL(cudaMemsetAsync(g_r, 0, numCols * sizeof(double)));
	_CUDA_CALL(cudaDeviceSynchronize());
	cublasDgemv(m_cuBLAS, CUBLAS_OP_T, numRows, numCols, &fAlpha_blas, g_A, numRows, g_Y, 1, &fBeta_blas, g_r, 1);
	_CUDA_CALL(cudaDeviceSynchronize());
	printDeviceVector(g_r, numCols, "g_r");

	// d = r
	double *g_d;
	_CUDA_CALL(cudaMalloc((void**)&g_d, numCols * sizeof(double)));
	_CUDA_CALL(cudaMemcpyAsync(g_d, g_r, numCols * sizeof(double), cudaMemcpyDeviceToDevice));
	printDeviceVector(g_d, numCols, "g_d");
	_CUDA_CALL(cudaDeviceSynchronize());

	// note that these are not squared
	double l2R_new, l2R_old, l2R_ini;

	// delta_new = <r, r> = L_2(r)^2
	double rr_dot{};
	if (cublasDdot(m_cuBLAS, numCols, g_r, 1, g_r, 1, &rr_dot) != CUBLAS_STATUS_SUCCESS) {
		_RUN_ERROR(ptLabel, "failed to calculate dot product (d^T q)");
	}
	_CUDA_CALL(cudaDeviceSynchronize());

	if (isnan(rr_dot))
		throw std::runtime_error("Failed to calculate L2 norm of residual\n");

	std::cout << "Initial L_2(r)^2 = " << rr_dot << std::endl;
	l2R_ini = rr_dot;
	l2R_new = l2R_ini;
	l2R_old = l2R_ini;

	_CUDA_CALL(cudaDeviceSynchronize());

	// Declare a pointer for the temporary vector
	double *temp;

	cudaError_t cudaStatus = cudaMalloc(&temp, numCols * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for temp vector: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemset(temp, 0, numCols * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed for temp vector: %s\n", cudaGetErrorString(cudaStatus));
	}

	double *g_q;
	_CUDA_CALL(cudaMalloc((void**)&g_q, numCols * sizeof(double)));
	cudaMemset(g_q, 0, numCols * sizeof(double));
	_CUDA_CALL(cudaDeviceSynchronize());

	while ((l2R_new > tau * tau * l2R_ini) && (lIter < m_lMaxIterPinv)) {
		// q = A^T * A * d
		double* g_temp;
		_CUDA_CALL(cudaMalloc((void**)&g_temp, numRows * sizeof(double)));
		cudaMemset(g_temp, 0, numRows * sizeof(double));
		_CUDA_CALL(cudaDeviceSynchronize());
		cublasDgemv(m_cuBLAS, CUBLAS_OP_N, numRows, numCols, &fAlpha_blas, g_A, numRows, g_d, 1, &fBeta_blas, g_temp, 1);
		_CUDA_CALL(cudaDeviceSynchronize());
		cublasDgemv(m_cuBLAS, CUBLAS_OP_T, numRows, numCols, &fAlpha_blas, g_A, numRows, g_temp, 1, &fBeta_blas, g_q, 1);
		_CUDA_CALL(cudaDeviceSynchronize());

		// Compute alpha = l2r_new / d^T q
		double d_dot_q;
		cublasDdot(m_cuBLAS, numCols, g_d, 1, g_q, 1, &d_dot_q);
		falpha_L2 = l2R_new / d_dot_q;
		_CUDA_CALL(cudaDeviceSynchronize());

		// Update x = x + alpha * d
		cublasDaxpy(m_cuBLAS, numCols, &falpha_L2, g_d, 1, g_X, 1);
		_CUDA_CALL(cudaDeviceSynchronize());

		// Update residual r = r - alpha * q
		double negAlpha = -falpha_L2;
		cublasDaxpy(m_cuBLAS, numCols, &negAlpha, g_q, 1, g_r, 1);

		// Compute new l2R_new
		l2R_old = l2R_new;
		cublasDdot(m_cuBLAS, numCols, g_r, 1, g_r, 1, &l2R_new);

		_CUDA_CALL(cudaDeviceSynchronize());

		// Compute beta using Fletcher-Reeves formula: beta = l2r_new / l2r_old
		fbeta_L2 = l2R_new / l2R_old;

		// Update d = r + beta * d
		cublasDscal(m_cuBLAS, numCols, &fbeta_L2, g_d, 1);
		cublasDaxpy(m_cuBLAS, numCols, &fAlpha_blas, g_r, 1, g_d, 1);

		_CUDA_CALL(cudaDeviceSynchronize());

		// Print diagnostics
		std::cout << "Iteration " << lIter << ": alpha = " << falpha_L2
			<< ", beta = " << fbeta_L2 << ", residual = " << sqrt(l2R_new) << std::endl;

		cudaDeviceSynchronize();
		lIter++;
	}

	// Compute Ax
	double* temp_Ax;
	cudaMalloc(&temp_Ax, numRows * sizeof(double));
	cublasDgemv(m_cuBLAS, CUBLAS_OP_N, numRows, numCols, &fAlpha_blas, g_A, numRows, g_X, 1, &fBeta_blas, temp_Ax, 1);
	_CUDA_CALL(cudaDeviceSynchronize());

	// Compute y - Ax (note: fR initially contains y)
	double negOne = -1.0f;
	cublasDaxpy(m_cuBLAS, numRows, &negOne, temp_Ax, 1, g_Y, 1);
	_CUDA_CALL(cudaDeviceSynchronize());

	// Compute and print the norm of the residual
	double residual_norm;
	cublasDnrm2(m_cuBLAS, numRows, g_Y, 1, &residual_norm);

	std::cout << "Final least squares residual norm: " << residual_norm << std::endl;
	printDeviceVector(g_X, numCols, "g_X");	
	exit(1);
}

__global__ void fillMatrixKernel(float *g_A, float *fA, long *lList, long lW, long lC, long lM, long lT, long numRows, long numCols, long AmntToAdd) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numRows * AmntToAdd) {
		int ii = idx / numRows;
		int row = idx % numRows;
		long index = numCols - AmntToAdd + ii;
		long sampleIndex = lList[index] / lT;
		long templateIndex = lList[index] % lT;
		if (row >= sampleIndex * lC && row < (sampleIndex + min(lM, lW - sampleIndex)) * lC) {
			g_A[index * lW * lC + row] = fA[templateIndex * lM * lC + (row - sampleIndex * lC)];
		}
	}
}

#define CALVINS
#ifdef CALVINS
// example call:
// cu_cgPseudoInverse(m_gfD2, lInds, m_gfV, m_gfX, lW, m_lM, m_lT, m_lC, lNInds, lAmountToAdd);
// first arg: templates
// second args: indices of templates
// third arg: Y
// fourth arg: X
void OnlineSpikes::cu_cgPseudoInverse(
	float *fA, // template matrix m_gfD2
	long *lList, // list of spike indices (timestamp & template) that spiked
	float *g_Y, // output: the residual y you get from applying the matrix to the final x
	float *g_X, // output: the resulting x you get from pseudoinversing
	long lW, // # of scans in window
	long lM, // samples per template
	long lT, // number of templates
	long lC, // number of channels
	long lNList, // number of spikes to add ???
	long AmntToAdd // 
) {

	static const char *ptLabel = { "OnlineSpikes::cu_cgPseudoInverse" };

	//Parameter
	float tau = 1e-3f;    //Halt criteria;

	float *g_A = m_fcgA; // matrix A
	float *g_d = m_fcgd; // direction vector
	float *g_q = m_fcgq; // q = A^T A d
	float *g_r = m_fcgs; // residual

	//parameters to make sure cublasSgemm works as intended
	const float fAlpha_blas = 1.0f;
	const float fAlpha_blas_negative = -1.0f;
	const float fBeta_blas = 0.0f;

	float falpha_L2, fbeta_L2;

	long numRows = lW * lC;
	long numCols = lNList;

	//----------------------------------- Fill Matrix/Vectors ---------------------
	for (int ii = 0; ii < AmntToAdd; ii++) {
		long index = numCols - AmntToAdd + ii; // the column we're inserting into
		long sampleIndex = lList[index] / lT;
		long templateIndex = lList[index] % lT;
		//std::cout << "Template index = " << templateIndex << std::endl;
		//std::cout << "Sample index = " << sampleIndex << std::endl;

		/*
		_CUDA_CALL(cudaMemcpyAsync(
			g_A + templateIndex * lW * lC + sampleIndex * lC,
			fA + templateIndex * lM * lC, 
			min(lM, lW - sampleIndex) * lC * sizeof(float), // copy an entire template as a column of gA (cuBLAS is column-majored)
			cudaMemcpyDeviceToDevice
		));*/
		
		_CUDA_CALL(cudaMemcpyAsync(
			g_A + index * lW * lC + sampleIndex * lC, // dest
			fA + templateIndex * lM * lC, // src
			min(lM, lW - sampleIndex) * lC * sizeof(float), // count
			cudaMemcpyDeviceToDevice
		));

	}

	_CUDA_CALL(cudaDeviceSynchronize());

	// x = zero-vector
	_CUDA_CALL(cudaMemsetAsync(g_X, 0, numCols * sizeof(float)));

	// r = A^t y
	cublasSgemv(m_cuBLAS, CUBLAS_OP_T, numRows, numCols, &fAlpha_blas, g_A, numRows, g_Y, 1, &fBeta_blas, g_r, 1);
	_CUDA_CALL(cudaDeviceSynchronize());

	// Set d = r
	_CUDA_CALL(cudaMemcpyAsync(g_d, g_r, numCols * sizeof(float), cudaMemcpyDeviceToDevice));
	_CUDA_CALL(cudaDeviceSynchronize());

	float residual_norm_square_new, residual_norm_square_old, residual_norm_square_init;

	// delta_new = <r, r>
	float rr_dot{};
	if (cublasSdot(m_cuBLAS, numCols, g_r, 1, g_r, 1, &rr_dot) != CUBLAS_STATUS_SUCCESS) {
		_RUN_ERROR(ptLabel, "failed to calculate dot product (d^T q)");
	}
	_CUDA_CALL(cudaDeviceSynchronize());

	if (isnan(rr_dot))
		throw std::runtime_error("Failed to calculate L2 norm of residual\n");

	//std::cout << "Initial L_2(residue)^2 = " << rr_dot << std::endl;
	residual_norm_square_init = rr_dot;
	residual_norm_square_new = residual_norm_square_init;
	residual_norm_square_old = residual_norm_square_init;

	_CUDA_CALL(cudaDeviceSynchronize());

	for (long lIter = 0; lIter < m_lMaxIterPinv; lIter++) {
		// exit if delta_new <= eps^2 delta_0,=
		if (residual_norm_square_new <= tau * tau * residual_norm_square_init) break;

		// q = A^T * A * d
		_CUDA_CALL(cudaMemsetAsync(g_q, 0, numCols * sizeof(float)));
		cudaDeviceSynchronize();
		cublasSgemv(m_cuBLAS, CUBLAS_OP_N, numRows, numCols, &fAlpha_blas, g_A, numRows, g_d, 1, &fBeta_blas, g_q, 1);
		cublasSgemv(m_cuBLAS, CUBLAS_OP_T, numRows, numCols, &fAlpha_blas, g_A, numRows, g_q, 1, &fBeta_blas, g_q, 1);
		_CUDA_CALL(cudaDeviceSynchronize());

		// alpha = l2r_new / d^T q
		float d_dot_q;
		cublasSdot(m_cuBLAS, numCols, g_d, 1, g_q, 1, &d_dot_q);
		falpha_L2 = residual_norm_square_new / d_dot_q;
		_CUDA_CALL(cudaDeviceSynchronize());

		// x = x + alpha * d
		cublasSaxpy(m_cuBLAS, numCols, &falpha_L2, g_d, 1, g_X, 1);
		_CUDA_CALL(cudaDeviceSynchronize());

		// r = r - alpha * q
		float negAlpha = -falpha_L2;
		cublasSaxpy(m_cuBLAS, numCols, &negAlpha, g_q, 1, g_r, 1);

		// delta_new = <r, r>
		residual_norm_square_old = residual_norm_square_new;	
		cublasSdot(m_cuBLAS, numCols, g_r, 1, g_r, 1, &residual_norm_square_new);
		_CUDA_CALL(cudaDeviceSynchronize());

		// beta = delta_new / delta_old
		fbeta_L2 = residual_norm_square_new / residual_norm_square_old;

		// d = r + beta * d
		cublasSscal(m_cuBLAS, numCols, &fbeta_L2, g_d, 1);
		cublasSaxpy(m_cuBLAS, numCols, &fAlpha_blas, g_r, 1, g_d, 1);

		_CUDA_CALL(cudaDeviceSynchronize());

		// Print diagnostics
		//std::cout << "Iteration " << lIter << ": alpha = " << falpha_L2
	//		<< ", beta = " << fbeta_L2 << ", residual = " << sqrt(residual_norm_square_new) << std::endl;

		cudaDeviceSynchronize();
	}

	// Compute Ax
	cudaMalloc(&m_gAx, numRows * sizeof(float));
	cublasSgemv(m_cuBLAS, CUBLAS_OP_N, numRows, numCols, &fAlpha_blas, g_A, numRows, g_X, 1, &fBeta_blas, m_gAx, 1);
	_CUDA_CALL(cudaDeviceSynchronize());

	// Compute y - Ax (note: g_Y initially contains y)
	float negOne = -1.0f;
	cublasSaxpy(m_cuBLAS, numRows, &negOne, m_gAx, 1, g_Y, 1);
	_CUDA_CALL(cudaDeviceSynchronize());
	// Now g_Y contains y - Ax (the residual of the original system)
}
#endif
#ifndef CALVINS
// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::cu_cgPseudoInverse()
//
// Description  : Calculate Pseudo Inverse on CUDA
// https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf (see page 57). 
// NOTE: this is mathematically not really the correct way to look at it, but it is the same algorithm/code/
// NOTE2: This does not work for cosamp, for it to work that way you need to rewrite the first part as currently the vector concatenated to the matrix, with cosamp you want to remake the whole matrix
// NOTE3: On 6/27/2022, deleted cgPsedudoInverse, a non-CUDA implementation. If have access to git history, may be helpful looking at that.
// ------------------------------------------------------------------------------
void OnlineSpikes::cu_cgPseudoInverse(float *fA, long *lList, float *fR, float *fX,
	long lN, long lM, long lT, long lC, long lNList, long AmntToAdd) {

	static const char *ptLabel = { "OnlineSpikes::cu_cgPseudoInverse" };

	long ll, ln;
	long lIter = 0;

	//Parameter
	float tau = 1e-10f;    //Halt criteria;

	long lda, ldb;

	float *gA = m_fcgA;
	float *gd = m_fcgd;
	float *gq = m_fcgq;
	float *gs = m_fcgs;

	//parameters to make sure cublasSgemm works as intended
	const float fAlpha_blas = 1.0f;
	const float fAlpha_blas_negative = -1.0f;
	float fBeta_blas = 0.0f;

	float falpha_L2, fbeta_L2;

	lda = lN * lC;
	ldb = lNList;

	// Zero out fX
	_CUDA_CALL(cudaMemsetAsync(fX, 0, lNList * sizeof(float)));

	//----------------------------------- Fill Matrix/Vectors ---------------------


		//temporary pointers 
	float *gA_c, *fA_c;

	for (int ii = 0; ii < AmntToAdd; ii++) {
		long index = ldb - AmntToAdd + ii;
		ln = lList[index] / lT; //Timepoint
		ll = lList[index] % lT;

		fA_c = fA + ll * lM * lC;
		gA_c = gA + index * lN * lC + ln * lC;

		_CUDA_CALL(cudaMemcpyAsync(gA_c, fA_c, min(lM, lN - ln) * lC * sizeof(float), cudaMemcpyDeviceToDevice));

	}

	cudaDeviceSynchronize();
	// Calculate d =  A^T * Y    (b x a) X (a x 1) = (b x 1)
	if (cublasSgemv(m_cuBLAS, CUBLAS_OP_T, lda, ldb, &fAlpha_blas, gA, lda, fR, 1, &fBeta_blas, gd, 1) != CUBLAS_STATUS_SUCCESS)
		_RUN_ERROR(ptLabel, "failed to calculate Vector Transpose");

	//note that these are not squared as opposed to the 
	float l2R_new, l2R_old, l2R_ini, l2R_temp;

	// Find the L2 norm of vector d
	if (cublasSnrm2(m_cuBLAS, ldb, gd, 1, &l2R_ini) != CUBLAS_STATUS_SUCCESS)
		_RUN_ERROR(ptLabel, "Failed to calculate initial L2 norm");

	l2R_ini = l2R_ini * l2R_ini;
	l2R_new = l2R_ini;
	l2R_old = l2R_ini;

	while ((l2R_new > tau * l2R_ini) && (lIter < m_lMaxIterPinv)) {

		//q = A * d       (a x b) X (b x 1) = (a x 1)
		if (cublasSgemv(m_cuBLAS, CUBLAS_OP_N, lda, ldb, &fAlpha_blas, gA, lda, gd, 1, &fBeta_blas, gq, 1) != CUBLAS_STATUS_SUCCESS) {
			_RUN_ERROR(ptLabel, "failed to calculate Vector Transpose");
		}

		//L2 norm
		if (cublasSnrm2(m_cuBLAS, lda, gq, 1, &l2R_temp) != CUBLAS_STATUS_SUCCESS) {
			_RUN_ERROR(ptLabel, "Failed to calculate new L2 norm");
		}

		falpha_L2 = l2R_new / (l2R_temp * l2R_temp);

		// X_new = x_old + alpha * d
		if (cublasSaxpy(m_cuBLAS, ldb, &falpha_L2, gd, 1, fX, 1) != CUBLAS_STATUS_SUCCESS) {
			_RUN_ERROR(ptLabel, "failed to calculate Saxpy (X_new = X_old + alpha * d");
		}

		// Calculate residual
		// R_new = b - Ax
		if (cublasSgemv(m_cuBLAS, CUBLAS_OP_N, lda, ldb, &fAlpha_blas_negative, gA, lda, fX, 1, &fAlpha_blas, fR, 1) != CUBLAS_STATUS_SUCCESS) {
			_RUN_ERROR(ptLabel, "failed to calculate new residual by matrix multiplication.");
		}

		//s = A^T * R   (b x a) X (a x 1) = (b x 1)
		if (cublasSgemv(m_cuBLAS, CUBLAS_OP_T, lda, ldb, &fAlpha_blas, gA, lda, fR, 1, &fBeta_blas, gs, 1) != CUBLAS_STATUS_SUCCESS) {
			_RUN_ERROR(ptLabel, "failed to calculate Vector Transpose");
		}

		l2R_old = l2R_new;

		//L2 norm
		if (cublasSnrm2(m_cuBLAS, ldb, gs, 1, &l2R_temp) != CUBLAS_STATUS_SUCCESS) {
			_RUN_ERROR(ptLabel, "Failed to calculate new L2 norm");
		}

		//Set new L2
		l2R_new = l2R_temp * l2R_temp;

		//Beta 
		fbeta_L2 = l2R_new / l2R_old;

		//---------------- d = s + beta * d
		if (cublasSscal(m_cuBLAS, ldb, &fbeta_L2, gd, 1) != CUBLAS_STATUS_SUCCESS) {
			_RUN_ERROR(ptLabel, "failed to calculate Scal");
		}

		if (cublasSaxpy(m_cuBLAS, ldb, &fAlpha_blas, gs, 1, gd, 1) != CUBLAS_STATUS_SUCCESS) {
			_RUN_ERROR(ptLabel, "failed to calculate saxpy (d = s + beta * d)");
		}

		lIter++;
		//	std::cout << "l2R_new = " << l2R_new << std::endl;

	}

}
#endif

// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::MoveFifo
//
// Description  : Move data from the back to the front (will be used asynchronysly most of the time)
//
// ------------------------------------------------------------------------------
void OnlineSpikes::MoveFifo() {
	for (long i = 0; i < m_lC * m_lMinWindow; i++)
		m_fY[i] = m_fY[m_lC * (m_lW - m_lMinWindow) + i];
}


// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::maxInd
//
// Description  : Find the index corresponding to the highest value
//
// ------------------------------------------------------------------------------
long OnlineSpikes::maxInd(float *vals, long l) {

	// vals is the array
	// l is the size of the array
	long lI, lMax = -1;
	double valMax;

	for (lI = 0; lI < l; lI++)
		if ((lMax < 0) || (vals[lI] > valMax))
			valMax = vals[lMax = lI];

	return lMax;
}

// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::maxInd
//
// Description  : Find the index corresponding to the highest value
//
// ------------------------------------------------------------------------------
long OnlineSpikes::maxInd(float *vals, long l, long inc) {

	// vals is the array
	// l is the size of the array
	long lI, lMax = -1;
	double valMax;

	for (lI = 0; lI < l; lI += inc)
		if ((lMax < 0) || (vals[lI] > valMax))
			valMax = vals[lMax = lI];

	return lMax;
}


// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::minInd
//
// Description  : Find the index corresponding to the lowest value
//
// ------------------------------------------------------------------------------
long OnlineSpikes::minInd(float *vals, long l) {

	// vals is the array
	// l is the size of the array
	long lI, lMin = 1000;
	double valMin;

	for (lI = 0; lI < l; lI++)
		if ((lMin > -1000) || (vals[lI] < valMin))
			valMin = vals[lMin = lI];

	return lMin;
}


// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::minInd
//
// Description  : Find the index corresponding to the lowest value
//
// ------------------------------------------------------------------------------
long OnlineSpikes::minInd(float *vals, long l, long inc) {

	// vals is the array
	// l is the size of the array
	long lI, lMin = 1000;
	double valMin;

	for (lI = 0; lI < l; lI += inc)
		if ((lMin > -1000) || (vals[lI] < valMin))
			valMin = vals[lMin = lI];

	return lMin;
}

// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::FindChanNumbers
//
// Description  : Fill the given array by the channel number where the neuron (probably) activates
//
// ------------------------------------------------------------------------------
void OnlineSpikes::FindChanNumbers(float *Templates, std::vector<double> &ChanFiller, std::string form) {
	static const char *ptLabel = { "OnlineSpikes::FindChanNumbers" };

	if (form == "Max") {
		//This is currently done by just taking the maximum value of the template. This can be done in any other way.
		for (int Temp = 0; Temp < m_lT; Temp++) {
			long maxval = maxInd(Templates + (Temp * m_lC * m_lM), m_lC * m_lM);
			ChanFiller.push_back(maxval % m_lC);
		}
	}

	
	else if (form == "MinMax") {
		// max min
		float minMaxTemp = 0;

		for (int Temp = 0; Temp < m_lT; Temp++) {
			float minMax = 0;
			float *temp_ptr = Templates + (Temp * m_lC * m_lM);

			long chanIndex = 0;

			for (int chan = 0; chan < m_lC; chan++) {
				//Calculate minmax per channel per template
				minMaxTemp = temp_ptr[maxInd(temp_ptr, m_lM * m_lC, m_lC)] - temp_ptr[minInd(temp_ptr, m_lM * m_lC, m_lC)];

				std::cout << "Max in this channel: " << minMaxTemp << std::endl;
				std::cout << "Overal Max: " << minMax << std::endl;

				//find the maximum minmax
				if (minMaxTemp > minMax) {
					minMax = minMaxTemp;					
					chanIndex = chan;
				}

				// Increment the pointer to the start of next channel
				temp_ptr += m_lC;

			}
			ChanFiller.push_back(chanIndex);
			
		}
	}

	else {
		_RUN_ERROR(ptLabel, "Undefined type of finding where the neuron is located, try \"Max\", or \"MinMax\" )");
	}
	

}


// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::Get functions
//
// Description  : Get functions which are mainly needed by the GUI
//
// ------------------------------------------------------------------------------
SorterParameters OnlineSpikes::GetSorterParams() {
	SorterParameters params = {	m_lT,
								m_lC,
								m_lM,
								m_lN,
								m_fSamplingRate,
								m_dChanActivated,
								m_vTemplateMap
	};
	return params;
}


void OnlineSpikes::FindMax(float *Input, long Length, int *Ind, float *Val) {
	static const char *ptLabel = { "OnlineSpikes::FindMax" };

	//static float sfMaxVal[1] = { 0 };
	//static int sfMaxInd[1] = { 0 };
	float sfMaxVal{};
	int sfMaxInd{};

	NppStatus err = nppsMaxIndx_32f(Input, Length, m_gfValBuf, m_giIndBuf, m_gNpp8MaxBufffer);

	if (err != NPP_SUCCESS) {
		_RUN_ERROR(ptLabel, "Error in finding the maximum value.");
	}

	_CUDA_CALL(cudaMemcpyAsync(&sfMaxVal, m_gfValBuf, sizeof(float), cudaMemcpyDeviceToHost));
	_CUDA_CALL(cudaMemcpy(&sfMaxInd, m_giIndBuf, sizeof(int), cudaMemcpyDeviceToHost));

	*Val = sfMaxVal;
	*Ind = sfMaxInd;

}


void OnlineSpikes::FindMin(float *Input, long Length, int *Ind, float *Val) {
	static const char *ptLabel = { "OnlineSpikes::FindMin" };

	//static float sfMinVal[1] = { 0 };
	//static int sfMinInd[1] = { 0 };
	float sfMinVal{};
	int sfMinInd{};

	NppStatus err = nppsMinIndx_32f(Input, Length, m_gfValBuf, m_giIndBuf, m_gNpp8MinBufffer);

	if (err != NPP_SUCCESS) {
		_RUN_ERROR(ptLabel, "Error in finding the minimum value.");
	}

	_CUDA_CALL(cudaMemcpyAsync(&sfMinVal, m_gfValBuf, sizeof(float), cudaMemcpyDeviceToHost));
	_CUDA_CALL(cudaMemcpy(&sfMinInd, m_giIndBuf, sizeof(int), cudaMemcpyDeviceToHost));

	*Val = sfMinVal;
	*Ind = sfMinInd;

}

void OnlineSpikes::P2P_calc(float *input, long length, float *P2P) {

	//static float sfMaxVal[1] = { 0 };
	//static float sfMinVal[1] = { 0 };
	float sfMaxVal{};
	float sfMinVal{};

	//static int sfMaxInd[1] = { 0 };
	//static int sfMinInd[1] = { 0 };
	int sfMaxInd{};
	int sfMinInd{};

	FindMax(input, length, &sfMaxInd, &sfMaxVal);
	FindMin(input, length, &sfMinInd, &sfMinVal);


	*P2P = sfMaxVal - sfMinVal;
}

// Helper function to insert in order
template <class T>
typename std::vector<T>::iterator insert_sorted(std::vector<T> &vec, T const& item){
	return vec.insert(std::upper_bound(vec.begin(), vec.end(), item), item);
}

// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes::saveSpikes
//
// Description  : Save Spike Events
//
// ------------------------------------------------------------------------------
void OnlineSpikes::saveSpikes(long lNInds, long *lInds, long lStreamSampleCtOffset, long lEndValid, std::vector<long>& Times, std::vector<long>& Templates, std::vector<float>& Amplitudes) {
	static const char *ptLabel = { "OnlineSpikes::saveSpikes" }; _UNUSED(*ptLabel);
	long  ln;
	long  ll;
	float fa;

	//Loop over found spikes
	for (long lI = 0; lI < lNInds; lI++) {
		ln = lInds[lI] / m_lT; //Which timepoint
		ll = lInds[lI] % m_lT; //Which template/neuron

		ln *= m_lDownsampling;

		//This means that if ln => lEndValid, skip inserting into vectors
		if (ln >=  lEndValid)
			continue;

		fa = m_fX[ lInds[lI] ];

		//Put time, template, and amplitudes into respective vectors, sorted by time
		std::vector<long>::iterator it = insert_sorted(Times, lStreamSampleCtOffset + ln);
		size_t pos = it - Times.begin(); // Get position of where new time was inserted
		Templates.insert(Templates.begin() + pos, m_vTemplateMap[ll]);
		Amplitudes.insert(Amplitudes.begin() + pos, fa);
	}
}

