/*
 * CudaOMP.h
 *
 *  Created on: Dec 17, 2018
 *      Author: basti
 */

#ifndef CUDAOMP_H_
#define CUDAOMP_H_

#include <vector>
#include <thread>
#include <complex>
#include <random>
#include <future>
#include <fstream>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <npp.h>

#include "CNPY/cnpy.h"
#include "dataSocket.h"
#include "myCudnnConvolution.h"
#include "myGPUhelpers.h"
#include "../Networking/inputParameters.h"
#include "../Networking/Sock.h"
#include "../Networking/sorterParameters.h"
#include "../Networking/FragmentManager.h"

#ifndef CMD_ESCAPE_STRING
#define CMD_ESCAPE_STRING "#EXIT#"
#endif // !CMD_ESCAPE_STRINGs




// ------------------------------------------------------------------------------
//
// Name			: OnlineSpikes
//
// Description  : Main Class for Online Spike Sorting
//	
// ------------------------------------------------------------------------------
class OnlineSpikes {
public:
	OnlineSpikes(InputParameters Params, sockaddr_in mainAddr, DataSocket* m_NC);
	OnlineSpikes(const OnlineSpikes&) = delete;
	OnlineSpikes& operator= (const OnlineSpikes&) = delete;
	~OnlineSpikes();

	// =================================================================
    // Main processing function
	// ================================================================= //TODO move functions to protected if possible
	void	runSpikeSorting();
	void	FindChanNumbers(float *Templates, std::vector<double> &ChanFiller, std::string form = "Max");
	
	//GUI Getters
	SorterParameters GetSorterParams();
	long getChannelCount() { return m_lC; };

protected:
	// =================================================================
    // Helper processing functions
	// =================================================================
	long	processData( float *fY, long lW, long* lInds );
    void	saveSpikes(long lNInds, long *lInds, long lStreamSampleCtOffset, long lEndValid, std::vector<long>& Times, std::vector<long>& Templates, std::vector<float>& Amplitudes);
	void	processNidqStream();

	void    MoveFifo();

	// =================================================================
    // Helper functions for processing online program // TODO help Bram here
	// =================================================================
	void normalize_cols(float* d_matrix, int num_rows, int num_cols);
	void test_cgPseudoInverse();
    void cu_cgPseudoInverse(float *dA, long *lList, float *fR, float *fX, long lN, long lM, long lT, long lC, long lNList, long AmntToAdd);
	void test_convolution();
	void computeDiagonalPreconditioner(const float* A, int m, int n, float* M);
	double computeConditionNumber(const double* A, int m, int n);

	void	P2P_calc(float *input, long length, float *P2P);

	// =================================================================
    // Static basic math functions
	// =================================================================
    // TODO: update Static basic math functions with CUDA gemm
	static long   		maxInd(float *vals, long l);
	static long   		maxInd(float *vals, long l, long inc);
	static long			minInd(float * vals, long l);
	static long			minInd(float * vals, long l, long inc);
	void FindMax(float *Input, long Length, int *Ind, float *Val);
	void FindMin(float *Input, long Length, int *Ind, float *Val);

	// =================================================================
	// Online Processing Settings
	// =================================================================
	long   m_lMinWindow;
	long   m_lMaxWindow;

	long   m_lMaxIterProcessing;
	long   m_lMaxIterPinv;
	double m_dTauProcessing;
	double m_dThresProcessing;
	double m_dRatToMax;
	long   m_lRatToMaxTimes;
	long   m_lTimeBehind;
	bool   m_bSmallskip;
	bool   m_bIsSendingFeedback;
	uint16 m_uSelectedDevice;

	int	   m_iNidqRefreshRate;
	t_sglxconn S;
	
	// =================================================================
	/* Online Processing Parameters
	 an 'm' before a parameter/variable indicates that is a member variable
	 g indicates in graphics card memory
	 the value behind the '_' indicates the type
	 so m_lN, can be just read as N which is long and is a member variable*/
	 //=================================================================
	// TODO make lN and lW non member, local variables
	long	m_lN; 				// lN Number of Samples 			// TODO remove m_lN as member variable
	long 	m_lM;				// lM Samples per Template
	long 	m_lT;				// lT Number of Templates
	long 	m_lC;				// lC Number of Channels
	long 	m_lW;				// lW Number of Scans in Window		// TODO remove m_lW as member variable

	long	m_lNInds;			// Number of identified events
	long   *m_lInds;			// List of identified events

	double  m_dVRMS;			// Root mean square 
	float   m_fP2P;

	long	m_lCtDC;			// Total Samples processed

	long 	m_lDownsampling;	// Temporal undersampling
	long 	m_lM_ReadIn;		// Samples per Template without undersampling
	long	m_lSkipCounter;		// Counter which counts the amount of skips that have happened
	long	m_lSpikeRateWindow; // Amount of Seconds over which the spike rate is averaged
	long	m_lRedundancy;


	// =================================================================
	// Processing Objects
	// =================================================================
	DataSocket*			m_mNC;			// Object for data accquisition
	Sock				m_imecSock;		// Network socket for imec data communication
	Sock				m_nidqSock;		// Network socket for nidq data communication
	FragmentManager		imecFm;			// Layer on top of socket to handle assembly of large imec packets
	FragmentManager		nidqFm;			// Layer on top of socket to handle assembly of large nidq packets
	myCudnnConvolution	m_mCC;			// Object for CUDA Convolution
	cublasHandle_t		m_cuBLAS;		// Object for CUDA cuBLAS Handle
	cusolverDnHandle_t  m_cuSolver;
	myDCRemover			m_DCRem;		// Object to remove DC with

	// Thread
	std::thread m_nidqProcessor;
	sockaddr_in m_decoderImecAddr; // TODO move these
	sockaddr_in m_decoderNidqAddr;


	// =================================================================
	// Host processing data containers
	// =================================================================
	float 			*m_fX;	// Spike Events:     lT x lN
	float			*m_fY;	// Recording Data:   lC x lN
	std::vector<short> sglx_vector;
	float 			*m_fD;	// Template Matrix:  lC x lM x lT
	float 			*m_fD2;	// Template Matrix:  lC x lM x lT
	float 			*m_fW;	// Whitening Matrix: lC x lC

	//cg arrays
	float			*m_fcgA;
	float			*m_fcgd;
	float			*m_fcgq;
	float			*m_fcgs;

	std::vector<double> m_dChanActivated;
	std::vector<int> m_vChannelMap;
	std::vector<int> m_vTemplateMap;

	// =================================================================
	// Device processing data containers
	// =================================================================
	float 			*m_gfD;  // Template Matrix:  lC x lM x lT
	float 			*m_gfU;
	float 			*m_gfV;
	float			*m_gfD2; // Template Matrix same as m_fD2, but always on gpu to increase transfer speed
	float			*m_gfY;
	float			*m_gfX; //retrieved spikes on gpu
	float			*m_gfW; //whitening matrix on gpu
	float			*m_gfYW; //gpu, y=data, whitened(whitened in place)
	float			*m_gfDC; //learned DC offset on GPU
	float           *m_gAx; // intermediate result of matrix multiplication Ax

	Npp8u           *m_gNpp8MaxBufffer;
	Npp8u           *m_gNpp8MinBufffer;
	float			*m_gfValBuf;
	int				*m_giIndBuf;

	// =================================================================
	// Display Settings
	// =================================================================
	float					m_fSamplingRate;

	// =================================================================
	// Debug
	// =================================================================
	std::string		m_sPythonVerifierDir;
};


#endif /* CUDAOMP_H_ */
