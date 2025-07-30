#ifndef ThresholdCrossing_H_
#define ThresholdCrossing_H_

#include "dataSocket.h"
#include "../Networking/inputParameters.h"
#include "../Networking/Sock.h"
#include "../Networking/sorterParameters.h"
#include "../Networking/FragmentManager.h"

 // ------------------------------------------------------------------------------
 //
 // Name			: ThresholdCrossing
 //
 // Description  : Main Class for Online Threshold Crossing
 //
 // ------------------------------------------------------------------------------
class ThresholdCrossing {
public:
	ThresholdCrossing(InputParameters Params, sockaddr_in mainAddr, DataSocket** &mNC);
	~ThresholdCrossing();

	// =================================================================
	// Main processing function
	// ================================================================= //TODO move functions to protected if possible
	void	runThresholdCrossing();

	//GUI Getters
	SorterParameters GetSorterParams();
	long getChannelCount() { return m_lC; };

protected:
	// =================================================================
	// Helper processing functions
	// =================================================================
	//long	processData(float *fY, long lW, long* lInds);
	//long	processData_TC(float *fY, long lW, long* lInds, float threshold);
	//void	saveSpikes(long lNInds, long *lInds, long lStreamSampleCtOffset, long lEndValid, std::vector<long>& Times, std::vector<long>& Templates, std::vector<float>& Amplitudes);
	//void	processNidqStream();

	void    MoveFifo();

	// =================================================================
	// Online Processing Settings
	// =================================================================

	long   m_lMinWindow;
	long   m_lMaxWindow;
	float  m_fThresholdStd;

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
	long 	m_lC;				// lC Number of Channels
	long 	m_lW;				// lW Number of Scans in Window		// TODO remove m_lW as member variable

	// =================================================================
	// Processing Objects
	// =================================================================
	DataSocket*			m_mNC;			// Object for data accquisition
	Sock				m_imecSock;		// Network socket for imec data communication
	Sock				m_nidqSock;		// Network socket for nidq data communication
	FragmentManager		imecFm;
	FragmentManager		nidqFm;
	// Thread
	std::thread m_nidqProcessor;

	sockaddr_in m_decoderImecAddr; // TODO move these
	sockaddr_in m_decoderNidqAddr;


	// =================================================================
	// Host processing data containers
	// =================================================================
	//float 			*m_fX;	// Spike Events:     lT x lN
	float			*m_fY;	// Recording Data:   lC x lN

	std::vector<double> m_dChanActivated;  // Channel numbers for the GUI
	std::vector<int>    m_vChannelMap;     // dummy channel map
	// =================================================================
	// Display Settings
	// =================================================================
	float					m_fSamplingRate;
};
#endif /* ThresholdCrossing_H_ */