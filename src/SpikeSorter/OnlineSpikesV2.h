#pragma once
#include <vector>
#include <thread>
#include <complex>
#include <random>
#include <future>
#include <fstream>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <npp.h>

#include <complex>
#include "CNPY/cnpy.h"
#include "dataSocket.h"
#include "myCudnnConvolution.h"
#include "myGPUhelpers.h"
#include "../Networking/inputParameters.h"
#include "../Networking/Sock.h"
#include "../Networking/sorterParameters.h"
#include "../Networking/FragmentManager.h"
#include "OnlineSpikesV2MemoryList.h"
#include "TensorWrapper.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

class OnlineSpikesV2
{
public:
	OnlineSpikesV2(InputParameters Params, sockaddr_in mainAddr, DataSocket* m_NC);
	~OnlineSpikesV2();
	void runSpikeSorting();

private:
	void initializeSorter(InputParameters params);
	void establishDecoderConnection(sockaddr_in mainAddr);
	void loadTemplatesShape(std::string filepath); 
	void loadKilosortParameters(std::string directoryPath);
	void loadPreclusterShapes(std::string filepath);
	void initializeStaticMemory(InputParameters params);
	void allocateMemory();
	void loadTemplates(std::string filepath);
	void loadWhitening(std::string filepath);
	void loadChannelMap(std::string filepath);
	void loadTemplateMap(std::string filepath);
	void loadKilosortTrainingData(std::string directoryPath);
	void loadKilosortClusteringData(std::string directoryPath);

	void fwdMaxPool1d(float* d_matrix, float* d_result, int len, int width);
	void findMaxAbs(float *input, long length, int *ind, float *val);
	void findMax(float *Input, long Length, int *Ind, float *Val);
	void findMin(float *Input, long Length, int *Ind, float *Val);
	float P2P_calc(float *input, long length);
	long kilosortMatchingPursuit(float* d_batch, long currBatchNumSamples);
	void computeClosestClusters(long currBatchNumSamples, long numSpikes);
	void highpassFilter(float* d_batch, int C, int currBatchNumSamples, float sampling_freq, float frequency_low);
	int closestCluster(const float x, const float y);

	SorterParameters getSorterParams();
	void writeSpikesToFile(std::vector<long> spikeTimes, std::vector<long> spikeTemplates, std::vector<float> spikeAmplitudes);
	void saveSpikes(long lNInds, long lStreamSampleCtOffset, long lEndValid, std::vector<long>& Times, std::vector<long>& Templates, std::vector<float>& Amplitudes);

	// Debug
	std::string ossOutputDir;
	std::ofstream spikesFileOut;
	long recordingOffset;

	// Which SpikeGLX probe we are grabbing batches from
	int substream;

	// Host + device pointers, defined in OnlineSpikesV2MemoryList.h --- read about X-macros online
	#define X(type, name, memType, size) type *name;
		MEMORY_VARIABLES
	#undef X

	myCudnnConvolution	cudnnConvObj; // object to handle convolutions
	cublasHandle_t		cublasHandle; // handle for cublas computations
	cusolverDnHandle_t  cuSolverHandle; // handle for cusolver computations
	int filterLen;

	static const int NOT_MAPPED = -1;
	long K; // number of principal components
	long unclu_T; // number of templates prior to Kilosort's clustering
	long T; // number of templates
	long M; // number of samples in a template
 	long C; // number of channels
	long W; // number of samples per batch
	long nt0min; // kilosort param
	long numNearestChans; // kilosort param
	long Th_learned; // kilosort param: how strong a spike has to be to be a spike
	long dt; // kilosort param: same-template spikes within dt are duplicates
	long minWindow;
	long maxWindow;
	long redundancy; // # of copies of memory to allocate because we don't know number of spikes apriori
	long timeBehind; // Time (ms) we are allowed to be behind from SGLX; 0: skip batches if behind, >= 100'000: no skip
	long downsampling; // Factor we are temporally downsampling
	long lookback; // # of samples to look into the past for each batch to ensure we're capturing all of the data. double counting should be culled in saveSpikesw
	float samplingRate; // IMEC sampling rate (hz)
	int nidqRefreshRate; // NIDQ refresh rate (no idea what units or anything)
	bool smallSkip;

	// Host pointers
	long  numSpikes; // number of spikes found
	std::vector<int> channelMap; // the set of channels this sorter will operate on
	std::vector<int> templateMap; // the set of neurons this sorter will operate on
	std::vector<float> xs;
	std::vector<float> ys;
	std::vector<long> lastSpikeTime; // size T: will remove duplicate spikes

	// Device pointers
	float p2p;
	long ctDc; // no idea what this is supposed to mean
	long latestCt;

	/* OMP */
	double rootMeanSquared;

	// Networking
	Sock imecSock; // IMEC socket
	Sock nidqSock; // NIDQ socket
	FragmentManager imecFm; // handles retransmission of network fragments
	FragmentManager nidqFm; // handles retransmission of network fragments
	sockaddr_in decoderImecAddr;
	sockaddr_in decoderNidqAddr;
	DataSocket* sglxSock;

	// Leftover stuff from previous code
	std::vector<double> activeChannels;

	// Thrust vector for matching to find local maxima, change later to normal device vector
	thrust::device_vector<long> d_spikeIndices;
};

