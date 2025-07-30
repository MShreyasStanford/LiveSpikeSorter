/*
 * myCudnnConvolution.h
 *
 *  Created on: Jan 28, 2019
 *      Author: basti
 */

#ifndef MYCUDNNCONVOLUTION_H_
#define MYCUDNNCONVOLUTION_H_

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <memory>
#include <vector>

#include <cuda.h>
#include <cudnn.h>


class myCudnnConvolution {
public:
	myCudnnConvolution( int iValue );
	~myCudnnConvolution();

	// Perform efficient convolution
	void fwdConvolve(float *fX, float *fW, float *fY, long lN, long lM, long lC, long lT);

	void fwdConvolvePCS(float* d_batch, float* d_wPCA, float* d_B, int K, int M, int C, int currBatchNumSamples);

	// Query size parameters
	void getOutputSizes( int *out_n, int *out_c, int *out_h, int *out_w );

    // Define descriptors and allocate workspace
	void setSizes(long lN, long lM, long lC, long lT);
	void setSizesPCs(long lN, long lM, long lC, long lT);
	void setSizesPCsV2(long numPcs, long C, long M, long currBatchNumSamples);
	
	// Select device
	bool selectDevice( int iValue );

	// Get device
	int getDevice( );

protected:
	// Handles
	::cudnnHandle_t cudnn;

	// Descriptors
    ::cudnnTensorDescriptor_t x_desc;
    ::cudnnTensorDescriptor_t y_desc;
    ::cudnnFilterDescriptor_t w_desc;
    ::cudnnConvolutionDescriptor_t conv_desc;

#if CUDNN_MAJOR < 8
    ::cudnnConvolutionFwdAlgo_t fwd_algo;
#else 
	::cudnnConvolutionFwdAlgoPerf_t fwd_algo;
	int fwd_algo_size;
	int num_of_algs;
#endif

	// Workspace Parameters
	size_t fwd_ws_size;
	float *fwd_ws_data;

	// Convolution Parameters
    int pad_w = 0;
    int pad_h = 0;
    int str_w = 1;
    int str_h = 1;
    int dil_w = 1;
    int dil_h = 1;

    // device variable
	int selected_dev;

    // workspace variables
    int y_n, y_c, y_h, y_w;

	// Allocate workspace
	void allocateWorkspace(size_t fwd_ws_req_size);
};

inline void myCudnnConvolution::getOutputSizes( int *out_n, int *out_c, int *out_h, int *out_w ) {
	*out_n = y_n;
	*out_c = y_c;
	*out_h = y_h;
	*out_w = y_w;
};

inline int myCudnnConvolution::getDevice( ) {
	return selected_dev;
};

#endif /* MYCUDNNCONVOLUTION_H_ */
