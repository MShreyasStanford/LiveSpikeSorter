
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include "myCudnnConvolution.h"

#include "../Helpers/Utils.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define _CHECK_CUDNN_RESULT(label, f) { \
  ::cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout << label << ":: " << #f ": " << err << std::endl; \
  } \
}

void printAlgoName(cudnnConvolutionFwdAlgoPerf_t fwd_algo) {
	switch (fwd_algo.algo) {
	case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
		std::cout << "  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM" << std::endl;
		break;
	case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
		std::cout << "  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM" << std::endl;
		break;
	case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
		std::cout << "  CUDNN_CONVOLUTION_FWD_ALGO_GEMM" << std::endl;
		break;
	case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
		std::cout << "  CUDNN_CONVOLUTION_FWD_ALGO_DIRECT" << std::endl;
		break;
	case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
		std::cout << "  CUDNN_CONVOLUTION_FWD_ALGO_FFT" << std::endl;
		break;
	case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
		std::cout << "  CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING" << std::endl;
		break;
	case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
		std::cout << "  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD" << std::endl;
		break;
	case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
		std::cout << "  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED" << std::endl;
		break;
	default:
		std::cout << "  Unknown algorithm" << std::endl;
	}
}
// ------------------------------------------------------------------------------
//
// Name			: myCudnnConvolution::myCudnnConvolution()
//
// Description  : Constructor, initialize descriptors and CUDA Environment
//
// ------------------------------------------------------------------------------
myCudnnConvolution::myCudnnConvolution( int iValue ) :
			  y_n (0)
			, y_c (0)
			, y_h (0)
			, y_w (0)
			, selected_dev( iValue )
			, num_of_algs(1)
			, fwd_algo_size(1)
{
	static const char *ptLabel = { "myCudnnConvolution::myCudnnConvolution" };

	// Initialize CUDA Environment
	if (cudaSetDevice(selected_dev) != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	// Create CuDNN Handle
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnCreate(&cudnn) );

	// Create Descriptors
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnCreateTensorDescriptor(&x_desc) ); // Input
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnCreateTensorDescriptor(&y_desc) ); // Output
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnCreateFilterDescriptor(&w_desc) ); // Filter
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnCreateConvolutionDescriptor(&conv_desc) ); // Conv

	fwd_ws_size = 0;
	fwd_ws_data = NULL;

	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &num_of_algs));



};


// ------------------------------------------------------------------------------
//
// Name			: myCudnnConvolution::~myCudnnConvolution()
//
// Description  : Destructor, destroy descriptors and reset device
//
// ------------------------------------------------------------------------------
myCudnnConvolution::~myCudnnConvolution()
{
	static const char *ptLabel = { "myCudnnConvolution::~myCudnnConvolution" };

	// finalizing
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnDestroyTensorDescriptor(y_desc));
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnDestroyConvolutionDescriptor(conv_desc));
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnDestroyFilterDescriptor(w_desc));
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnDestroyTensorDescriptor(x_desc));
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnDestroy(cudnn));

	cudaFree(fwd_ws_data);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//if (cudaDeviceReset() != cudaSuccess) // Comment this out to avoid issues with mutliple instances
		//fprintf(stderr, "cudaDeviceReset failed!");
};


// ------------------------------------------------------------------------------
//
// Name			: myCudnnConvolution::selectDevice
//
// Description  : Select CUDA device
//
// ------------------------------------------------------------------------------
bool myCudnnConvolution::selectDevice( int iValue )
{
	static const char *ptLabel = { "myCudnnConvolution::selectDevice" };

	// Reset device before selecting new one
	if (cudaDeviceReset() != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return false;
	}

	// Initialize CUDA Environment
	if (cudaSetDevice(iValue) != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return false;
	}

	// Overwrite CuDNN Handle
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnCreate(&cudnn) );

	selected_dev = iValue;

	return true;
};



// ------------------------------------------------------------------------------
//
// Name			: myCudnnConvolution::allocateWorkspace()
//
// Description  : Allocate workspace for intermediate processing
//
// ------------------------------------------------------------------------------
void myCudnnConvolution::allocateWorkspace(size_t fwd_ws_req_size)
{
	static const char *ptLabel = { "myCudnnConvolution::allocateWorkspace" };

	if ( fwd_ws_data ) {
		std::cout << "Warning workspace already allocated:" << std::endl;
		std::cout << "Current Size: " << fwd_ws_size << " / Requested Size: " << fwd_ws_req_size << std::endl;
		std::cout << "...trying to reallocate" << std::endl;

	    if ( cudaFree(fwd_ws_data) != cudaSuccess )
	        throw std::runtime_error("failed to free workspace memory");
	}

	if ( cudaMalloc((void**)&fwd_ws_data, fwd_ws_req_size * sizeof(float)) != cudaSuccess ) {
		fwd_ws_size = 0;
		fwd_ws_data = NULL;
		throw std::runtime_error("failed to allocate device memory");
	} else
		fwd_ws_size = fwd_ws_req_size;

}
// ------------------------------------------------------------------------------
//
// Name			: myCudnnConvolution::setSizes
//
// Description  : Set sizes of Tensor descriptors
//
// ------------------------------------------------------------------------------
void myCudnnConvolution::setSizes(long lN, long lM, long lC, long lT)
{
	static const char *ptLabel = { "myCudnnConvolution::setSizes" };

	/*const int x_w = lN;
	const int x_h = lC;
	const int x_c = 1;
	const int x_n = 1;

	const int w_w = lM;
	const int w_h = lC;
	const int w_c = 1;
	const int w_k = lT;*/

	//TODO: Verify if dimensionality changes enable faster convolutions...

	//x is the input
	const int x_w = lC;			//width
	const int x_h = lN;			//height
	const int x_c = 1;			//#channels (note: not the channels that we are used to)
	const int x_n = 1;			//batch size 

	//w is the filter that is going to be used
	const int w_w = lC;			//width
	const int w_h = lM;			//height 
	const int w_c = 1;			//#channels (note: not the channels that we are used to)
	const int w_k = lT;			//#output channels (note: not the channels that we are used to)

	// Pad symmetrically by filter width
	//pad_h = 0;
    pad_h = lM - 1; // original
	//pad_h = (lM - 1) / 2; // LAST CORRECT

	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnSetTensor4dDescriptor(
			x_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, x_n, x_c, x_h, x_w));

	// filter
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnSetFilter4dDescriptor(
							/*Output descriptor filter in this case*/w_desc,
							/*Data Type*/							 CUDNN_DATA_FLOAT, 
							/*Format*/								 CUDNN_TENSOR_NCHW,
							/*out_channels*/						 w_k,
							/*batch channels*/						 w_c, 
							/*filter height*/						 w_h,
							/*filter width*/						 w_w));

	// convolution
#if CUDNN_MAJOR >= 6
	// LAST CORRECT: CUDNN_CONVOLUTION
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnSetConvolution2dDescriptor(
										/*Descriptor changed*/conv_desc,
										/*Padding height*/    pad_h,
										/*Padding  width*/    pad_w,
										/*Vertical stride*/   str_h,
										/*Horizontal stride*/ str_w,
										/*Dilation Height*/   dil_h,
										/*Dilation Width*/    dil_w,
										/*Mode*/              CUDNN_CROSS_CORRELATION, 
										/*Compute Type*/      CUDNN_DATA_FLOAT));
#else
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnSetConvolution2dDescriptor(
			conv_desc,
			pad_h, pad_w, str_h, str_w, dil_h, dil_w,
			CUDNN_CONVOLUTION));
#endif  // CUDNN_MAJOR

	// output
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnGetConvolution2dForwardOutputDim(
			conv_desc, x_desc, w_desc, &y_n, &y_c, &y_h, &y_w));

	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnSetTensor4dDescriptor(
			y_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, y_n, y_c, y_h, y_w));


#if CUDNN_MAJOR < 8
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnGetConvolutionForwardAlgorithm(
			cudnn,
			x_desc, w_desc, conv_desc, y_desc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo));

	// workspaces
	size_t fwd_ws_req_size;
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolutionForwardWorkspaceSize(
		cudnn, x_desc, w_desc, conv_desc, y_desc, fwd_algo, &fwd_ws_req_size));

	// secure sufficient work space memory
	if (fwd_ws_req_size > fwd_ws_size)
		allocateWorkspace(fwd_ws_req_size);

	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnFindConvolutionForwardAlgorithm(
		cudnn,
		x_desc, w_desc, conv_desc, y_desc,
		fwd_algo_size, &fwd_algo_size, &fwd_algo));



#else
	
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
		x_desc, w_desc, conv_desc, y_desc,
		fwd_algo_size, &fwd_algo_size, &fwd_algo));

	
	//printAlgoName(fwd_algo);

	/*_CHECK_CUDNN_RESULT(ptLabel, ::cudnnFindConvolutionForwardAlgorithm(cudnn,
		x_desc, w_desc, conv_desc, y_desc,
		fwd_algo_size, &fwd_algo_size, &fwd_algo));*/


	// workspaces
	size_t fwd_ws_req_size;
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolutionForwardWorkspaceSize(
		cudnn, x_desc, w_desc, conv_desc, y_desc, fwd_algo.algo, &fwd_ws_req_size));

	// secure sufficient work space memory
	if (fwd_ws_req_size > fwd_ws_size)
		allocateWorkspace(fwd_ws_req_size);

#endif


};


// ------------------------------------------------------------------------------
//
// Name			: myCudnnConvolution::fwdConvolve
//
// Description  : perform actual convolution
//
// ------------------------------------------------------------------------------
void myCudnnConvolution::fwdConvolve(float *fX, float *fW, float *fY, long lN, long lM, long lC, long lT)
{
	static const char *ptLabel = { "myCudnnConvolution::fwdConvolve" };

	// Define descriptors
	setSizesPCsV2(lN, lM, lC, lT);

	// perform forward operation
	float fwd_alpha = 1.f;
	float fwd_beta  = 0.0f;

#if CUDNN_MAJOR >= 8
	_CHECK_CUDNN_RESULT( ptLabel, ::cudnnConvolutionForward(
			cudnn,
			&fwd_alpha,
			x_desc, fX, // set in setSizes()
			w_desc, fW, // set in setSizes()
			conv_desc, fwd_algo.algo, // sets padding, dilation, etc, algo used is CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
			fwd_ws_data, fwd_ws_size, // fwd_ws_data seems to be just a buffer for it to do intermediate computations with?
			&fwd_beta,
			y_desc, fY));

#else
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnConvolutionForward(
		cudnn,
		&fwd_alpha,
		x_desc, fX,
		w_desc, fW,
		conv_desc, &fwd_algo,
		fwd_ws_data, fwd_ws_size,
		&fwd_beta,
		y_desc, fY));
#endif

	// Move pointer for assymmetric padding
	//fY += y_n*y_c*y_h * pad_w;
	//std::cout << " dims: yn " << y_n << " x yc "<< y_c << " x yh " << y_h << " x yw " << y_w << std::endl;
	//std::cout << " dims: yn " << y_n << " x yc "<< y_c << " x yh " << y_h << " x yw " << y_w << std::endl;
	//for ( int i = y_n*y_c*y_h*pad_w ; i < y_n*y_c*y_h*y_w ; i++ )
		//fY[ i - y_n*y_c*y_h*pad_w ] = fY[ i ];

};

void myCudnnConvolution::fwdConvolvePCS(float* d_batch, float* d_wPCA, float* d_B,
	int K, int M, int C, int currBatchNumSamples)
{
	static const char *ptLabel = { "myCudnnConvolution::fwdConvolvePCS" };

	// Set the sizes:
	setSizesPCs(K, C, M, currBatchNumSamples);

	// Perform the convolution
	float alpha = 1.0f;
	float beta = 0.0f;

	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnConvolutionForward(
		cudnn,
		&alpha,
		x_desc, d_batch,        // Input tensor
		w_desc, d_wPCA,          // Filter tensor
		conv_desc, fwd_algo.algo, // Convolution descriptor and selected algorithm
		fwd_ws_data, fwd_ws_size, // Workspace
		&beta,
		y_desc, d_B));           // Output tensor
	std::cout << " dims: yn " << y_n << " x yc "<< y_c << " x yh " << y_h << " x yw " << y_w << std::endl;
}

void myCudnnConvolution::setSizesPCs(long numPcs, long C, long M, long currBatchNumSamples)
{
	static const char *ptLabel = { "myCudnnConvolution::setSizesPCS" };

	/*const int x_w = lN;
	const int x_h = lC;
	const int x_c = 1;
	const int x_n = 1;

	const int w_w = lM;
	const int w_h = lC;
	const int w_c = 1;
	const int w_k = lT;*/

	//TODO: Verify if dimensionality changes enable faster convolutions...

	//x is the input
	// Last "WORKING"
	
	const int x_w = C;			//width
	const int x_h = currBatchNumSamples;			//height
	const int x_c = 1;			//#channels (note: not the channels that we are used to)
	const int x_n = 1;			//batch size 

	//w is the filter that is going to be used
	const int w_w = 1;			//width
	const int w_h = M;			//height 
	const int w_c = 1;			//#channels (note: not the channels that we are used to)
	const int w_k = numPcs;			//#output channels (note: not the channels that we are used to)


	// Pad symmetrically by filter width
	//pad_h = 0;
	//pad_h = M - 1; // original
	pad_h = (M - 1) / 2; // LAST CORRECT

	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnSetTensor4dDescriptor(
		x_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, x_n, x_c, x_h, x_w));

	// filter
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnSetFilter4dDescriptor(
		/*Output descriptor filter in this case*/w_desc,
		/*Data Type*/							 CUDNN_DATA_FLOAT,
		/*Format*/								 CUDNN_TENSOR_NCHW,
		/*out_channels*/						 w_k,
		/*batch channels*/						 w_c,
		/*filter height*/						 w_h,
		/*filter width*/						 w_w));

	// convolution
#if CUDNN_MAJOR >= 6
	// LAST CORRECT: CUDNN_CONVOLUTION
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnSetConvolution2dDescriptor(
		/*Descriptor changed*/conv_desc,
		/*Padding height*/    pad_h,
		/*Padding  width*/    pad_w,
		/*Vertical stride*/   str_h,
		/*Horizontal stride*/ str_w,
		/*Dilation Height*/   dil_h,
		/*Dilation Width*/    dil_w,
		/*Mode*/              CUDNN_CROSS_CORRELATION,
		/*Compute Type*/      CUDNN_DATA_FLOAT));
#else
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnSetConvolution2dDescriptor(
		conv_desc,
		pad_h, pad_w, str_h, str_w, dil_h, dil_w,
		CUDNN_CONVOLUTION));
#endif  // CUDNN_MAJOR

	// output
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolution2dForwardOutputDim(
		conv_desc, x_desc, w_desc, &y_n, &y_c, &y_h, &y_w));

	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnSetTensor4dDescriptor(
		y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, y_n, y_c, y_h, y_w));


#if CUDNN_MAJOR < 8
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolutionForwardAlgorithm(
		cudnn,
		x_desc, w_desc, conv_desc, y_desc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo));

	// workspaces
	size_t fwd_ws_req_size;
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolutionForwardWorkspaceSize(
		cudnn, x_desc, w_desc, conv_desc, y_desc, fwd_algo, &fwd_ws_req_size));

	// secure sufficient work space memory
	if (fwd_ws_req_size > fwd_ws_size)
		allocateWorkspace(fwd_ws_req_size);

	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnFindConvolutionForwardAlgorithm(
		cudnn,
		x_desc, w_desc, conv_desc, y_desc,
		fwd_algo_size, &fwd_algo_size, &fwd_algo));



#else

	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
		x_desc, w_desc, conv_desc, y_desc,
		fwd_algo_size, &fwd_algo_size, &fwd_algo));


	//printAlgoName(fwd_algo);

	/*_CHECK_CUDNN_RESULT(ptLabel, ::cudnnFindConvolutionForwardAlgorithm(cudnn,
		x_desc, w_desc, conv_desc, y_desc,
		fwd_algo_size, &fwd_algo_size, &fwd_algo));*/


		// workspaces
	size_t fwd_ws_req_size;
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolutionForwardWorkspaceSize(
		cudnn, x_desc, w_desc, conv_desc, y_desc, fwd_algo.algo, &fwd_ws_req_size));

	// secure sufficient work space memory
	if (fwd_ws_req_size > fwd_ws_size)
		allocateWorkspace(fwd_ws_req_size);

#endif
}

void myCudnnConvolution::setSizesPCsV2(long numPcs, long C, long M, long currBatchNumSamples)
{
	static const char* ptLabel = { "myCudnnConvolution::setSizesPCsV2" };

	int inputDims[3] = { static_cast<int>(C), 1, static_cast<int>(currBatchNumSamples) }; // N, C, W
	int inputStrides[3] = { inputDims[1] * inputDims[2], inputDims[2], 1 }; // Strides for N, C, W
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnSetTensorNdDescriptor(
		x_desc,
		CUDNN_DATA_FLOAT, // Data type
		3,                // Number of dimensions
		inputDims,
		inputStrides
	));

	int filterDims[3] = { static_cast<int>(numPcs), 1, static_cast<int>(M) }; // K, C, W

	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnSetFilterNdDescriptor(
		w_desc,
		CUDNN_DATA_FLOAT,  // Data type
		CUDNN_TENSOR_NCHW, // Tensor format
		3,                 // Number of dimensions
		filterDims
	));

	int pad[1] = { static_cast<int>(M) / 2 }; // Full padding
	int stride[1] = { 1 };
	int dilation[1] = { 1 };

	cudnnSetConvolutionNdDescriptor(
		conv_desc,
		1,                   // Number of spatial dimensions
		pad,                 // Padding
		stride,              // Stride
		dilation,            // Dilation
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT     // Data type
	);
	
	int outputDims[3]; // N, C, W_out
	int outputStrides[3] = { outputDims[1] * outputDims[2], outputDims[2], 1 }; // Strides for N, C, W

	cudnnSetTensorNdDescriptor(
		y_desc,
		CUDNN_DATA_FLOAT, // Data type
		3,                // Number of dimensions
		outputDims,
		outputStrides
	);


	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
		x_desc, w_desc, conv_desc, y_desc,
		fwd_algo_size, &fwd_algo_size, &fwd_algo));


	//printAlgoName(fwd_algo);

	/*_CHECK_CUDNN_RESULT(ptLabel, ::cudnnFindConvolutionForwardAlgorithm(cudnn,
		x_desc, w_desc, conv_desc, y_desc,
		fwd_algo_size, &fwd_algo_size, &fwd_algo));*/


		// workspaces
	size_t fwd_ws_req_size;
	_CHECK_CUDNN_RESULT(ptLabel, ::cudnnGetConvolutionForwardWorkspaceSize(
		cudnn, x_desc, w_desc, conv_desc, y_desc, fwd_algo.algo, &fwd_ws_req_size));

	// secure sufficient work space memory
	if (fwd_ws_req_size > fwd_ws_size)
		allocateWorkspace(fwd_ws_req_size);
}