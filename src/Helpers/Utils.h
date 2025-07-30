/*
 * Utils.h
 *
 *  Created on: Feb 12, 2019
 *      Author: basti
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <string>

// Define Platform
#if defined(_WIN32) || defined(_WIN64)
#define WINDOWS
#else
#define LINUX
#endif

// Define Debugging Output Macros
#ifdef _DEBUG
#define _DEBUG_PUT_0(label,msg)			std::cout << label << ":: " << msg << std::endl

#else
#define _DEBUG_PUT_0(label,msg)			std::cout << label << ":: " << msg << std::endl

#endif

#ifdef _DEBUG
#define _DEBUG_CALL(x) x
#define _DEBUG_CALL_2(x)
#else
#define _DEBUG_CALL(x)
#define _DEBUG_CALL_2(x) x
#endif



// Define Runtime Error

#define _RUN_ERROR(label, msg) {\
std::cout << (std::string) label + std::string(":: ") + msg + \
							std::string(" on line ") + std::to_string(__LINE__) + std::string(" in file ") + std::string(__FILE__) << std::endl; \
throw std::runtime_error( (std::string) label + std::string(":: ") + msg + \
							std::string(" on line ") + std::to_string(__LINE__) + std::string(" in file ") + std::string(__FILE__)  );\
}

#define _UNUSED(x) (void)x;

// Helper Functions
//running into cuda error "invalid argument"
#define _CUDA_CALL(f) { ::cudaError_t err = (f); \
  if (err != cudaSuccess){ \
	std::cout << cudaGetErrorString(err) << std::endl; \
	_RUN_ERROR( ptLabel, #f ); \
	} \
  }

#define _NPP_CALL(f, errMsg) { ::NppStatus err = (f); \
  if (err != NPP_SUCCESS){ \
	std::cout << errMsg << std::endl; \
	_RUN_ERROR( ptLabel, #f ); \
	} \
  }

#endif /* UTILS_H_ */

