#pragma once
#include <Eigen/Core>

using Eigen::Index;

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

#ifndef CUSTOM_OMP_FUNCTIONS
	#define CUSTOM_OMP_FUNCTIONS
	#if __has_include(<omp.h>)
		#include <omp.h>
__host__ __device__ static inline int get_max_threads() {
		#ifdef __CUDA_ARCH__
	return 1;
		#else
	return omp_get_max_threads();
		#endif
}
__host__ __device__ static inline int get_thread_num() {
		#ifdef __CUDA_ARCH__
	return 0;
		#else
	return omp_get_thread_num();
		#endif
}
	#else
constexpr static inline int get_max_threads() { return 1; }
constexpr static inline int get_thread_num() { return 0; }
	#endif
#endif

#define cuCHECK(call)                                                          \
	{                                                                          \
		const cudaError_t error = call;                                        \
		if(error != cudaSuccess) {                                             \
			printf("cuCHECK Error: %s:%d,  ", __FILE__, __LINE__);             \
			printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
			assert(error == cudaSuccess);                                      \
		}                                                                      \
	};

#ifdef __CUDA_ARCH__
	#define cuASSERT(condition, message)                                               \
		if(!(condition)) {                                                             \
			printf("Assertion failed: %s. File:%s:%d\n", message, __FILE__, __LINE__); \
			asm("trap;");                                                              \
		}
#else
	#define cuASSERT(condition, message)                                                 \
		if(!(condition)) {                                                               \
			std::cerr << "Assertion failed: " << message << ". File:" << __FILE__ << ":" \
			          << __LINE__ << std::endl;                                          \
			std::exit(EXIT_FAILURE);                                                     \
		}
#endif

#ifndef DEBUG
	#if defined(NDEBUG)
		#define DEBUG(arg)
	#else
		#define DEBUG(arg) (arg)
	#endif
#endif

#ifndef DEVICE
	#ifdef __CUDA_ARCH__
		#define DEVICE "GPU"
	#else
		#define DEVICE "CPU"
	#endif
#endif