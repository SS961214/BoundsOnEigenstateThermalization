#pragma once

#if __has_include(<mkl.h>)
	#ifndef MKL
		#define MKL
	#endif
	#if !defined(MKL_ILP64) && !defined(EIGEN_USE_MKL_ALL)
		#define EIGEN_USE_MKL_ALL
	#endif
	#ifdef MKL_ILP64
		#define EIGEN_DEFAULT_DENSE_INDEX_TYPE long long
	#endif
#else
	#if __has_include(<Accelerate/Accelerate.h>)
		#ifndef ACCELERATE
			#define ACCELERATE
		#endif
		#ifndef EIGEN_USE_BLAS
			#define EIGEN_USE_BLAS
		#endif
	#endif
#endif

#ifndef DEBUG
	#if defined(NDEBUG) || defined(__CUDA_ARCH__)
		#define DEBUG(arg)
	#else
		#define DEBUG(arg) (arg)
	#endif
#endif

#ifndef cuCHECK
	#ifndef __CUDA_ARCH__
		#define cuCHECK(call)                                                                   \
			{                                                                                   \
				const cudaError_t error = call;                                                 \
				if(error != cudaSuccess) {                                                      \
					fprintf(stderr, "cuCHECK Error: %s:%d,  ", __FILE__, __LINE__);             \
					fprintf(stderr, "code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
					assert(error == cudaSuccess);                                               \
					std::exit(EXIT_FAILURE);                                                    \
				}                                                                               \
			};
	#else
		#define cuCHECK(call)                                                          \
			{                                                                          \
				const cudaError_t error = call;                                        \
				if(error != cudaSuccess) {                                             \
					printf("cuCHECK Error: %s:%d,  ", __FILE__, __LINE__);             \
					printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
					assert(error == cudaSuccess);                                      \
				}                                                                      \
			};
	#endif
#endif

#ifndef CUSOLVER_CHECK
	#ifndef __CUDA_ARCH__
		#define CUSOLVER_CHECK(err)                                                            \
			do {                                                                               \
				cusolverStatus_t err_ = (err);                                                 \
				if(err_ != CUSOLVER_STATUS_SUCCESS) {                                          \
					fprintf(stderr, "cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
					throw std::runtime_error("cusolver error");                                \
				}                                                                              \
			} while(0);
	#else
		#define CUSOLVER_CHECK(err)                                                   \
			do {                                                                      \
				cusolverStatus_t err_ = (err);                                        \
				if(err_ != CUSOLVER_STATUS_SUCCESS) {                                 \
					printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
					assert(error == cudaSuccess);                                     \
				}                                                                     \
			} while(0);
	#endif
#endif
