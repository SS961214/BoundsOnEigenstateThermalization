#pragma once

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