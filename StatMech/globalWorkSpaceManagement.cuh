#pragma once
#include <HilbertSpace>
#include <iostream>

namespace StatMech {  // Work stograge management for the GPU
	__device__ static inline int get_smid() {
		int smid;
		asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
		return smid;
	}

	constexpr int                slots_per_sm = 4;
	constexpr unsigned long long busy         = 0xFFFFFFFFULL;

	__device__ static inline int get_slot(unsigned long long* sm_slots) {
		// printf("# (block=%d/%d):\t %s\n", blockIdx.x, gridDim.x, __func__);
		unsigned long long my_slots;
		bool               done = false;
		int                my_slot;
		while(!done) {
			// wait until we get an available slot
			while((my_slots = atomicExch(sm_slots, busy)) == busy) {};
			my_slot = __ffsll(~my_slots) - 1;
			if(my_slot < slots_per_sm)
				done = true;
			else {
				// handle case where all slots busy, should not happen
				atomicExch(sm_slots, my_slots);
			}
		}
		unsigned long long my_slot_bit = static_cast<unsigned long long>(1) << my_slot;
		unsigned long long retval      = my_slots | my_slot_bit;
		if(atomicExch(sm_slots, retval) != busy) {
			printf("# Error: Lock is failed in %s\n", __func__);
			assert([]() { return false; });
		}
		// assert(atomicExch(sm_slots, retval) == busy);
		DEBUG(printf("#\t (block=%d/%d)\t Got a slot at (slot=%d, sm=%d)\n", blockIdx.x, gridDim.x,
		             my_slot, get_smid()));
		return my_slot;
	}

	__device__ static inline void release_slot(unsigned long long* sm_slots, int slot) {
		// printf("# (block=%d/%d):\t %s\n", blockIdx.x, gridDim.x, __func__);
		unsigned long long my_slots;
		// wait until slot access not busy
		while((my_slots = atomicExch(sm_slots, busy)) == busy) {};
		unsigned long long my_slot_bit = static_cast<unsigned long long>(1) << slot;
		unsigned long long retval      = my_slots ^ my_slot_bit;
		if(atomicExch(sm_slots, retval) != busy) {
			printf("# Error: Lock is failed in %s\n", __func__);
			assert([]() { return false; });
		}
		// assert(atomicExch(sm_slots, retval) == busy);
		DEBUG(printf("#\t (block=%d/%d)\t Released a slot at (slot=%d, sm=%d)\n", blockIdx.x,
		             gridDim.x, slot, get_smid()));
	}

	template<class Func>
	inline void configureKernel(int& nThreads, int& shMem, int const shMemPerBlock,
	                            int const shMemPerThread, Func const func) {
		int dev;
		cuCHECK(cudaGetDevice(&dev));

		cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, func));
		nThreads = attr.maxThreadsPerBlock;
		if(shMemPerThread > 0) {
			nThreads = std::min(
			    nThreads, int(attr.maxDynamicSharedSizeBytes - shMemPerBlock) / shMemPerThread);
		}
		shMem = shMemPerBlock + nThreads * shMemPerThread;

		// #pragma omp critical
		std::cout << "# (Device " << dev << ")\t nThreads = " << nThreads << ", shMem = " << shMem
		          << ", attr.maxThreadsPerBlock = " << attr.maxThreadsPerBlock
		          << ", attr.maxDynamicSharedSizeBytes = " << attr.maxDynamicSharedSizeBytes
		          << ", shMemPerBlock = " << shMemPerBlock
		          << ", shMemPerThread = " << shMemPerThread << std::endl;

		if(nThreads < 0) {
			std::cerr << "Error: " << __FILE__ << ":" << __LINE__
			          << "\tShared memory on the device is insufficient:"
			          << " shMemPerBlock = " << shMemPerBlock
			          << ", shMemPerThread = " << shMemPerThread
			          << ", attr.maxDynamicSharedSizeBytes = " << attr.maxDynamicSharedSizeBytes
			          << std::endl;
			std::exit(EXIT_FAILURE);
		}
		assert(shMem <= attr.maxDynamicSharedSizeBytes);
		assert(nThreads >= 1);
	}
}  // namespace StatMech