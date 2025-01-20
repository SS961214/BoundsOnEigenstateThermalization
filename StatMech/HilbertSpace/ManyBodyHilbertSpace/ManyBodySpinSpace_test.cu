#include "tests.hpp"
#include "ManyBodySpaceBase_test.cuh"
#include "ObjectOnGPU.cuh"
#include "ManyBodySpinSpace.hpp"
#include <iostream>

TEST_CASE("ManyBodySpinSpace_onGPU", "test") {
	size_t pValue;
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;
	pValue *= 8;
	cuCHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, pValue));
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;

	constexpr int dimLoc = 2;
	constexpr int LMax   = 20;
	{
		// Default constructor
		ObjectOnGPU<ManyBodySpinSpace> dObj;
	}
	for(auto L = 1;L <= LMax; ++L) {
		std::cout << std::endl;
		ManyBodySpinSpace hObj(L, dimLoc);
		ObjectOnGPU<ManyBodySpinSpace> dObj(L, dimLoc);
		std::cout << "# L = " << L << ", dimLoc = " << dimLoc << ", ptr = " << dObj.ptr() << std::endl;
		test_ManyBodySpaceBase(dObj, hObj);
	}
}