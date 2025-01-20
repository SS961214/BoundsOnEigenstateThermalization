#include "tests.hpp"
#include "ManyBodySpaceBase_test.cuh"
#include "ObjectOnGPU.cuh"
#include "ManyBodyBosonSpace.hpp"
#include <iostream>

TEST_CASE("ManyBodyBosonSpace_onGPU", "test") {
	size_t pValue;
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;
	pValue *= 16;
	cuCHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, pValue));
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;

	constexpr int NMin = 4;
	constexpr int NMax = 11;
	auto const sysSize = [] (int N) { return N; }; // unit filling
	{
		// Default constructor
		ObjectOnGPU<ManyBodyBosonSpace> dObj;
	}
	for(auto N = NMin; N <= NMax; ++N) {
		int L = sysSize(N);
		std::cout << std::endl;
		ManyBodyBosonSpace              hObj(L, N);
		ObjectOnGPU<ManyBodyBosonSpace> dObj(L, N);
		std::cout << "# L = " << L << ", N = " << N << ", ptr = " << dObj.ptr() << std::endl;
		test_ManyBodySpaceBase(dObj, hObj);
	}
}