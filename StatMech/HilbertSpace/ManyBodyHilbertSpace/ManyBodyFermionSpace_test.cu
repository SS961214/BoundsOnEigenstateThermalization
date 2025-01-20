#include "tests.hpp"
#include "ManyBodySpaceBase_test.cuh"
#include "ObjectOnGPU.cuh"
#include "ManyBodyFermionSpace.hpp"
#include <iostream>

TEST_CASE("ManyBodyFermionSpace_onGPU", "test") {
	size_t pValue;
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;
	pValue *= 16;
	cuCHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, pValue));
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;

	constexpr int NMin = 4;
	constexpr int NMax = 11;
	auto const sysSize = [] (int N) { return 2*N; }; // half filling
	{
		// Default constructor
		ObjectOnGPU<ManyBodyFermionSpace> dObj;
	}
	for(auto N = NMin; N <= NMax; ++N) {
		int const L = sysSize(N);
		std::cout << std::endl;
		ManyBodyFermionSpace              hObj(L, N);
		ObjectOnGPU<ManyBodyFermionSpace> dObj(L, N);
		std::cout << "# L = " << L << ", N = " << N << ", ptr = " << dObj.ptr() << std::endl;
		test_ManyBodySpaceBase(dObj, hObj);
	}
}