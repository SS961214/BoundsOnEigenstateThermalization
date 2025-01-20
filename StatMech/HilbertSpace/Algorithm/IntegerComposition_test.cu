#include "tests.hpp"
#include "IntegerComposition.hpp"
#include "ObjectOnGPU.cuh"
#include <Eigen/Dense>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void check_bijectiveness_kernel(IntegerComposition const* iComp_ptr) {
	int                       ordinal = blockDim.x * blockIdx.x + threadIdx.x;
	IntegerComposition const& iComp   = *iComp_ptr;
	if(ordinal >= iComp.dim()) return;
	if(ordinal == 0) printf("# %s\n", __func__);

	extern __shared__ int      data[];
	Eigen::Map<Eigen::ArrayXi> config(data + iComp.length() * threadIdx.x, iComp.length());
	iComp.ordinal_to_config(config, ordinal);

	assert(iComp.value() == static_cast<Index>(config.sum()));
	assert(iComp.max() >= static_cast<Index>(config.maxCoeff()));
	assert(iComp.config_to_ordinal(config) == ordinal);
	for(auto pos = 0; pos < iComp.length(); ++pos) {
		assert(iComp.locNumber(ordinal, pos) == config(pos));
	}
}

__global__ void check_translate_kernel(IntegerComposition const* iComp_ptr, Index const transEqDim,
                                       int* transEqClassRep, int* appeared) {
	Index                     eqClass = blockDim.x * blockIdx.x + threadIdx.x;
	IntegerComposition const& iComp   = *iComp_ptr;
	if(eqClass >= transEqDim) return;
	if(eqClass == 0) printf("# %s\n# transEqDim = %d\n", __func__, int(transEqDim));
	// printf("# iComp.length() = %d\n", int(iComp.length()));

	extern __shared__ int      data[];
	Eigen::Map<Eigen::ArrayXi> config(data + iComp.length() * threadIdx.x, iComp.length());

	auto const rep = transEqClassRep[eqClass];
	atomicAdd(&appeared[rep], 1);
	for(auto trans = 1; trans != iComp.length(); ++trans) {
		auto const translated = iComp.translate(rep, trans, config);
		if(translated == rep) break;
		atomicAdd(&appeared[translated], 1);
	}
}

__global__ void check_reverse_kernel(IntegerComposition const* iComp_ptr) {
	int                       ordinal = blockDim.x * blockIdx.x + threadIdx.x;
	IntegerComposition const& iComp   = *iComp_ptr;
	if(ordinal >= iComp.dim()) return;
	if(ordinal == 0) printf("# %s\n", __func__);

	extern __shared__ int      data[];
	Eigen::Map<Eigen::ArrayXi> config(data + iComp.length() * threadIdx.x, iComp.length());

	iComp.ordinal_to_config(config, ordinal);
	auto const reversed = iComp.config_to_ordinal(config.reverse());
	assert(iComp.reverse(ordinal, config) == reversed);
}

void test_IntegerComposition(Index N, Index Length, Index Max) {
	std::cout << "# " << __PRETTY_FUNCTION__ << std::endl;
	std::cout << "##\t (N, Length, Max) = (" << N << ", " << Length << ", " << Max << ")"
	          << std::endl;
	IntegerComposition iComp(N, Length, Max);
	std::cout << "##\t iComp.dim() = " << iComp.dim() << std::endl;
	ObjectOnGPU<IntegerComposition> diComp(N, Length, Max);
	{
		struct cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, check_bijectiveness_kernel));
		int nThreads = min(attr.maxThreadsPerBlock, int(iComp.dim()));
		if(iComp.length() > 0)
			nThreads = min(nThreads,
			               int(attr.maxDynamicSharedSizeBytes / (iComp.length() * sizeof(int))));
		if(nThreads == 0) nThreads = 1;
		int nGrids = (iComp.dim() / nThreads) + (iComp.dim() % nThreads == 0 ? 0 : 1);
		if(nGrids == 0) nGrids = 1;
		auto shMem = nThreads * (iComp.length() * sizeof(int));

		std::cout << "# nGrids = " << nGrids << ", nThreads = " << nThreads << ", shMem = " << shMem
		          << std::endl;
		check_bijectiveness_kernel<<<nGrids, nThreads, shMem>>>(diComp.ptr());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
	}
	std::cout << "# Passed check_bijectiveness_kernel.\n#" << std::endl;

	thrust::host_vector<int> state_to_rep(iComp.dim());
#pragma omp parallel for
	for(Index j = 0; j < iComp.dim(); ++j) {
		state_to_rep[j] = j;
		for(auto trans = 1; trans != iComp.length(); ++trans) {
			int const translated = iComp.translate(j, trans);
			if(j == translated) break;
			state_to_rep[j] = std::min(state_to_rep[j], translated);
		}
	}
	std::sort(state_to_rep.begin(), state_to_rep.end());
	Index transEqDim = std::unique(state_to_rep.begin(), state_to_rep.end()) - state_to_rep.begin();
	state_to_rep.resize(transEqDim);
	std::cout << "##\t transEqDim = " << transEqDim << std::endl;

	thrust::device_vector<int> dtransEqClassRep(state_to_rep);
	{
		struct cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, check_translate_kernel));
		int nThreads = min(attr.maxThreadsPerBlock, int(transEqDim));
		if(iComp.length() > 0)
			nThreads = min(nThreads,
			               int(attr.maxDynamicSharedSizeBytes / (iComp.length() * sizeof(int))));
		if(nThreads == 0) nThreads = 1;
		int nGrids = (transEqDim / nThreads) + (transEqDim % nThreads == 0 ? 0 : 1);
		if(nGrids == 0) nGrids = 1;
		auto shMem = nThreads * (iComp.length() * sizeof(int));

		std::cout << "# nGrids = " << nGrids << ", nThreads = " << nThreads << ", shMem = " << shMem
		          << std::endl;
		thrust::device_vector<int> dappeared(iComp.dim(), 0);
		check_translate_kernel<<<nGrids, nThreads, shMem>>>(
		    diComp.ptr(), transEqDim, dtransEqClassRep.data().get(), dappeared.data().get());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());

		thrust::host_vector<int> appeared(dappeared);
#pragma omp parallel for
		for(auto j = 0; j < iComp.dim(); ++j) { REQUIRE(appeared[j] == 1); }
	}
	std::cout << "# Passed check_translate_kernel.\n#" << std::endl;

	{
		struct cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, check_reverse_kernel));
		int nThreads = min(attr.maxThreadsPerBlock, int(iComp.dim()));
		if(iComp.length() > 0)
			nThreads = min(nThreads,
			               int(attr.maxDynamicSharedSizeBytes / (iComp.length() * sizeof(int))));
		if(nThreads == 0) nThreads = 1;
		int nGrids = (iComp.dim() / nThreads) + (iComp.dim() % nThreads == 0 ? 0 : 1);
		if(nGrids == 0) nGrids = 1;
		auto shMem = nThreads * (iComp.length() * sizeof(int));

		std::cout << "# nGrids = " << nGrids << ", nThreads = " << nThreads << std::endl;
		check_reverse_kernel<<<nGrids, nThreads, shMem>>>(diComp.ptr());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
	}
	std::cout << "# Passed check_reverse_kernel.\n#" << std::endl;
}

TEST_CASE("IntegerComposition_onGPU", "test") {
	size_t pValue;
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;
	pValue *= 32;
	cuCHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, pValue));
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;

	{
		test_IntegerComposition(0, 0, 0);
		std::cout << std::endl;
	}

	// Dimension check for hard-core boson case
	int LMax = 20;
	int NMax = 12;
	for(auto N = 1; N <= NMax; ++N)
		for(auto L = N; L <= LMax; ++L) {
			test_IntegerComposition(N, L, 1);
			std::cout << std::endl;
		}

	LMax = 12;
	NMax = 9;
	// Dimension check for soft-core boson case
	for(auto N = 1; N <= NMax; ++N)
		for(auto L = 1; L <= LMax; ++L) {
			test_IntegerComposition(N, L, N);
			std::cout << std::endl;
		}
}