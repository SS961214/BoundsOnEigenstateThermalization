#pragma once

#include "tests.hpp"
#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"
#include "ManyBodyOpSpaceBase.hpp"
#include "ObjectOnGPU.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<class Derived>
__global__ void check_basics_kernel(Derived const* mbSpacePtr, Index* res) {
	printf("# %s\n", __func__);
	auto const& mbSpace = *mbSpacePtr;
	printf("##\t mbSpace.dim() = %lu\n", static_cast<unsigned long>(mbSpace.dim()));
	res[0] = mbSpace.dim();
	res[1] = mbSpace.sysSize();
}

template<class Derived>
__global__ void check_bijectiveness_kernel(Derived const* mbSpacePtr, Index* result,
                                           Index* config) {
	auto const& mbSpace  = *mbSpacePtr;
	Index const stateNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(stateNum >= mbSpace.dim()) return;
	if(stateNum == 0) printf("# %s\n", __func__);

	extern __shared__ int      data[];
	Eigen::Map<Eigen::ArrayXi> work(data + mbSpace.sysSize() * threadIdx.x, mbSpace.sysSize());

	mbSpace.ordinal_to_config(work, stateNum);
	result[stateNum] = mbSpace.config_to_ordinal(work);
	for(auto pos = 0; pos < mbSpace.sysSize(); ++pos) {
		config[stateNum * mbSpace.sysSize() + pos] = mbSpace.locState(stateNum, pos);
	}
}

template<class Derived>
__global__ void check_translation_kernel(Derived const* mbSpacePtr, Index transEqDim,
                                         int* transEqClassRep, int* transPeriod, int* appeared) {
	auto const& mbSpace = *mbSpacePtr;
	Index const eqClass = blockIdx.x * blockDim.x + threadIdx.x;
	if(eqClass >= transEqDim) return;
	if(eqClass == 0) printf("# %s\n", __func__);

	extern __shared__ int      data[];
	Eigen::Map<Eigen::ArrayXi> work(data + mbSpace.sysSize() * threadIdx.x, mbSpace.sysSize());

	auto const eqClassRep = transEqClassRep[eqClass];
	atomicAdd(&appeared[eqClassRep], 1);
	for(auto trans = 1; trans < transPeriod[eqClass]; ++trans) {
		auto const translated = mbSpace.translate(eqClassRep, trans, work);
		atomicAdd(&appeared[translated], 1);
	}
}

// template<class Derived>
// __global__ void check_computeTransEqClass_kernel(Derived const* mbSpacePtr, Index* dimPtr) {
// 	mbSpacePtr->compute_transEqClass();
// 	*dimPtr = mbSpacePtr->transEqDim();
// }

// template<class Derived>
// __global__ void check_transEqClass_kernel(Derived const* mbSpacePtr, int* appeared) {
// 	auto const& mbSpace    = *mbSpacePtr;
// 	Index const eqClassNum = blockIdx.x * blockDim.x + threadIdx.x;
// 	if(eqClassNum >= mbSpace.transEqDim()) return;
// 	if(eqClassNum == 0) printf("# %s\n", __func__);

// 	auto const stateNum = mbSpace.transEqClassRep(eqClassNum);
// 	atomicAdd(&appeared[stateNum], 1);
// 	// printf("# dim=%d, eqClassNum=%d, period=%d, trans=0, translated=%d\n", int(mbSpace.dim()), int(eqClassNum), int(mbSpace.transPeriod(eqClassNum)), int(stateNum));
// 	for(auto trans = 1; trans < mbSpace.transPeriod(eqClassNum); ++trans) {
// 		auto const translated = mbSpace.translate(stateNum, trans);
// 		atomicAdd(&appeared[translated], 1);
// 		// printf("# dim=%d, eqClassNum=%d, period=%d, trans=%d, translated=%d\n", int(mbSpace.dim()), int(eqClassNum), int(mbSpace.transPeriod(eqClassNum)), int(trans), int(translated));
// 	}
// }

// template<class Derived>
// __global__ void check_stateToTransEqClass_kernel(Derived const* mbSpacePtr, Index* shifted) {
// 	auto const& mbSpace  = *mbSpacePtr;
// 	Index const stateNum = blockIdx.x * blockDim.x + threadIdx.x;
// 	if(stateNum >= mbSpace.dim()) return;
// 	if(stateNum == 0) printf("# %s\n", __func__);

// 	auto const eqClass    = mbSpace.state_to_transEqClass(stateNum);
// 	auto const eqClassRep = mbSpace.transEqClassRep(eqClass);
// 	auto const trans      = mbSpace.state_to_transShift(stateNum);
// 	if(!(0 <= stateNum && stateNum < mbSpace.dim())) {
// 		printf("##\t stateNum=%d, eqClass=%d, eqClassRep=%d, trans=%d\n", int(stateNum),
// 		       int(eqClass), int(eqClassRep), int(trans));
// 	}
// 	shifted[stateNum] = mbSpace.translate(eqClassRep, trans);
// }

template<class Derived>
__global__ void check_reverseOp_kernel(Derived const* mbSpacePtr, Index* reversed) {
	auto const& mbSpace  = *mbSpacePtr;
	int const   stateNum = blockIdx.x * blockDim.x + threadIdx.x;
	if(stateNum >= mbSpace.dim()) return;
	if(stateNum == 0) printf("# %s\n", __func__);

	extern __shared__ int      data[];
	Eigen::Map<Eigen::ArrayXi> work(data + mbSpace.sysSize() * threadIdx.x, mbSpace.sysSize());

	reversed[stateNum] = mbSpace.reverse(stateNum, work);
}

template<class Derived>
__host__ void test_ManyBodySpaceBase(ObjectOnGPU<Derived> const& dmbSpace,
                                     Derived const&              hmbSpace) {
	static_assert(std::is_convertible_v<Derived, ManyBodySpaceBase<Derived>>);
	std::cout << "# " << __PRETTY_FUNCTION__ << std::endl;

	{
		std::cout << "##\t mbSpace.dim() = " << hmbSpace.dim()
		          << ", mbSpace.sysSize() = " << hmbSpace.sysSize() << std::endl;
		thrust::device_vector<Index> dRes(2);
		check_basics_kernel<<<1, 1>>>(dmbSpace.ptr(), dRes.data().get());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
		REQUIRE(hmbSpace.dim() == dRes[0]);
		REQUIRE(hmbSpace.sysSize() == dRes[1]);
	}
	if(hmbSpace.dim() == 0) return;

	// test locState
	// test ordinal_to_config
	// test config_to_ordinal
	{
		int const                 shMemPerThread = hmbSpace.sysSize() * sizeof(int);
		int const                 requestedSize  = hmbSpace.dim();
		struct cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, check_bijectiveness_kernel<Derived>));
		int nThreads = min(attr.maxThreadsPerBlock, requestedSize);
		if(shMemPerThread > 0)
			nThreads = min(nThreads, int(attr.maxDynamicSharedSizeBytes / shMemPerThread));
		if(nThreads == 0) nThreads = 1;
		int nGrids = (requestedSize / nThreads) + (requestedSize % nThreads == 0 ? 0 : 1);
		if(nGrids == 0) nGrids = 1;
		auto const shMem = nThreads * shMemPerThread;

		thrust::device_vector<Index> dRes(hmbSpace.dim(), 0);
		thrust::device_vector<Index> dConfig(hmbSpace.dim() * hmbSpace.sysSize());
		check_bijectiveness_kernel<<<nGrids, nThreads, shMem>>>(dmbSpace.ptr(), dRes.data().get(),
		                                                        dConfig.data().get());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
		thrust::host_vector<Index> const res(dRes);
		thrust::host_vector<Index> const config(dConfig);
#pragma omp parallel for
		for(Index j = 0; j < hmbSpace.dim(); ++j) {
			REQUIRE(j == res[j]);
			auto const refConf = hmbSpace.ordinal_to_config(j);
			for(auto pos = 0; pos < hmbSpace.sysSize(); ++pos) {
				REQUIRE(refConf(pos) == config[j * hmbSpace.sysSize() + pos]);
			}
		}
		std::cout << "# Passed tests: check_bijectiveness_kernel.\n#" << std::endl;
	}

	// constexpr int threads = 512;
	// int           grids;
	// auto          upperQuotient = [&](int x, int y) { return (x % y == 0 ? x / y : x / y + 1); };

	// test for translation operations
	hmbSpace.compute_transEqClass();
	{
		int const                 shMemPerThread = hmbSpace.sysSize() * sizeof(int);
		int const                 requestedSize  = hmbSpace.transEqDim();
		struct cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, check_translation_kernel<Derived>));
		int nThreads = min(attr.maxThreadsPerBlock, requestedSize);
		if(shMemPerThread > 0)
			nThreads = min(nThreads, int(attr.maxDynamicSharedSizeBytes / shMemPerThread));
		if(nThreads == 0) nThreads = 1;
		int nGrids = (requestedSize / nThreads) + (requestedSize % nThreads == 0 ? 0 : 1);
		if(nGrids == 0) nGrids = 1;
		auto const shMem = nThreads * shMemPerThread;

		std::cout << "# nGrids = " << nGrids << ", nThreads = " << nThreads << ", shMem = " << shMem
		          << std::endl;
		thrust::host_vector<int> transEqClassRep(hmbSpace.transEqDim());
		thrust::host_vector<int> transPeriod(hmbSpace.transEqDim());
#pragma omp parallel for
		for(Index j = 0; j < hmbSpace.transEqDim(); ++j) {
			transEqClassRep[j] = hmbSpace.transEqClassRep(j);
			transPeriod[j]     = hmbSpace.transPeriod(j);
		}
		thrust::device_vector<int> dTransEqClassRep(transEqClassRep);
		thrust::device_vector<int> dTransPeriod(transPeriod);
		thrust::device_vector<int> dWork(hmbSpace.dim(), 0);
		check_translation_kernel<<<nGrids, nThreads, shMem>>>(
		    dmbSpace.ptr(), dTransEqClassRep.size(), dTransEqClassRep.data().get(),
		    dTransPeriod.data().get(), dWork.data().get());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
		thrust::host_vector<int> appeared(dWork);
		for(Index stateNum = 0; stateNum != hmbSpace.dim(); ++stateNum) {
			if(appeared[stateNum] != 1)
				std::cout << "##\t appeared[" << stateNum << "] = " << appeared[stateNum]
				          << std::endl;
			REQUIRE(appeared[stateNum] == 1);
		}
	}
	std::cout << "# Passed test: check_translation_kernel.\n#" << std::endl;

	// // computeTransEqClass within kernel is duplicated.
	// {
	// 	thrust::device_vector<Index> ddim(1);
	// 	check_computeTransEqClass_kernel<<<1,1,0,0>>>(dmbSpace.ptr(), ddim.data().get());
	// 	cuCHECK(cudaGetLastError());
	// 	cuCHECK(cudaDeviceSynchronize());
	// 	REQUIRE(hmbSpace.transEqDim() == ddim[0]);
	// 	std::cout << "##\t mbSpace.transEqDim() = " << hmbSpace.transEqDim() << std::endl;
	// }
	// {
	// 	grids = upperQuotient(hmbSpace.transEqDim(), threads);
	// 	std::cout << "##\t grids = " << grids << ", threads = " << threads << std::endl;
	// 	thrust::device_vector<int> dWork(hmbSpace.dim(), 0);
	// 	check_transEqClass_kernel<<<grids, threads>>>(dmbSpace.ptr(), dWork.data().get());
	// 	cuCHECK(cudaGetLastError());
	// 	cuCHECK(cudaDeviceSynchronize());
	// 	thrust::host_vector<int> appeared(dWork);
	// 	// std::cout << appeared << std::endl;
	// 	for(Index stateNum = 0; stateNum != hmbSpace.dim(); ++stateNum) {
	// 		if(appeared[stateNum] != 1)
	// 			std::cout << "##\t appeared[" << stateNum << "] = " << appeared[stateNum] << std::endl;
	// 		REQUIRE(appeared[stateNum] == 1);
	// 	}
	// }
	// std::cout << "# Passed test: check_transEqClass_kernel.\n#" << std::endl;

	// 	// test for state_to_transEqClass (Duplicated within kernel)
	// 	// test for state_to_transShift (Duplicated within kernel)
	// 	std::cout << "# check_stateToTransEqClass_kernel.\n#" << std::endl;
	// 	{
	// 		grids = upperQuotient(hmbSpace.dim(), threads);
	// 		std::cout << "##\t grids = " << grids << ", threads = " << threads << std::endl;
	// 		thrust::device_vector<Index> dWork(hmbSpace.dim(), 0);
	// 		check_stateToTransEqClass_kernel<<<grids, threads>>>(dmbSpace.ptr(), dWork.data().get());
	// 		cuCHECK(cudaGetLastError());
	// 		cuCHECK(cudaDeviceSynchronize());
	// 		thrust::host_vector<Index> shifted(dWork);
	// #pragma omp parallel for
	// 		for(Index stateNum = 0; stateNum != hmbSpace.dim(); ++stateNum) {
	// 			REQUIRE(shifted[stateNum] == stateNum);
	// 		}
	// 	}
	// 	std::cout << "# Passed test: check_stateToTransEqClass_kernel.\n#" << std::endl;

	// test for reverse()
	{
		int const                 shMemPerThread = hmbSpace.sysSize() * sizeof(int);
		int const                 requestedSize  = hmbSpace.dim();
		struct cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, check_reverseOp_kernel<Derived>));
		int nThreads = min(attr.maxThreadsPerBlock, requestedSize);
		if(shMemPerThread > 0)
			nThreads = min(nThreads, int(attr.maxDynamicSharedSizeBytes / shMemPerThread));
		if(nThreads == 0) nThreads = 1;
		int nGrids = (requestedSize / nThreads) + (requestedSize % nThreads == 0 ? 0 : 1);
		if(nGrids == 0) nGrids = 1;
		auto const shMem = nThreads * shMemPerThread;

		thrust::device_vector<Index> dWork(hmbSpace.dim(), 0);
		check_reverseOp_kernel<<<nGrids, nThreads, shMem>>>(dmbSpace.ptr(), dWork.data().get());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
		thrust::host_vector<Index> reversed(dWork);
#pragma omp parallel for
		for(auto j = 0; j < hmbSpace.dim(); ++j) { REQUIRE(reversed[j] == hmbSpace.reverse(j)); }
	}
	std::cout << "# Passed tests: check_reverseOp_kernel.\n#" << std::endl;
}