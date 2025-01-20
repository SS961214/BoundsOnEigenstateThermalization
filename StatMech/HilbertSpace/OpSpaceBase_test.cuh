#pragma once

#include "OpSpaceBase_test.hpp"
#include "ObjectOnGPU.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<class Derived>
__global__ void calculate_GramMat_kernel(Derived const* opSpacePtr, Index size, Index* indices,
                                         typename Derived::Scalar* res) {
	using Scalar        = typename Derived::Scalar;
	auto const& opSpace = *opSpacePtr;
	Index const idx1    = blockIdx.x * blockDim.x + threadIdx.x;
	Index const idx2    = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx1 >= size || idx2 >= size) return;
	if(idx1 > idx2) return;

	int const                  threadId = threadIdx.x + blockDim.x * threadIdx.y;
	extern __shared__ int      data[];
	int const                  workSize = opSpace.actionWorkSize();
	Eigen::Map<Eigen::ArrayXi> work(data + workSize * threadId, workSize);

	Index const opNum1 = indices[idx1];
	Index const opNum2 = indices[idx2];

	auto& elem = res[idx1 + size * idx2];
	elem       = 0.0;
	Index  inner1, inner2;
	Scalar coeff1, coeff2;
	for(Index k = 0; k < opSpace.baseSpace().dim(); ++k) {
		opSpace.action(inner2, coeff2, opNum2, k, work);
		opSpace.action(inner1, coeff1, opNum1, k, work);
		if(inner1 == inner2) elem += conj(coeff1) * coeff2;
	}
	res[idx2 + size * idx1] = conj(elem);
}

template<class Derived>
__host__ void test_OpSpace(ObjectOnGPU<Derived> const& dOpSpace, Derived const& hOpSpace) {
	static_assert(std::is_convertible_v<Derived, OpSpaceBase<Derived>>);
	std::cout << "# opSpace.dim() = " << hOpSpace.dim() << std::endl;
	if(hOpSpace.dim() == 0) return;
	constexpr double precision    = 1.0e-12;
	auto const       innerProduct = [&](Index j, Index k) {
        return (hOpSpace.basisOp(j).adjoint() * hOpSpace.basisOp(k)).eval().diagonal().sum();
	};
	using Scalar = typename Derived::Scalar;

	Index                                nSample = 1000;
	std::random_device                   seed_gen;
	std::default_random_engine           engine(seed_gen());
	std::uniform_int_distribution<Index> dist(0, hOpSpace.dim() - 1);
	Eigen::ArrayX<Index>                 index;
	if(nSample >= hOpSpace.dim()) {
		index.resize(hOpSpace.dim());
#pragma omp parallel for
		for(Index j = 0; j != hOpSpace.dim(); ++j) index(j) = j;
		nSample = hOpSpace.dim();
	}
	else {
		index = index.NullaryExpr(nSample, [&]() { return dist(engine); });
	}

	{
		int const                 shMemPerThread = hOpSpace.actionWorkSize() * sizeof(int);
		int const                 requestedSize  = nSample;
		struct cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, calculate_GramMat_kernel<Derived>));
		int nThreads = min(attr.maxThreadsPerBlock, requestedSize * requestedSize);
		if(shMemPerThread > 0)
			nThreads = min(nThreads, int(attr.maxDynamicSharedSizeBytes / shMemPerThread));
		if(nThreads == 0) nThreads = 1;
		nThreads    = std::sqrt(nThreads);
		int nBlocks = (requestedSize / nThreads) + (requestedSize % nThreads == 0 ? 0 : 1);
		if(nBlocks == 0) nBlocks = 1;
		assert(nBlocks * nThreads >= requestedSize);
		dim3 const blocks(nBlocks, nBlocks);
		dim3 const threads(nThreads, nThreads);
		auto const shMem = (nThreads * nThreads) * shMemPerThread;

		thrust::device_vector<Index>  dIndices(index.begin(), index.end());
		thrust::device_vector<Scalar> dRes(nSample * nSample, 0);
		calculate_GramMat_kernel<<<blocks, threads, shMem>>>(
		    dOpSpace.ptr(), nSample, dIndices.data().get(), dRes.data().get());
		thrust::host_vector<Scalar> res(dRes);

		thrust::host_vector<Scalar> innerProd(res.size());
#pragma omp parallel for schedule(dynamic, 10)
		for(Index idx1 = 0; idx1 < nSample; ++idx1)
			for(Index idx2 = idx1; idx2 < nSample; ++idx2) {
				Index const opNum1               = index[idx1];
				Index const opNum2               = index[idx2];
				innerProd[idx1 + nSample * idx2] = innerProduct(opNum1, opNum2);
			}

		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());

#pragma omp parallel for schedule(dynamic, 10)
		for(Index idx1 = 0; idx1 < nSample; ++idx1)
			for(Index idx2 = idx1; idx2 < nSample; ++idx2)
				REQUIRE(abs(res[idx1 + nSample * idx2] - innerProd[idx1 + nSample * idx2])
				        < precision);
	}
}