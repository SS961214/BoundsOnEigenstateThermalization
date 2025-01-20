#pragma once

#include <MatrixGPU>
#include <HilbertSpace>
#include <Eigen/Dense>

template<typename Scalar>
__global__ void construct_globalOp_onGPU_kernel(Index const dim, Scalar* res, Index const resLD,
                                                Index const dimLocOp, Scalar const* locOp,
                                                Index const                 locOpLD,
                                                SparseMatrix<Scalar> const* basisPtr) {
	Index idx1 = blockIdx.x * blockDim.x + threadIdx.x;
	Index idx2 = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx1 >= dim || idx2 >= dim) return;
	if(idx1 > idx2) return;

	SparseMatrix<Scalar> const& basis = *basisPtr;

	Scalar coeff = 0.0;
	for(Index pos2 = basis.outerIndexPtr()[idx2]; pos2 < basis.outerIndexPtr()[idx2 + 1]; ++pos2) {
		Index const  inBasis    = basis.innerIndexPtr()[pos2];
		Scalar const basisElem2 = basis.valuePtr()[pos2];
		Index const  locIdx2    = inBasis % dimLocOp;
		Index const  residual2  = inBasis / dimLocOp;

		for(Index pos1 = basis.outerIndexPtr()[idx1]; pos1 < basis.outerIndexPtr()[idx1 + 1];
		    ++pos1) {
			Index const  outBasis   = basis.innerIndexPtr()[pos1];
			Scalar const basisElem1 = basis.valuePtr()[pos1];
			Index const  locIdx1    = outBasis % dimLocOp;
			Index const  residual1  = outBasis / dimLocOp;
			if(residual1 != residual2) continue;

			coeff += conj(basisElem1) * locOp[locIdx1 + locOpLD * locIdx2] * basisElem2;

			// if(outBasis != inBasis) continue;
			// coeff += conj(basisElem1) * basisElem2;
		}
	}
	res[idx1 + resLD * idx2] = coeff;
	if(idx1 == idx2)
		res[idx1 + resLD * idx2] = real(coeff);
	else
		res[idx2 + resLD * idx1] = conj(res[idx1 + resLD * idx2]);
}

template<typename Scalar, class TotalSpace_>
GPU::MatrixGPU<Eigen::MatrixX<Scalar>> construct_globalOp_onGPU(
    Eigen::MatrixX<Scalar> const& locOp, TransSector<TotalSpace_, Scalar> const& subSpace) {
	auto const& basis    = subSpace.basis();
	auto const  dimLocOp = locOp.rows();

	GPU::MatrixGPU<std::decay_t<decltype(locOp)>>       res(subSpace.dim(), subSpace.dim());
	GPU::MatrixGPU<std::decay_t<decltype(locOp)>> const dLocOp(locOp);
	ObjectOnGPU<SparseMatrix<Scalar>> const             dBasis(subSpace.basis());

	assert(res.cols() == res.rows());
	assert(locOp.cols() == locOp.rows());

	struct cudaFuncAttributes attr;
	cuCHECK(cudaFuncGetAttributes(&attr, construct_globalOp_onGPU_kernel<Scalar>));
	int const shMem = dimLocOp * dimLocOp * sizeof(Scalar);
	assert(shMem <= attr.maxDynamicSharedSizeBytes);
	int const nThreads = min(subSpace.dim(), Index(std::sqrt(attr.maxThreadsPerBlock)));
	int const nBlocks  = subSpace.dim() / nThreads + (subSpace.dim() % nThreads == 0 ? 0 : 1);
	assert(nBlocks * nThreads >= subSpace.dim());
	dim3 const blocks(nBlocks, nBlocks);
	dim3 const threads(nThreads, nThreads);
	std::cout << "#\t  attr.maxDynamicSharedSizeBytes = " << attr.maxDynamicSharedSizeBytes
	          << std::endl;
	std::cout << "#\t         attr.maxThreadsPerBlock = " << attr.maxThreadsPerBlock << std::endl;
	std::cout << "#\t  nBlocks = " << nBlocks << std::endl;
	std::cout << "#\t nThreads = " << nThreads << std::endl;
	std::cout << "#\t nThreads*nThreads = " << nThreads * nThreads << std::endl;
	std::cout << "#\t    shMem = " << shMem << std::endl;

	construct_globalOp_onGPU_kernel<<<blocks, threads>>>(
	    res.cols(), res.data(), res.LD(), dLocOp.cols(), dLocOp.data(), dLocOp.LD(), dBasis.ptr());
	cuCHECK(cudaGetLastError());
	cuCHECK(cudaDeviceSynchronize());

	return res;
}