#pragma once
#include "globalWorkSpaceManagement.cuh"
#include <HilbertSpace>
#include <cuda/std/complex>

namespace StatMech {
	template<class Matrix_, class OpSpace_, typename Scalar_>
	__device__ void calculateGramMat(Matrix_&                               gramMat,
	                                 mBodyOpSpace<OpSpace_, Scalar_> const& opSpace, int blockSize,
	                                 Index const* __restrict__ eqClassRep, int* const    work) {
		using Scalar     = Scalar_;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		static_assert(std::is_same_v<std::decay_t<OpSpace_>, ManyBodyBosonSpace>
		              || std::is_same_v<std::decay_t<OpSpace_>, ManyBodyFermionSpace>);
		static_assert(std::is_same_v<typename Matrix_::Scalar, Scalar_>
		              || std::is_same_v<typename Matrix_::Scalar, RealScalar>);

		// Configure shared memory
		Eigen::Map<Eigen::ArrayX<RealScalar>> partRes(reinterpret_cast<RealScalar*>(work),
		                                              blockDim.x);
		int* const smWorkPtr = reinterpret_cast<int*>(partRes.data() + partRes.size())
		                       + opSpace.actionWorkSize() * threadIdx.x;
		Eigen::Map<Eigen::ArrayXi> smWork(smWorkPtr, opSpace.actionWorkSize());

		// Calculate the entries of the Gram matrix
		for(int idx1 = 0; idx1 < blockSize; ++idx1) {
			Index const opIdx1 = eqClassRep[idx1];
			{
				// Divide and calculate the inner product
				partRes(threadIdx.x) = 0.0;
				for(Index inBasis = threadIdx.x; inBasis < opSpace.baseDim();
				    inBasis += blockDim.x) {
					auto   outBasis1 = inBasis;
					Scalar coeff1    = 1.0;
					opSpace.action(outBasis1, coeff1, opIdx1, inBasis, smWork);
					partRes[threadIdx.x] += (cuda::std::real(cuda::std::conj(coeff1) * coeff1));
				}
				__syncthreads();

				// Reduction
				int size = blockDim.x;
				while(size > 1) {
					if((size & 1) && threadIdx.x == 0) partRes[0] += partRes[size - 1];
					size = (size >> 1);
					if(threadIdx.x < size) partRes[threadIdx.x] += partRes[threadIdx.x + size];
					__syncthreads();
				}
			}
			if(threadIdx.x == 0) { gramMat(idx1, idx1) = partRes[0]; }

			for(int idx2 = 0; idx2 < idx1; ++idx2) {
				Index const opIdx2 = eqClassRep[idx2];
				{
					// Divide and calculate the inner product
					partRes(threadIdx.x) = 0.0;
					for(Index inBasis = threadIdx.x; inBasis < opSpace.baseDim();
					    inBasis += blockDim.x) {
						auto   outBasis1 = inBasis;
						Scalar coeff1    = 1.0;
						opSpace.action(outBasis1, coeff1, opIdx1, inBasis, smWork);

						auto   outBasis2 = inBasis;
						Scalar coeff2    = 1.0;
						opSpace.action(outBasis2, coeff2, opIdx2, inBasis, smWork);

						if(outBasis1 == outBasis2)
							partRes[threadIdx.x]
							    += (cuda::std::real(cuda::std::conj(coeff1) * coeff2));
					}
					__syncthreads();

					// Reduction
					int size = blockDim.x;
					while(size > 1) {
						if((size & 1) && threadIdx.x == 0) partRes[0] += partRes[size - 1];
						size = (size >> 1);
						if(threadIdx.x < size) partRes[threadIdx.x] += partRes[threadIdx.x + size];
						__syncthreads();
					}
					// assert(abs((partRes[0] - sum) / (partRes[0] + sum)) < precision);
				}
				if(threadIdx.x == 0) {
					gramMat(idx1, idx2) = partRes[0];
					gramMat(idx2, idx1) = partRes[0];
				}
			}
		}
	}

	//---------- Tested on 2024-09-28 with "calculateGramMat_kernel_test.cu" ----------//
	template<class OpSpace_, typename Scalar_, typename RealScalar_>
	__global__ void calculateGramMat_kernel(
	    RealScalar_* __restrict__ res,
	    mBodyOpSpace<OpSpace_, Scalar_> const* __restrict__ opSpacePtr, int const blockSize,
	    Index const* __restrict__ eqClassRep) {
		using Scalar     = Scalar_;
		using RealScalar = RealScalar_;
		int const idx    = blockIdx.x;
		if(idx >= blockSize * (blockSize + 1) / 2) return;

		auto const&              opSpace = *opSpacePtr;
		extern __shared__ double shMem[];
		double* __restrict__ dataPtr = reinterpret_cast<double*>(shMem);
		Eigen::Map<Eigen::VectorXd> partialRes(dataPtr, blockDim.x);
		partialRes[threadIdx.x]   = 0.0;
		int* __restrict__ workPtr = reinterpret_cast<int*>(partialRes.data() + partialRes.size())
		                            + opSpace.actionWorkSize() * threadIdx.x;
		Eigen::Map<Eigen::ArrayXi> work(workPtr, opSpace.actionWorkSize());

		int const   idx1   = sqrt(2 * idx + 0.25) - 0.5;
		int const   idx2   = idx - idx1 * (idx1 + 1) / 2;
		Index const opIdx1 = eqClassRep[idx1];
		Index const opIdx2 = eqClassRep[idx2];
		for(Index inBasis = threadIdx.x; inBasis < opSpace.baseDim(); inBasis += blockDim.x) {
			auto   outBasis1 = inBasis;
			Scalar coeff1    = 1.0;
			opSpace.action(outBasis1, coeff1, opIdx1, inBasis, work);

			auto   outBasis2 = inBasis;
			Scalar coeff2    = 1.0;
			opSpace.action(outBasis2, coeff2, opIdx2, inBasis, work);

			if(outBasis1 == outBasis2)
				partialRes[threadIdx.x] += (cuda::std::real(cuda::std::conj(coeff1) * coeff2));
		}
		__syncthreads();

		// Reduction
		int size = blockDim.x;
		while(size > 1) {
			if((size & 1) && threadIdx.x == 0) partialRes[0] += partialRes[size - 1];
			size >>= 1;
			if(threadIdx.x < size) partialRes[threadIdx.x] += partialRes[threadIdx.x + size];
			__syncthreads();
		}
		if(threadIdx.x == 0) {
			res[idx1 + blockSize * idx2] = partialRes[0];
			res[idx2 + blockSize * idx1] = partialRes[0];
		}

		return;
	}
}  // namespace StatMech