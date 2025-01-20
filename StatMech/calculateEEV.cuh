#pragma once
#include "globalWorkSpaceManagement.cuh"
#include <HilbertSpace>
#include <MatrixGPU>
#include <iostream>
#include <cassert>

namespace StatMech {
	template<class Matrix_, class OpSpace_, typename Scalar_, typename RealScalar_>
	__device__ void calculateEEV(Matrix_&          res, RealScalar_ const* __restrict__ dEigValPtr,
	                             RealScalar_ const dE, Scalar_ const* __restrict__ dEigVecPtr,
	                             Index const LD, SparseMatrix<Scalar_> const& adBasis,
	                             mBodyOpSpace<OpSpace_, Scalar_> const& opSpace, int blockSize,
	                             Index const* __restrict__ eqClassRep,
	                             Scalar_* __restrict__ const expvalPtr,
	                             int* __restrict__ const workPtr) {
		using Scalar     = Scalar_;
		using RealScalar = RealScalar_;
		static_assert(std::is_same_v<RealScalar, typename Eigen::NumTraits<Scalar>::Real>);
		static_assert(std::is_same_v<std::decay_t<OpSpace_>, ManyBodyBosonSpace>
		              || std::is_same_v<std::decay_t<OpSpace_>, ManyBodyFermionSpace>);
		static_assert(std::is_same_v<typename Matrix_::Scalar, Scalar_>
		              || std::is_same_v<typename Matrix_::Scalar, RealScalar>);

		// Declare variables
		int const                                          sectorDim = adBasis.rows();
		Eigen::Map<Eigen::VectorX<RealScalar> const> const eigVals(dEigValPtr, sectorDim);
		Eigen::Map<Eigen::MatrixX<Scalar> const, 0, Eigen::OuterStride<>> const eigVecs(
		    dEigVecPtr, sectorDim, sectorDim, Eigen::OuterStride<>(LD));
		int const dimHtot = opSpace.baseSpace().dim();

		/** Step 0: Configure shared memory */
		// Eigen::Map<Eigen::VectorX<Scalar>> expval(reinterpret_cast<Scalar*>(work), sectorDim);
		// int* __restrict__ smWorkPtr = reinterpret_cast<int*>(expval.data() + expval.size())
		//                               + opSpace.actionWorkSize() * threadIdx.x;
		// Eigen::Map<Eigen::ArrayXi> smWork(smWorkPtr, opSpace.actionWorkSize());
		Eigen::Map<Eigen::VectorX<Scalar>> expval(expvalPtr, sectorDim);
		int* __restrict__ const thWorkPtr = workPtr + opSpace.actionWorkSize() * threadIdx.x;
		Eigen::Map<Eigen::ArrayXi> work(thWorkPtr, opSpace.actionWorkSize());
		assert(dimHtot == adBasis.cols());
		assert(sectorDim == adBasis.rows());
		assert(sectorDim == expval.size());
		assert(sectorDim == eigVecs.rows());
		assert(sectorDim == eigVecs.cols());

		for(auto opIdx = 0; opIdx < blockSize; ++opIdx) {
			Index const opOrdinal = eqClassRep[opIdx];
			/** Initialize the expectation values */
			__syncthreads();
			for(int alpha = threadIdx.x; alpha < sectorDim; alpha += blockDim.x)
				expval(alpha) = 0.0;
			__syncthreads();

			/** Step 1: Compute eigenstate expectation values of basis operators */
			for(Index inBasis = threadIdx.x; inBasis < dimHtot; inBasis += blockDim.x) {
				auto   outBasis = inBasis;
				Scalar coeff    = 1.0;
				opSpace.action(outBasis, coeff, opOrdinal, inBasis, work);
				// if(coeff == 0.0) continue;

				for(auto pos2 = adBasis.outerIndexPtr()[inBasis];
				    pos2 < adBasis.outerIndexPtr()[inBasis + 1]; ++pos2) {
					assert(adBasis.outerIndexPtr()[inBasis] == inBasis);
					auto const   idx2     = adBasis.innerIndexPtr()[pos2];
					Scalar const adBElem2 = adBasis.valuePtr()[pos2];
					assert(0 <= idx2 && idx2 < sectorDim);

					for(auto pos1 = adBasis.outerIndexPtr()[outBasis];
					    pos1 < adBasis.outerIndexPtr()[outBasis + 1]; ++pos1) {
						assert(adBasis.outerIndexPtr()[outBasis] == outBasis);
						auto const   idx1     = adBasis.innerIndexPtr()[pos1];
						Scalar const adBElem1 = adBasis.valuePtr()[pos1];
						assert(0 <= idx1 && idx1 < sectorDim);

						// TODO: Optimize this part by using __syncthreads() instead of atomicAdd
						for(auto beta = 0; beta < sectorDim; ++beta) {
							// To avoid bank conflicts in accessing "expval".
							auto const   alpha = (beta + threadIdx.x) % sectorDim;
							Scalar const elem  = conj(eigVecs(idx1, alpha)) * adBElem1 * coeff
							                    * conj(adBElem2) * eigVecs(idx2, alpha);
							// This instruction takes much time. (without) 3s → (with) 11s
							atomicAdd(reinterpret_cast<RealScalar*>(&expval(alpha)), real(elem));
							atomicAdd(reinterpret_cast<RealScalar*>(&expval(alpha)) + 1,
							          imag(elem));
						}
					}
				}
			}
			__syncthreads();

			/** Step 2: Compute the difference between EEV of basis operators and the microcanonical average */
			for(auto alpha = threadIdx.x; alpha < sectorDim; alpha += blockDim.x) {
				int idMin = alpha, idMax = alpha;
				for(idMin = alpha; idMin >= 0; --idMin) {
					if(eigVals(alpha) - eigVals(idMin) > dE) break;
				}
				++idMin;
				for(idMax = alpha; idMax < sectorDim; ++idMax) {
					if(eigVals(idMax) - eigVals(alpha) > dE) break;
				}
				--idMax;

				Scalar mcAve = 0.0;
				for(auto beta = idMin; beta <= idMax; ++beta) mcAve += expval(beta);
				mcAve /= RealScalar(idMax - idMin + 1);
				res(alpha, opIdx) = expval(alpha) - mcAve;
			}
		}
	}

	template<class OpSpace_, typename Scalar_, typename RealScalar_>
	__global__ void calculateEEV_kernel(
	    Scalar_* __restrict__ res, unsigned long long* sm_slots,
	    Scalar_* const __restrict__* gWorks, RealScalar_ const* __restrict__ dEigValPtr,
	    RealScalar_ const dE, Scalar_ const* __restrict__ dEigVecPtr, Index const LD,
	    SparseMatrix<Scalar_> const* __restrict__ adBasisPtr,
	    mBodyOpSpace<OpSpace_, Scalar_> const* __restrict__ opSpacePtr, int const blockSize,
	    Index const* __restrict__ eqClassRepPtr) {
		using Scalar     = Scalar_;
		using RealScalar = RealScalar_;
		static_assert(std::is_same_v<RealScalar, typename Eigen::NumTraits<Scalar>::Real>);
		static_assert(std::is_same_v<std::decay_t<OpSpace_>, ManyBodyBosonSpace>
		              || std::is_same_v<std::decay_t<OpSpace_>, ManyBodyFermionSpace>);

		if(blockIdx.x >= blockSize) return;
		SparseMatrix<Scalar> const&                        adBasis   = *adBasisPtr;
		int const                                          sectorDim = adBasis.rows();
		Eigen::Map<Eigen::VectorX<RealScalar> const> const eigVals(dEigValPtr, sectorDim);
		Eigen::Map<Eigen::MatrixX<Scalar> const, 0, Eigen::OuterStride<>> const eigVecs(
		    dEigVecPtr, sectorDim, sectorDim, Eigen::OuterStride<>(LD));

		auto const& opSpace = *opSpacePtr;
		int const   dimHtot = opSpace.baseSpace().dim();

		Index const opIdx = eqClassRepPtr[blockIdx.x];
		assert(opIdx < opSpace.dim());

		// Step 0: Prepare memories
		// TODO: Use global memory instead of shared memory to store "expval"
		// Get a work slot for this block
		int const                       my_sm   = get_smid();
		int                             my_slot = -1;
		Scalar* __restrict__ __shared__ workPtr;
		if(threadIdx.x == 0) {
			DEBUG(printf("# %s: block = %d/%d,\t blockSize = %d\n", __func__, blockIdx.x, gridDim.x,
			             blockSize));
			my_slot = get_slot(sm_slots + my_sm);
			workPtr = gWorks[my_sm * slots_per_sm + my_slot];
			assert(my_sm >= 0);
			assert(0 <= my_slot && my_slot < slots_per_sm);
			// asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
		}
		__syncthreads();
		assert(workPtr != nullptr);
		// (END) Get a work slot for this block
		Eigen::Map<Eigen::VectorX<Scalar>> expval(workPtr, sectorDim);
		for(auto alpha = threadIdx.x; alpha < sectorDim; alpha += blockDim.x) expval(alpha) = 0.0;
		__syncthreads();
		// extern __shared__ __align__(sizeof(Scalar)) double shMem[];
		// Scalar* __restrict__ dataPtr = reinterpret_cast<Scalar*>(shMem);
		// Eigen::Map<Eigen::VectorX<Scalar>> expval(dataPtr, sectorDim);
		// for(auto alpha = threadIdx.x; alpha < sectorDim; alpha += blockDim.x) expval(alpha) = 0.0;
		// __syncthreads();
		// TODO: Use shared memory soley for "work"
		extern __shared__ int sm_data[];
		int* __restrict__ smWorkPtr = sm_data + opSpace.actionWorkSize() * threadIdx.x;
		Eigen::Map<Eigen::ArrayXi> work(smWorkPtr, opSpace.actionWorkSize());

		// Step 1: Compute eigenstate expectation values
		for(Index inBasis = threadIdx.x; inBasis < dimHtot; inBasis += blockDim.x) {
			auto   outBasis = inBasis;
			Scalar coeff    = 1.0;
			opSpace.action(outBasis, coeff, opIdx, inBasis, work);

			for(auto pos2 = adBasis.outerIndexPtr()[inBasis];
			    pos2 < adBasis.outerIndexPtr()[inBasis + 1]; ++pos2) {
				assert(adBasis.outerIndexPtr()[inBasis] == inBasis);
				auto const   idx2     = adBasis.innerIndexPtr()[pos2];
				Scalar const adBElem2 = adBasis.valuePtr()[pos2];
				assert(0 <= idx2 && idx2 < sectorDim);

				for(auto pos1 = adBasis.outerIndexPtr()[outBasis];
				    pos1 < adBasis.outerIndexPtr()[outBasis + 1]; ++pos1) {
					assert(adBasis.outerIndexPtr()[outBasis] == outBasis);
					auto const   idx1     = adBasis.innerIndexPtr()[pos1];
					Scalar const adBElem1 = adBasis.valuePtr()[pos1];
					assert(0 <= idx1 && idx1 < sectorDim);

					for(auto beta = 0; beta < sectorDim; ++beta) {
						// To avoid bank conflicts in accessing "expval".
						auto const   alpha = (beta + threadIdx.x) % sectorDim;
						Scalar const elem  = conj(eigVecs(idx1, alpha)) * adBElem1 * coeff
						                    * conj(adBElem2) * eigVecs(idx2, alpha);
						// This instruction takes much time. (without) 3s → (with) 11s
						atomicAdd(reinterpret_cast<RealScalar*>(&expval(alpha)), real(elem));
						atomicAdd(reinterpret_cast<RealScalar*>(&expval(alpha)) + 1, imag(elem));
					}
				}
			}
		}
		__syncthreads();

		// Step 2: Compute the difference between EEV and the microcanonical average
		for(auto alpha = threadIdx.x; alpha < sectorDim; alpha += blockDim.x) {
			int idMin = alpha, idMax = alpha;
			for(idMin = alpha; idMin >= 0; --idMin) {
				if(eigVals(alpha) - eigVals(idMin) > dE) break;
			}
			++idMin;
			for(idMax = alpha; idMax < sectorDim; ++idMax) {
				if(eigVals(idMax) - eigVals(alpha) > dE) break;
			}
			--idMax;

			Scalar mcAve = 0.0;
			for(auto beta = idMin; beta <= idMax; ++beta) mcAve += expval(beta);
			mcAve /= RealScalar(idMax - idMin + 1);
			res[alpha + sectorDim * blockIdx.x] = expval(alpha) - mcAve;
		}

		__syncthreads();
		if(threadIdx.x == 0) release_slot(sm_slots + my_sm, my_slot);

		return;
	}
}  // namespace StatMech