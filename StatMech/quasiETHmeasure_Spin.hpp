#pragma once

#include <HilbertSpace>
#include <Eigen/Dense>

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

#ifndef CUSTOM_OMP_FUNCTIONS
	#define CUSTOM_OMP_FUNCTIONS
	#if __has_include(<omp.h>)
		#include <omp.h>
__host__ __device__ static inline int get_max_threads() {
		#ifdef __CUDA_ARCH__
	return 1;
		#else
	return omp_get_max_threads();
		#endif
}
__host__ __device__ static inline int get_thread_num() {
		#ifdef __CUDA_ARCH__
	return 0;
		#else
	return omp_get_thread_num();
		#endif
}
	#else
constexpr static inline int get_max_threads() { return 1; }
constexpr static inline int get_thread_num() { return 0; }
	#endif
#endif

#include <iostream>

namespace StatMech {
	template<class Matrix, class Array, class TotalSpace_, typename Scalar_>
	__host__ Eigen::ArrayXd ETHmeasure2Sq(Matrix const& eigVecs, Array const& eigVals,
	                                      OpSpace<Scalar_> const&               opSpace,
	                                      SubSpace<TotalSpace_, Scalar_> const& subSpace,
	                                      double const                          shWidth) {
		std::cout << "# " << __PRETTY_FUNCTION__ << std::endl;
		assert(eigVecs.rows() == eigVals.size());
		assert(eigVecs.rows() == static_cast<long>(subSpace.dim()));
		assert(opSpace.baseDim() == subSpace.dimTot());

		Eigen::ArrayXd  res = Eigen::ArrayXd::Zero(eigVals.rows());
		Eigen::ArrayXXd expValues(eigVals.rows(), get_max_threads());

#pragma omp parallel for
		for(Index p = 0; p != opSpace.dim(); ++p) {
			auto&& expVals = expValues.col(get_thread_num());

			expVals
			    = (eigVecs.adjoint()
			       * (subSpace.basis().adjoint() * opSpace.basisOp(p) * subSpace.basis()) * eigVecs)
			          .diagonal()
			          .real();

			for(auto j = 0; j != eigVals.size(); ++j) {
				long idMin, idMax;
				for(idMin = j; idMin >= 0; --idMin) {
					if(eigVals(j) - eigVals(idMin) > shWidth) break;
				}
				++idMin;
				for(idMax = j; idMax < eigVals.size(); ++idMax) {
					if(eigVals(idMax) - eigVals(j) > shWidth) break;
				}
				--idMax;

				double MCave = 0.0;
				for(int k = idMin; k != idMax + 1; ++k) MCave += expVals(k);
				MCave /= static_cast<double>(idMax - idMin + 1);
				auto const diff = (expVals(j) - MCave) * (expVals(j) - MCave);
#pragma omp atomic
				res(j) += diff;
			}
		}
		return res;
	}

	template<class OpSpace_>
	struct IsSpinOpSpace : std::false_type {};
	template<typename Scalar_>
	struct IsSpinOpSpace<mBodyOpSpace<ManyBodySpinSpace, Scalar_>> : std::true_type {};

	template< class Matrix_, class Array_, class OpSpace_, class SubSpace_,
	          std::enable_if_t<IsSpinOpSpace<OpSpace_>::value>* = nullptr >
	__host__ Eigen::ArrayXd ETHmeasure2Sq(Matrix_ const& eigVecs, Array_ const& eigVals,
	                                      OpSpace_ const& opSpace, SubSpace_ const& subSpace,
	                                      double const shWidth) {
		using Scalar     = typename Matrix_::Scalar;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		using StateSpace = typename OpSpace_::BaseSpace;
		static_assert(std::is_same_v<RealScalar, typename Array_::Scalar>);
		static_assert(std::is_same_v<StateSpace, ManyBodySpinSpace>);
		static_assert(std::is_same_v<SubSpace_, TransSector<StateSpace, Scalar>>
		              || std::is_same_v<SubSpace_, TransParitySector<StateSpace, Scalar>>);
		assert(eigVecs.rows() == eigVals.size());
		assert(eigVecs.rows() == static_cast<long>(subSpace.dim()));
		assert(opSpace.baseDim() == static_cast<Index>(subSpace.dimTot()));

		opSpace.compute_transEqClass();
		std::cout << "\topSpace.transEqDim() = " << opSpace.transEqDim() << std::endl;

		Eigen::ArrayXd                           res = Eigen::ArrayXd::Zero(eigVals.rows());
		Eigen::ArrayXXd                          expValues(eigVals.rows(), get_max_threads());
		std::decay_t<decltype(subSpace.basis())> adjointBasis = subSpace.basis().adjoint();

		bool const parallelFlag = (opSpace.transEqDim() > static_cast<Index>(eigVals.size()));
		std::cout << "\tparallelFlag = " << parallelFlag << std::endl;
#pragma omp parallel for schedule(dynamic, 1) if(parallelFlag)
		for(Index p = 0; p != opSpace.transEqDim(); ++p) {
			auto&&     expVals = expValues.col(get_thread_num());
			auto const opNum   = opSpace.transEqClassRep(p);

			expVals = Eigen::ArrayXd::Zero(expVals.size());
#pragma omp parallel if(!parallelFlag)
			{
#pragma omp for schedule(dynamic, 10)
				for(Index innerY = 0; innerY != subSpace.dimTot(); ++innerY) {
					auto [innerX, coeff] = opSpace.action(opNum, innerY);
					auto const outerX    = adjointBasis.innerIndexPtr()[innerX];
					auto const outerY    = adjointBasis.innerIndexPtr()[innerY];
					coeff *= conj(adjointBasis.valuePtr()[innerX]);
					coeff *= adjointBasis.valuePtr()[innerY];

					for(auto j = 0; j != expVals.size(); ++j) {
#pragma omp atomic
						expVals(j) += real(conj(eigVecs(outerX, j)) * coeff * eigVecs(outerY, j));
					}
				}

				double const coeff = 1.0 * opSpace.transPeriod(p);
#pragma omp for schedule(dynamic, 10)
				for(auto j = 0; j != eigVals.size(); ++j) {
					long idMin, idMax;
					for(idMin = j; idMin >= 0; --idMin) {
						if(eigVals(j) - eigVals(idMin) > shWidth) break;
					}
					++idMin;
					for(idMax = j; idMax < eigVals.size(); ++idMax) {
						if(eigVals(idMax) - eigVals(j) > shWidth) break;
					}
					--idMax;

					double MCave = 0.0;
					for(int k = idMin; k != idMax + 1; ++k) MCave += expVals(k);
					MCave /= static_cast<double>(idMax - idMin + 1);
					auto const diff = coeff * (expVals(j) - MCave) * (expVals(j) - MCave);
#pragma omp atomic
					res(j) += diff;
				}
			}
		}
		return res;
	}
}  // namespace StatMech

#ifdef __NVCC__
	#include "quasiETHmeasure_Spin.cuh"
#endif