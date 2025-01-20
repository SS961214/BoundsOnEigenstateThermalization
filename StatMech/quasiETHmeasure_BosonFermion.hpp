#pragma once

#include "../Microcanonical.hpp"
#include <HilbertSpace>
#include <iostream>
#include <algorithm>
#include <omp.h>

namespace StatMech {
	template<class OpSpace_>
	struct IsParticleOpSpace : std::false_type {};
	template<typename Scalar_>
	struct IsParticleOpSpace<mBodyOpSpace<ManyBodyBosonSpace, Scalar_>> : std::true_type {};
	template<typename Scalar_>
	struct IsParticleOpSpace<mBodyOpSpace<ManyBodyFermionSpace, Scalar_>> : std::true_type {};

	template< class Matrix_, class Array_, class OpSpace_, class SubSpace_,
	          std::enable_if_t<IsParticleOpSpace<OpSpace_>::value>* = nullptr >
	__host__ Eigen::ArrayXd ETHmeasure2Sq(Matrix_ const& eigVecs, Array_ const& eigVals,
	                                      OpSpace_ const& opSpace, SubSpace_ const& subSpace,
	                                      double const shWidth) {
		using Scalar     = typename Matrix_::Scalar;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		using StateSpace = typename OpSpace_::BaseSpace;
		static_assert(std::is_same_v<RealScalar, typename Array_::Scalar>);
		static_assert(std::is_same_v<StateSpace, ManyBodyBosonSpace>
		              || std::is_same_v<StateSpace, ManyBodyFermionSpace>);
		static_assert(std::is_same_v<SubSpace_, TransSector<StateSpace, Scalar>>
		              || std::is_same_v<SubSpace_, TransParitySector<StateSpace, Scalar>>);

		std::cout << "# " << __PRETTY_FUNCTION__ << std::endl;
		assert(eigVecs.rows() == eigVals.size());
		assert(eigVecs.rows() == static_cast<long>(subSpace.dim()));
		assert(opSpace.baseDim() == subSpace.dimTot());
		// Index opDim = 0;

		auto const     blocks  = blocksInGramMat(opSpace);
		Eigen::ArrayXi indices = Eigen::ArrayXi::LinSpaced(blocks.size() - 1, 1, blocks.size() - 1);
		// std::cout << indices << std::end;
		std::sort(indices.begin(), indices.end(),
		          [&blocks](int i, int j) { return blocks[i][0].size() > blocks[j][0].size(); });
		// for(Index c = 0; c < indices.size(); ++c) {
		// 	int b = indices[c];
		// 	std::cout << "#\t Block " << b << " has " << blocks[b][0].size() << " elements." << std::endl;
		// 	// for(Index j = 0; j < blocks[b].size(); ++j) { std::cout << blocks[b][j].size() << " "; }
		// 	// std::cout << std::endl;
		// }

		Eigen::ArrayXd res = Eigen::ArrayXd::Zero(eigVals.size());
		Eigen::ArrayXd sum = Eigen::ArrayXd::Zero(eigVals.size());

		auto const eigVecsFull = (subSpace.basis() * eigVecs).eval();
		// std::cout << "#\t eigVecs:\n" << (eigVecsFull.adjoint() * eigVecsFull) << std::endl;

		// omp_set_max_active_levels(2);
		double const start    = omp_get_wtime();
		int          finished = 1;
// #pragma omp parallel for schedule(dynamic, 1) reduction(+ : res)
#pragma omp parallel
#pragma omp single
		{
#pragma omp taskgroup task_reduction(+ : res, sum)
			{
#pragma omp taskloop priority(10) grainsize(1) in_reduction(+ : res)
				for(Index c = 0; c < Index(indices.size()); ++c) {
					Index const b = indices[c];
					// opDim += blocks[b][0].size() * blocks[b].size();
					Eigen::MatrixX<Scalar> expval(blocks[b][0].size(), eigVals.size());

					// (1) Calculate the Gram matrix and eigenstate expectation values (EEVs)
					Eigen::MatrixX<Scalar> gramMat(blocks[b][0].size(), blocks[b][0].size());
					for(Index idx1 = 0; idx1 < gramMat.cols(); ++idx1) {
						auto const  j          = blocks[b][0][idx1];
						auto const& adBasisOp1 = opSpace.basisOp(j).adjoint().eval();
						gramMat(idx1, idx1)    = adBasisOp1.squaredNorm();

						for(Index idx2 = 0; idx2 < idx1; ++idx2) {
							auto const k        = blocks[b][0][idx2];
							auto const basisOp2 = opSpace.basisOp(k).eval();

							gramMat(idx1, idx2)
							    = std::real((adBasisOp1 * basisOp2).eval().diagonal().sum());
							gramMat(idx2, idx1) = gramMat(idx1, idx2);
						}

						// auto const temp = (adBasisOp1 * eigVecsFull).eval();
						for(Index alpha = 0; alpha < eigVals.size(); ++alpha) {
							expval(idx1, alpha) = (eigVecsFull.col(alpha).adjoint() * adBasisOp1
							                       * eigVecsFull.col(alpha))(0);
						}
					}

					// std::cout << "#\t gramMat:" << std::endl;
					// std::cout << gramMat << std::endl;

					// std::cout << "#\t EEV:" << std::endl;
					// std::cout << expval << std::endl;

					// std::cout << "#\t MCAverages:" << std::endl;
					// std::cout << MCAverages(eigVals, shWidth, eigVals, expval.adjoint()).adjoint()
					//           << std::endl;

					// (2) Calculate the difference between EEVs and MCAs.
					expval -= MCAverages(eigVals, shWidth, eigVals, expval.adjoint()).adjoint();

					// std::cout << "#\t expval:" << std::endl;
					// std::cout << expval << std::endl;

					auto const temp2 = gramMat.llt().solve(expval).eval();
					// std::cout << "#\t gramMat * temp2:" << std::endl;
					// std::cout << gramMat * temp2 << std::endl;

					// std::cout << "#\t temp2:" << std::endl;
					// std::cout << temp2 << std::endl;

					// std::cout << "#\t res:" << std::endl;
					// std::cout << res << std::endl;
					for(Index alpha = 0; alpha < eigVals.size(); ++alpha) {
						res(alpha) += blocks[b].size()
						              * (expval.col(alpha).adjoint() * temp2.col(alpha)).real()(0);
					}
					// std::cout << "#\t res:" << std::endl;
					// std::cout << res << std::endl;
					// if(b % 10 == 0) std::cout << "#\t b = " << b << std::endl;
					double p = 0;
					int    num;
#pragma omp atomic capture
					{
						++finished;
						num = finished;
					}
					p = num / double(blocks.size()) * 100;
					if(std::abs(p - int(p / 5) * 5.0) < 100.0 / blocks.size()) {
#pragma omp critical
						std::cout << "#\t Progress: " << int(p / 5) * 5
						          << "%\t Elapsed: " << omp_get_wtime() - start << " (sec)"
						          << std::endl;
					}
				}

// #pragma omp parallel for schedule(dynamic, 10) reduction(+ : sum)
// #pragma omp single
#pragma omp taskloop priority(0) grainsize(10) in_reduction(+ : sum)
				for(auto c = 0; c < Index(blocks[0].size()); ++c) {
					// opDim += blocks[0][c].size();
					auto const j       = blocks[0][c][0];
					auto const basisOp = opSpace.basisOp(j).eval();

					Eigen::ArrayX<Scalar> expval(eigVals.size());
					for(Index alpha = 0; alpha < eigVals.size(); ++alpha) {
						expval(alpha) = (eigVecsFull.col(alpha).adjoint() * basisOp.adjoint()
						                 * eigVecsFull.col(alpha))(0);
					}
					auto const MCAve = MCAverages(eigVals, shWidth, eigVals, expval);
					expval -= MCAve.array();

					sum += (double(blocks[0][c].size()) / basisOp.squaredNorm()) * expval.abs2();

					int p = c / double(blocks[0].size()) * 100;
					if(std::abs(p - int(p / 5) * 5.0) < 100.0 / blocks[0].size()) {
#pragma omp critical
						std::cout << "#\t Progress 2: " << int(p / 5) * 5
						          << "%\t Elapsed: " << omp_get_wtime() - start << " (sec)"
						          << std::endl;
					}
				}
			}
		}
		// assert(opSpace.dim() == opDim);

		res += sum;

		return res;
	}

}  // namespace StatMech

#ifdef __NVCC__
	#include "quasiETHmeasure_BosonFermion.cuh"
#endif