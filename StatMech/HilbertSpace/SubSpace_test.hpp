#pragma once

#include "tests.hpp"
#include "SubSpace.hpp"
#include <iostream>

template<class TotalSpace_, typename Scalar>
void test_SubSpace(SubSpace<TotalSpace_, Scalar> const& subSpace) {
	constexpr double epsilon = 1.0E-14;
	std::cout << "# " << __func__ << std::endl;
	std::cout << "##\t subSpace.dim() = " << subSpace.dim() << std::endl;

	auto const& basis   = subSpace.basis();
	auto const  adjoint = Eigen::SparseMatrix<Scalar>(subSpace.basis().adjoint());
	assert(adjoint.isCompressed());
	if(subSpace.dim() >= 1) {
#pragma omp parallel
		{
			Eigen::ArrayX<Scalar> coeff;
#pragma omp for schedule(dynamic, 10)
			for(auto j = 0; j < subSpace.dim(); ++j) {
				coeff    = Eigen::ArrayX<Scalar>::Zero(subSpace.dim());
				coeff(j) = -1.0;
				for(auto posJ = basis.outerIndexPtr()[j]; posJ < basis.outerIndexPtr()[j + 1];
				    ++posJ) {
					auto const state = basis.innerIndexPtr()[posJ];
					for(auto posK = adjoint.outerIndexPtr()[state];
					    posK < adjoint.outerIndexPtr()[state + 1]; ++posK) {
						auto const k = adjoint.innerIndexPtr()[posK];
						if(!(0 <= k && k < coeff.size())) {
#pragma omp critical
							std::cout << "##\t k = " << k << ", subSpace.dim() = " << subSpace.dim()
							          << ", state = " << state
							          << ", subSpace.dimTot() = " << subSpace.dimTot() << std::endl;
						}
						coeff(k) += adjoint.valuePtr()[posK] * basis.valuePtr()[posJ];
					}
				}
				REQUIRE(coeff.abs().maxCoeff() < epsilon);
			}
		}
	}
	std::cout << "# Passed " << __func__ << "\n#" << std::endl;
}