#pragma once

#include "tests.hpp"
#include "OpSpaceBase.hpp"
#include "ManyBodyHilbertSpace/ManyBodySpinSpace.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <random>

template<class Derived>
void test_OpSpace(Derived const& opSpace) {
	static_assert(std::is_convertible_v<Derived, OpSpaceBase<Derived>>);
	std::cout << "opSpace.dim() = " << opSpace.dim() << std::endl;
	constexpr double precision    = 1.0e-12;
	auto const       innerProduct = [&](Index j, Index k) {
        return (opSpace.basisOp(j).adjoint() * opSpace.basisOp(k)).eval().diagonal().sum();
	};
	using Scalar = typename Derived::Scalar;

	// 	// Gram matrix
	// 	Eigen::MatrixX<Scalar> gramMat(opSpace.dim(), opSpace.dim());
	// #pragma omp parallel for schedule(dynamic, 10)
	// 	for(Index j = 0; j < opSpace.dim(); ++j)
	// 		for(Index k = 0; k <= j; ++k) {
	// 			gramMat(j, k) = innerProduct(j, k);
	// 			gramMat(k, j) = std::conj(gramMat(j, k));
	// 		}
	// 	std::cout << "# Gram matrix" << std::endl;
	// 	std::cout << gramMat << std::endl;
	// 	if(opSpace.dim() > 0) {
	// 		Eigen::SelfAdjointEigenSolver<decltype(gramMat)> solver(gramMat, Eigen::EigenvaluesOnly);
	// 		std::cout << solver.eigenvalues() << std::endl;
	// 	}

	constexpr Index                      nSample = 1000;
	std::random_device                   seed_gen;
	std::default_random_engine           engine(seed_gen());
	std::uniform_int_distribution<Index> dist(0, opSpace.dim() - 1);
	Eigen::ArrayX<Index>                 index;
	if(nSample > opSpace.dim() * (opSpace.dim() + 1)) {
		index.resize(opSpace.dim() * (opSpace.dim() + 1));
		Index id = 0;
		for(Index j = 0; j != opSpace.dim(); ++j)
			for(Index k = j; k != opSpace.dim(); ++k) {
				index(id++) = j;
				index(id++) = k;
			}
		REQUIRE(id == opSpace.dim() * (opSpace.dim() + 1));
	}
	else {
		index = index.NullaryExpr(2 * nSample, [&]() { return dist(engine); });
	}

#pragma omp parallel for
	for(auto sample = 0; sample < index.size() / 2; ++sample) {
		auto j = index(2 * sample);
		auto k = index(2 * sample + 1);
		if(j != k) {
			Scalar offDiag = innerProduct(j, k);
			if constexpr(std::is_same_v<typename Derived::BaseSpace, ManyBodySpinSpace>)
				REQUIRE(abs(offDiag) < precision);
			if(abs(offDiag) >= precision)
				std::cout << "## Non-zero inner product at (j,k) = (" << j << ", " << k
				          << ") : offDiag = " << offDiag << std::endl;
		}
	}
}

#ifdef __NVCC__
	#include "OpSpaceBase_test.cuh"
#endif