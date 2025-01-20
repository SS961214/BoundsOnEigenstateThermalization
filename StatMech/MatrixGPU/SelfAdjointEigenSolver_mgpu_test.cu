#include <catch2/catch_test_macros.hpp>
#include "tests/error.hpp"
#include "tests/generateRandomMatrix.hpp"
#include "SelfAdjointEigenSolver_mgpu.cuh"
#include <Eigen/Dense>
#include <iostream>

#ifdef FLOAT
using RealScalar = float;
#else
using RealScalar = double;
#endif
using ScalarCPU = std::complex<RealScalar>;

TEST_CASE("MatrixGPU_mgpu", "test") {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	GPU::MAGMA::get_controller();
	int const ngpus = GPU::MAGMA::ngpus();
	std::cout << "# ngpus = " << ngpus << std::endl;

	// constexpr int dim = 14602;  // Dimension of the zero momentum sector for Spin systems with L = 18
	// constexpr int dim = 26214;  // Dimension of the zero-momentum & even-parity sector for Spin systems with L = 20
	constexpr int dim
	    = 52428;  // Dimension of the zero momentum sector for Spin systems with L = 20 (Requires MKL_ILP64 interface)

	std::cout << "# dim = " << dim << std::endl;
	Eigen::MatrixX<ScalarCPU> mat(dim, dim);
	GPU::internal::generateRandomMatrix(mat, dim);

	constexpr double precision = 1.0E-4;
	{
		std::cout << "## Enter point 1" << std::endl;
		double                                          T_diag = GPU::internal::getETtime();
		GPU::SelfAdjointEigenSolver_mgpu<decltype(mat)> hsolver(ngpus, mat);
		T_diag = GPU::internal::getETtime() - T_diag;
		auto const diff
		    = GPU::internal::diagError(mat, hsolver.eigenvectors(), hsolver.eigenvalues());
		REQUIRE(diff < precision);
		std::cout << "## diff = " << diff << std::endl;
		std::cout << "## Passed point 1.\t T_diag = " << T_diag << " (sec)" << std::endl;
	}
	// {
	// 	std::cout << "## Enter point 2" << std::endl;
	// 	GPU::SelfAdjointEigenSolver_mgpu<decltype(mat)> hsolver;
	// 	double                                          T_diag = GPU::internal::getETtime();
	// 	hsolver.compute(ngpus, mat);
	// 	T_diag                            = GPU::internal::getETtime() - T_diag;
	// 	Eigen::MatrixX<ScalarCPU> eigVecs = hsolver.eigenvectors();
	// 	auto const diff = (mat * eigVecs - eigVecs * hsolver.eigenvalues().asDiagonal()).norm();
	// 	REQUIRE(diff < precision);
	// 	std::cout << "## Passed point 2.\t T_diag = " << T_diag << " (sec)" << std::endl;
	// }
}