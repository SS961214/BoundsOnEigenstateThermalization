#include <catch2/catch_test_macros.hpp>
#include "tests/error.hpp"
#include "tests/generateRandomMatrix.hpp"
#include "SelfAdjointEigenSolverGPU.cuh"
#include <Eigen/Dense>
#include <iostream>

#ifdef FLOAT
using RealScalar = float;
#else
using RealScalar = double;
#endif
// using Scalar = std::complex<RealScalar>; // Passes the test for std::complex<RealScalar> after ~400 sec
using Scalar    = cuda::std::complex<RealScalar>;  //
using ScalarCPU = std::complex<RealScalar>;

TEST_CASE("MatrixGPU", "test") {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	GPU::MAGMA::get_controller();

	// constexpr int dim
	//     = 14602;  // Dimension of the zero momentum sector for Spin systems with L = 18
	constexpr int             dim = 1000;
	Eigen::MatrixX<ScalarCPU> mat(dim, dim);
	GPU::internal::generateRandomMatrix(mat, dim);
	std::cout << "## Preparing a matrix on GPU." << std::endl;
	GPU::MatrixGPU<decltype(mat)> dmat(mat);
	std::cout << "## Prepared a matrix on GPU." << std::endl;
	REQUIRE(dmat.rows() == dim);
	REQUIRE(dmat.cols() == dim);

	constexpr double precision = 1.0E-4;
	{
		std::cout << "## Enter point 1" << std::endl;
		GPU::SelfAdjointEigenSolver<decltype(mat)> hsolver(mat);
		auto const                                 diff
		    = GPU::internal::diagError(mat, hsolver.eigenvectors(), hsolver.eigenvalues());
		REQUIRE(diff < precision);
		std::cout << "## diff = " << diff << std::endl;
		std::cout << "## Passed point 1" << std::endl;
	}
	{
		std::cout << "## Enter point 2" << std::endl;
		GPU::SelfAdjointEigenSolver<decltype(mat)> hsolver;
		hsolver.compute(mat);
		auto const diff
		    = GPU::internal::diagError(mat, hsolver.eigenvectors(), hsolver.eigenvalues());
		REQUIRE(diff < precision);
		std::cout << "## diff = " << diff << std::endl;
		std::cout << "## Passed point 2" << std::endl;
	}
	{
		std::cout << "## Enter point 3" << std::endl;
		GPU::SelfAdjointEigenSolver<decltype(dmat)> dsolver(dmat);
		auto const                                  diff
		    = GPU::internal::diagError(mat, dsolver.eigenvectors(), dsolver.eigenvalues());
		REQUIRE(diff < precision);
		std::cout << "## diff = " << diff << std::endl;
		std::cout << "## Passed point 3" << std::endl;
	}
	{
		std::cout << "## Enter point 4" << std::endl;
		GPU::SelfAdjointEigenSolver<decltype(dmat)> dsolver;
		dsolver.compute(dmat);
		auto const diff
		    = GPU::internal::diagError(mat, dsolver.eigenvectors(), dsolver.eigenvalues());
		REQUIRE(diff < precision);
		std::cout << "## diff = " << diff << std::endl;
		std::cout << "## Passed point 4" << std::endl;
	}
}