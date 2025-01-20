#include <catch2/catch_test_macros.hpp>
#include "tests/error.hpp"
#include "tests/generateRandomMatrix.hpp"
#include "MatrixGPU.cuh"
#include <Eigen/Dense>
#include <iostream>

#ifdef FLOAT
using RealScalar = float;
#else
using RealScalar = double;
#endif
using Scalar = std::complex<RealScalar>;

TEST_CASE("MatrixGPU", "test") {
	std::mt19937                         engine(0);
	std::normal_distribution<RealScalar> dist(0.0, 1.0);

	constexpr double       precision = 1.0E-10;
	constexpr int          dim       = 1000;
	Eigen::MatrixX<Scalar> mat(dim, dim);
	GPU::internal::generateRandomMatrix(mat, dim);
	REQUIRE(mat.norm() > precision);
	GPU::MatrixGPU<decltype(mat)> dmat(mat);
	REQUIRE(mat.rows() == dmat.rows());
	REQUIRE(mat.cols() == dmat.cols());

	Eigen::MatrixX<Scalar> res;
	dmat.copyTo(res);
	REQUIRE((mat - res).norm() < precision);
}