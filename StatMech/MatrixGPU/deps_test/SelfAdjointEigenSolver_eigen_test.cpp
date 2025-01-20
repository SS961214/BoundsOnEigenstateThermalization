#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <random>
#include <complex>
#include <iostream>

using RealScalar = double;
using Scalar     = std::complex<RealScalar>;

TEST_CASE("SelfAdjointEigenSolver_magma", "test") {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "# EIGEN_USE_MKL_ALL is set." << std::endl;
#else
	std::cout << "# EIGEN_USE_MKL_ALL is NOT set." << std::endl;
#endif
	Eigen::initParallel();
	constexpr double precision = 1.0e-12;
	constexpr int    dim       = 1000;
	constexpr int    Nsample   = 10;

	std::random_device                   seed_gen;
	std::mt19937                         engine(seed_gen());
	std::normal_distribution<RealScalar> dist(0.0, 1.0);
	auto const                           RME = [&](int dim) {
        Eigen::MatrixX<Scalar> mat = Eigen::MatrixX<Scalar>::NullaryExpr(
            dim, dim, [&]() { return Scalar(dist(engine), dist(engine)); });
        mat = (mat + mat.adjoint()).eval();
        mat /= mat.norm();
        return mat;
	};
	// std::cout << "# Sample 1:\n" << RME(dim) << "\n\n" << "# Sample 2:\n" << RME(dim) << std::endl;

	for(auto n = 0; n < Nsample; ++n) {
		Eigen::MatrixX<Scalar> const mat = RME(dim);
		std::cout << "# Prepared a random matrix." << std::endl;
		Eigen::SelfAdjointEigenSolver<std::decay_t<decltype(mat)>> const solver(mat);
		std::cout << "# Diagonalized a random matrix." << std::endl;
		// Eigen::MatrixX<Scalar> const temp = mat.selfadjointView<Eigen::Lower>() * solver.eigenvectors();
		// auto const diff = (mat.selfadjointView<Eigen::Lower>() * solver.eigenvectors() - solver.eigenvectors() * solver.eigenvalues().asDiagonal()).norm();
		auto const diff = (mat * solver.eigenvectors()
		                   - solver.eigenvectors() * solver.eigenvalues().asDiagonal())
		                      .norm();
		REQUIRE(diff < precision);
	}
}