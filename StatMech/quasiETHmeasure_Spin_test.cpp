#if __has_include(<mkl.h>)
	#ifndef MKL
		#define MKL
	#endif
	#ifndef EIGEN_USE_MKL_ALL
		#define EIGEN_USE_MKL_ALL
	#endif
#else
	#if __has_include(<Accelerate/Accelerate.h>)
		#ifndef ACCELERATE
			#define ACCELERATE
		#endif
	#endif
#endif

#define EIGEN_DEFAULT_IO_FORMAT \
	Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", ";\n", " ", "", "[", ";\n]")
// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE
// #define EIGEN_DONT_PARALLELIZE

#include <catch2/catch_test_macros.hpp>
#include "quasiETHmeasure.hpp"
#include <HilbertSpace>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <iomanip>
#include <sys/time.h>

double getETtime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

using Scalar = std::complex<double>;

std::random_device               seed_gen;
std::mt19937                     mt(seed_gen());
std::normal_distribution<double> Gaussian(0.0, 1.0);
static inline Index              powi(Index base, Index expo) {
    Index res = 1;
    for(Index j = 0; j < expo; ++j) res *= base;
    return res;
}

template<class OpSpace, class SubSpace_>
__host__ double compute_maxDiff_forRandomMatrices(OpSpace const&   opSpace,
                                                  SubSpace_ const& subSpace) {
	auto const       dim = subSpace.dim();
	Eigen::MatrixXcd mat = Eigen::MatrixXcd::NullaryExpr(
	    dim, dim, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
	mat = (mat + mat.adjoint().eval()) / 2.0;

	Eigen::SelfAdjointEigenSolver< decltype(mat) > eigSolver(mat);

	auto const energyRange = eigSolver.eigenvalues()(dim - 1) - eigSolver.eigenvalues()(0);
	double     shWidth     = 0.1 * energyRange;

	Eigen::ArrayXi const shellDims
	    = get_shellDims(eigSolver.eigenvalues(), shWidth, eigSolver.eigenvalues());
	Eigen::ArrayXd const theorySum
	    = Eigen::ArrayXd::Ones(dim) - shellDims.template cast<double>().cwiseInverse();

	auto execTime   = getETtime();
	auto ETHmeasure = StatMech::ETHmeasure2Sq(eigSolver.eigenvectors(), eigSolver.eigenvalues(),
	                                          opSpace, subSpace, shWidth);
	execTime        = getETtime() - execTime;
	std::cout << "\t exec time = " << execTime << " (sec)" << std::endl;
	return (ETHmeasure - theorySum).cwiseAbs().maxCoeff();
}

TEST_CASE("quasiETHmeasure", "test") {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif

	constexpr double precision = 1.0e-12;
	constexpr Index  dLoc      = 2;
	{
		// for OpSpace without translation invariance
		constexpr Index LMin = 1;
		constexpr Index LMax = 10;
		for(Index l = LMin; l <= LMax; ++l) {
			Index const                        dim = powi(dLoc, l) / l;
			HilbertSpace<int>                  hSpace(dim);
			OpSpace<Scalar>                    opSpace(hSpace);
			SubSpace<decltype(hSpace), Scalar> subSpace(hSpace);
			subSpace.basis().resize(dim, dim);
			subSpace.basis().setIdentity();

			std::cout << "l = " << std::setw(2) << l << ",     dim = " << dim
			          << ", opDim = " << opSpace.dim() << std::endl;
			auto const maxDiff = compute_maxDiff_forRandomMatrices(opSpace, subSpace);
			std::cout << "\t   maxDiff = " << maxDiff << "\n" << std::endl;
			REQUIRE(maxDiff < precision);
		}
	}

	{
		// for translation-invariant Hamiltonian
		// and OpSpace being a space of m-body operators
		constexpr Index LMin = 8;
		constexpr Index LMax = 15;
		for(Index l = LMin; l <= LMax; ++l) {
			ManyBodySpinSpace                     hSpace(l, dLoc);
			TransSector<decltype(hSpace), Scalar> subSpace(0, hSpace);

			auto const       dim = subSpace.dim();
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::NullaryExpr(
			    dim, dim, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
			mat = (mat + mat.adjoint().eval()) / 2.0;

			Eigen::SelfAdjointEigenSolver< decltype(mat) > eigSolver(mat);
			auto const energyRange = eigSolver.eigenvalues()(dim - 1) - eigSolver.eigenvalues()(0);
			double     shWidth     = 0.1 * energyRange;
			Eigen::ArrayXi const shellDims
			    = get_shellDims(eigSolver.eigenvalues(), shWidth, eigSolver.eigenvalues());
			Eigen::ArrayXd const theorySum
			    = Eigen::ArrayXd::Ones(dim) - shellDims.template cast<double>().cwiseInverse();

			double         totExecTime = 0;
			Eigen::ArrayXd ETHmeasure  = Eigen::ArrayXd::Zero(dim);
			for(Index m = 1; m <= l; ++m) {
				mBodyOpSpace<decltype(hSpace), Scalar> opSpace(m, l, dLoc);
				opSpace.compute_transEqClass();

				std::cout << "l = " << std::setw(2) << l << ", m = " << std::setw(2) << m
				          << ", dim = " << dim << ", opDim = " << opSpace.dim() << std::endl;
				auto execTime = getETtime();
				ETHmeasure += StatMech::ETHmeasure2Sq(
				    eigSolver.eigenvectors(), eigSolver.eigenvalues(), opSpace, subSpace, shWidth);
				execTime = getETtime() - execTime;
				totExecTime += execTime;
				std::cout << "\t     exec time = " << execTime << " (sec)" << std::endl;
			}
			auto const maxDiff = (ETHmeasure - theorySum).cwiseAbs().maxCoeff();
			std::cout << "\t       maxDiff = " << maxDiff
			          << ",\t(total exec time) = " << totExecTime << " (sec)\n"
			          << std::endl;
			REQUIRE(maxDiff < precision);
		}
	}
}