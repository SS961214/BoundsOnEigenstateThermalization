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

#include "quasiETHmeasure.hpp"
#include <HilbertSpace>
#include <MatrixGPU>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <iomanip>
#include <omp.h>

using Scalar = cuda::std::complex<double>;

std::random_device               seed_gen;
std::mt19937                     mt(seed_gen());
std::normal_distribution<double> Gaussian(0.0, 1.0);

int main(int argc, char* argv[]) {
	if(argc != 4) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N) 3.(parity)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	GPU::MAGMA::get_controller();  // Initialize MAGMA
	constexpr double precision = 1.0e-12;
	int const        L         = std::atoi(argv[1]);
	int const        N         = std::atoi(argv[2]);
	int const        parity    = std::atoi(argv[3]);
	std::cout << "#(ManyBodyBosonSpace) L = " << L << ", N = " << N << ", parity = " << parity
	          << std::endl;
	ManyBodyBosonSpace                          hSpace(L, N);
	TransParitySector<decltype(hSpace), Scalar> subSpace(parity, hSpace);

	auto const             dim = subSpace.dim();
	Eigen::MatrixX<Scalar> mat = Eigen::MatrixX<Scalar>::NullaryExpr(
	    dim, dim, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
	// Eigen::MatrixXd mat
	//     = Eigen::MatrixXd::NullaryExpr(dim, dim, [&]() { return Gaussian(mt); });
	mat = (mat + mat.adjoint()).eval() / 2.0;
	std::cout << "# Prepared a random matrix" << std::endl;

	GPU::SelfAdjointEigenSolver_mgpu<decltype(mat)> solver(GPU::MAGMA::ngpus(), std::move(mat));
	auto const   energyRange = solver.eigenvalues().maxCoeff() - solver.eigenvalues().minCoeff();
	double const shWidth     = 0.2 * energyRange;
	Eigen::ArrayXi const shellDims
	    = get_shellDims(solver.eigenvalues(), shWidth, solver.eigenvalues());
	Eigen::ArrayXd const theorySum
	    = Eigen::ArrayXd::Ones(dim) - shellDims.template cast<double>().inverse();

	double          totExecTime = 0;
	int const       MMax        = N;
	Eigen::ArrayXXd ETHmeasure  = Eigen::ArrayXXd::Zero(dim, MMax + 1);
	for(auto m = 1; m <= MMax; ++m) {
		mBodyOpSpace<decltype(hSpace), Scalar> hmbOpSpace(m, L, N);

		std::cout << "L = " << std::setw(2) << L << ", N = " << std::setw(2) << N
		          << ", m = " << std::setw(2) << m << ", dim = " << dim
		          << ", dimTot = " << subSpace.dimTot() << ", opDim = " << hmbOpSpace.dim()
		          << std::endl;
		auto execTime     = omp_get_wtime();
		ETHmeasure.col(m) = StatMech::ETHmeasure2Sq(solver, hmbOpSpace, subSpace, shWidth);
		execTime          = omp_get_wtime() - execTime;
		totExecTime += execTime;
		std::cout << "\t     exec time = " << execTime << " (sec)\n" << std::endl;
	}
	auto const maxDiff = (ETHmeasure.col(MMax) - theorySum).cwiseAbs().maxCoeff();
	if(maxDiff > precision) {
		std::cout << "#\t ETHmeasure:\n";
		std::cout << ETHmeasure.transpose() << std::endl;
		std::cout << "#\t theorySum:\n";
		std::cout << theorySum.transpose() << std::endl;
	}
	std::cout << "\t       maxDiff = " << maxDiff << ",\t(total exec time) = " << totExecTime
	          << " (sec)\n"
	          << std::endl;

	return EXIT_SUCCESS;
}