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
#include <MatrixGPU>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <iomanip>
#include <sys/time.h>

template<class Array>
__host__ Eigen::ArrayXi get_shellDims(Array const& eigVals, double const shWidth) {
	Eigen::ArrayXi res = Eigen::ArrayXi::Zero(eigVals.size());
#pragma omp parallel for
	for(auto j = 0; j != eigVals.size(); ++j) {
		auto idMin = j, idMax = j;
		for(idMin = j; idMin >= 0; --idMin) {
			if(eigVals(j) - eigVals(idMin) > shWidth) break;
		}
		++idMin;
		for(idMax = j; idMax < eigVals.size(); ++idMax) {
			if(eigVals(idMax) - eigVals(j) > shWidth) break;
		}
		--idMax;
		res(j) = idMax - idMin + 1;
	}
	return res;
}

double getETtime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

using Scalar = cuda::std::complex<double>;

std::random_device               seed_gen;
std::mt19937                     mt(seed_gen());
std::normal_distribution<double> Gaussian(0.0, 1.0);

TEST_CASE("quasiETHmeasure_Spin_onGPU", "test") {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif

	constexpr double precision = 1.0e-12;
	constexpr Index  dLoc      = 2;
	{
		// for translation-invariant Hamiltonian
		// and OpSpace being a space of m-body operators
		constexpr Index LMin = 10;
		constexpr Index LMax = 10;
		for(Index l = LMin; l <= LMax; ++l) {
			ManyBodySpinSpace                     hSpace(l, dLoc);
			TransSector<decltype(hSpace), Scalar> subSpace(0, hSpace);
			auto const                            dim = subSpace.dim();
			std::cout << "\n#l = " << std::setw(2) << l << ", dim = " << dim << std::endl;
			Eigen::MatrixX<Scalar> mat = Eigen::MatrixX<Scalar>::NullaryExpr(
			    dim, dim, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
			mat = (mat + mat.adjoint().eval()) / 2.0;

			// Diagonalize the Hamiltonian to obtain eigenvalue and eigenvectors
			GPU::MatrixGPU<decltype(mat)>               dmat(mat);
			GPU::SelfAdjointEigenSolver<decltype(dmat)> dsolver(dmat);
			auto const energyRange = dsolver.eigenvalues()(dim - 1) - dsolver.eigenvalues()(0);
			double     shWidth     = 0.1 * energyRange;

			// Calculate the theoretical value of the ETH measure
			Eigen::ArrayXi const shellDims = get_shellDims(dsolver.eigenvalues(), shWidth);
			Eigen::ArrayXd const theorySum
			    = Eigen::ArrayXd::Ones(dim) - shellDims.template cast<double>().cwiseInverse();

			double                        totExecTime = 0;
			Eigen::ArrayXXd               ETHmeasure  = Eigen::ArrayXXd::Zero(dim, l + 1);
			thrust::device_vector<double> dEigVals(dsolver.eigenvalues().begin(),
			                                       dsolver.eigenvalues().end());
			{
				double const norm
				    = (dsolver.eigenvectors().adjoint() * dsolver.eigenvectors()
				       - Eigen::MatrixX<Scalar>::Identity(subSpace.dim(), subSpace.dim()))
				          .norm();
				std::cout << "#\t norm = " << norm << std::endl;
			}
			for(Index m = 1; m <= l; ++m) {
				mBodyOpSpace<decltype(hSpace), Scalar> hOpSpace(m, l, dLoc);
				ObjectOnGPU<decltype(hOpSpace)>        dOpSpace(hOpSpace);

				std::cout << "\n# l = " << std::setw(2) << l << ", m = " << std::setw(2) << m
				          << ", dim = " << dim << ", opDim = " << hOpSpace.dim() << std::endl;
				auto execTime = getETtime();
				ETHmeasure.col(m) += StatMech::ETHMeasure2(dEigVals, dsolver.eigenvectorsGPU(),
				                                           hOpSpace, dOpSpace, subSpace, shWidth);
				execTime = getETtime() - execTime;
				totExecTime += execTime;
				std::cout << "\t     exec time = " << execTime << " (sec)" << std::endl;
			}
			// std::cout << ETHmeasure << std::endl;
			// {
			// 	Eigen::ArrayXXd res(dim, 2);
			// 	res.col(0) = ETHmeasure.rowwise().sum();
			// 	res.col(1) = theorySum;
			// 	std::cout << res << std::endl;
			// }
			auto const maxDiff = (ETHmeasure.rowwise().sum() - theorySum).cwiseAbs().maxCoeff();
			std::cout << "\t       maxDiff = " << maxDiff
			          << ",\t(total exec time) = " << totExecTime << " (sec)\n"
			          << std::endl;
			REQUIRE(maxDiff < precision);
		}
	}
}