/**
 * @file quasiETHmeasure_mBody.cpp
 * @author Shoki Sugimoto (sugimoto@cat.phys.u-tokyo.ac.jp)
 * @brief
 * @version 0.1
 * @date 2023-08-27
 *
 * @copyright Copyright (c) 2023
 *
 */

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
	Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", ";\n", " ", "", "[", ";]")
#define EIGEN_DONT_PARALLELIZE

#include "constructGlobalOp.hpp"
#include "dataHandler.hpp"
#include <HilbertSpace>
#include <StatMech>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <sys/time.h>
#include <filesystem>
namespace fs = std::filesystem;

using Scalar = cuda::std::complex<double>;

int sampleNo;
int ell, k;

template<>
fs::path IOManager::generate_filepath(Index sysSize, Index N) const {
	return this->rootDir()
	       / ("mBodyETH/PBC_TI/ShortRange_Fermion_" + std::to_string(ell) + "local_"
	          + std::to_string(k) + "body")
	       / ("Sample_No" + std::to_string(sampleNo))
	       / ("SystemSize_L" + std::to_string(sysSize) + "_N" + std::to_string(N))
	       / ("mBody_quasiETHmeasureSq_dE" + this->shWith() + std::string(".txt"));
}

int main(int argc, char** argv) {
	if(argc != 11) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N) 3.(MMax) 4.(MMin) "
		             "5.(ell) 6.(k) 7.(sampleMin) 8.(sampleMax) "
		             "9.(shellWidthParam) 10.(OutDir)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	GPU::MAGMA::get_controller();

	constexpr int parity              = +1;
	Index const   L                   = std::atoi(argv[1]);
	Index const   N                   = std::atoi(argv[2]);
	Index const   MMax                = std::atoi(argv[3]);
	Index const   MMin                = std::atoi(argv[4]);
	ell                               = std::atoi(argv[5]);
	k                                 = std::atoi(argv[6]);
	Index const       sampleMin       = std::atoi(argv[7]);
	Index const       sampleMax       = std::atoi(argv[8]);
	double const      shellWidthParam = std::atof(argv[9]);
	std::string const shellWidthStr(argv[9]);
	IOManager         data_writer(argv[10]);
	std::cout << "rootDir = " << data_writer.rootDir() << std::endl;
	{
		// std::stringstream buff("");
		// buff << "#Jx = " << Jx << "\n"
		//      << "#Bz = " << Bz << "\n"
		//      << "#Bx = " << Bx << "\n";
		// data_writer.custom_header() = buff.str();
		data_writer.shWith() = shellWidthStr;
	}
	using StateSpace = ManyBodyFermionSpace;

	// Two-local operator without parity symmetry
	Combination const                      locOpConfig(ell, k);
	int const                              locOpDim = locOpConfig.dim() * locOpConfig.dim();
	mBodyOpSpace<StateSpace, Scalar> const globOpSpace(k, L, N);
	Eigen::ArrayX<Scalar>                  coeff;
	constexpr int                          seed = 0;
	std::mt19937                           mt(seed);
	std::normal_distribution<double>       Gaussian(0.0, 1.0);

	for(sampleNo = 0; sampleNo < sampleMin; ++sampleNo) {
		coeff = Eigen::ArrayX<Scalar>::NullaryExpr(
		    locOpDim, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
	}
	for(sampleNo = sampleMin; sampleNo <= sampleMax; ++sampleNo) {
		coeff = Eigen::ArrayX<Scalar>::NullaryExpr(
		    locOpDim, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });

		double                                      temp_t;
		double                                      T_pre = 0, T_diag = 0, T_meas = 0, T_IO = 0;
		StateSpace const                            mbSpace(L, N);
		TransParitySector<StateSpace, Scalar> const sector(parity, mbSpace);

		temp_t                   = omp_get_wtime();
		Eigen::MatrixX<Scalar> H = construct_globalOp(coeff, locOpConfig, globOpSpace, sector);
		T_pre += omp_get_wtime() - temp_t;

		temp_t = omp_get_wtime();
		GPU::SelfAdjointEigenSolver_mgpu<std::decay_t<decltype(H)>> const solver(
		    GPU::MAGMA::ngpus(), std::move(H));
		T_diag += omp_get_wtime() - temp_t;
		std::cout << "# (L, N) = (" << L << ", " << N
		          << "): Diagonalized the Hamiltonian.\t Elapsed = " << T_diag << " (sec)"
		          << std::endl;

		auto const   eigRange  = solver.eigenvalues().maxCoeff() - solver.eigenvalues().minCoeff();
		double const shWidth   = eigRange * shellWidthParam / L;
		auto const   shellDims = get_shellDims(solver.eigenvalues(), shWidth, solver.eigenvalues());
		double const lsRatio   = LevelSpacingRatio(solver.eigenvalues());
		{
			std::stringstream buff("");
			buff << "#coeff = " << coeff.cast<std::complex<double>>().transpose() << "\n"
			     << "#parity = " << parity << "\n"
			     << "#LevelSpacingRatio = " << lsRatio << "\n";
			data_writer.custom_header() = buff.str();
		}

		Eigen::ArrayXXd ETHmeasure
		    = Eigen::ArrayXXd::Constant(solver.eigenvectors().cols(), N + 1, std::nan(""));
		temp_t = omp_get_wtime();
		for(auto m = MMin; m <= std::min(MMax, N); ++m) {
			mBodyOpSpace<StateSpace, Scalar> const opSpace(m, L, N);

			std::cout << "\n# SampleNo." << sampleNo << ", L = " << std::setw(2) << L
			          << ", N = " << std::setw(2) << N << ", m = " << std::setw(2) << m
			          << ", dim = " << sector.dim() << ", dimTot = " << sector.dimTot()
			          << ", opDim = " << opSpace.dim() << std::endl;
			auto                 execTime = omp_get_wtime();
			Eigen::ArrayXd const res = StatMech::ETHmeasure2Sq(solver, opSpace, sector, shWidth);
			execTime                 = omp_get_wtime() - execTime;

			temp_t = omp_get_wtime();
			data_writer.load_Data(ETHmeasure, L, N);
			ETHmeasure.col(m) = res;
			data_writer.save_ResultsToFile(solver.eigenvalues(), ETHmeasure, L, N, shellDims,
			                               std::nan(""));
			T_IO += omp_get_wtime() - temp_t;
			// totExecTime += execTime;
			std::cout << "#\t exec time = " << execTime << " (sec)" << std::endl;
		}
		std::cout << std::endl;
		T_meas += omp_get_wtime() - temp_t;

		double maxDiff = std::nan("");
		{  // Verifying the results
			auto theorySum
			    = Eigen::ArrayXd::Ones(shellDims.size()) - shellDims.cast<double>().inverse();
			maxDiff = (ETHmeasure.col(N).rowwise().sum() - theorySum).cwiseAbs().maxCoeff();
			if(!std::isnan(maxDiff) && maxDiff > precision) {
				std::cout << "ETHmeasure.rowwise().sum():\n"
				          << ETHmeasure.rowwise().sum() << "\n"
				          << std::endl;
				std::cout << "theorySum:\n" << theorySum << "\n" << std::endl;
				std::cout << "ETHmeasure:\n" << ETHmeasure << "\n" << std::endl;
				std::cerr << "Error : maxDiff = " << maxDiff << " is too large." << std::endl;
				continue;
				// std::exit(EXIT_SUCCESS);
			}
		}

		temp_t = omp_get_wtime();
		data_writer.save_ResultsToFile(solver.eigenvalues(), ETHmeasure, L, N, shellDims, maxDiff);
		T_IO += omp_get_wtime() - temp_t;

		std::cout << "# SampleNo." << sampleNo << ", L=" << L << ", N=" << N << " T_pre: " << T_pre
		          << " T_diag: " << T_diag << " T_meas: " << T_meas << " T_IO: " << T_IO
		          << std::endl;
	}

	return EXIT_SUCCESS;
}