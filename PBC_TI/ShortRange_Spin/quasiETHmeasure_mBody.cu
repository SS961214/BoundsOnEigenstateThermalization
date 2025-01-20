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
	Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", ";\n", " ", "", "[", ";\n]")
#define EIGEN_DONT_PARALLELIZE

#include "constructGlobalOp.hpp"
#include "dataHandler.hpp"
#include <StatMech/HilbertSpace/OpSpace/mBodyOpSpace_Spin.hpp>
#include <StatMech/quasiETHmeasure.hpp>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <sys/time.h>
#include <filesystem>
namespace fs = std::filesystem;

using Scalar = cuda::std::complex<double>;

static inline double getETtime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int sampleNo;

template<>
fs::path IOManager::generate_filepath(Index sysSize) const {
	return this->rootDir() / "mBodyETH/PBC_TI/ShortRange_Spin" / ("Sample_No" + std::to_string(sampleNo))
	       / ("SystemSize_L" + std::to_string(sysSize))
	       / ("mBody_quasiETHmeasureSq_dE" + this->shWith() + std::string(".txt"));
}

int main(int argc, char** argv) {
	if(argc != 9) {
		std::cerr
		    << "Usage: 0.(This) 1.(LMax) 2.(LMin) 3.(MMax) 4.(MMin) 5.(sampleMin) 6.(sampleMax) "
		       "7.(shellWidthParam) "
		       "8.(OutDir)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	GPU::MAGMA::get_controller();

	constexpr Index dimLoc          = 2;
	Index const     LMax            = std::atoi(argv[1]);
	Index const     LMin            = std::atoi(argv[2]);
	Index const     MMax            = std::atoi(argv[3]);
	Index const     MMin            = std::atoi(argv[4]);
	Index const     sampleMin       = std::atoi(argv[5]);
	Index const     sampleMax       = std::atoi(argv[6]);
	double const    shellWidthParam = std::atof(argv[7]);
	IOManager       data_writer(argv[8]);
	std::cout << "rootDir = " << data_writer.rootDir() << std::endl;
	{
		// std::stringstream buff("");
		// buff << "#Jx = " << Jx << "\n"
		//      << "#Bz = " << Bz << "\n"
		//      << "#Bx = " << Bx << "\n";
		// data_writer.custom_header() = buff.str();
		data_writer.shWith() = std::string(argv[7]);
	}

	// Two-local operator without parity symmetry
	Eigen::MatrixX<Scalar>           locH;
	constexpr int                    seed = 0;
	std::mt19937                     mt(seed);
	std::normal_distribution<double> Gaussian(0.0, 1.0);

	constexpr int momentum = 0;
	for(sampleNo = 0; sampleNo < sampleMin; ++sampleNo) {
		locH = Eigen::MatrixX<Scalar>::NullaryExpr(
		    dimLoc * dimLoc, dimLoc * dimLoc, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
		locH = (locH + locH.adjoint()).eval() / 2.0;
	}
	for(sampleNo = sampleMin; sampleNo <= sampleMax; ++sampleNo) {
		locH = Eigen::MatrixX<Scalar>::NullaryExpr(
		    dimLoc * dimLoc, dimLoc * dimLoc, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
		locH = (locH + locH.adjoint()).eval() / 2.0;

		double temp_t;
		double T_pre = 0, T_diag = 0, T_meas = 0, T_IO = 0;
		for(auto L = LMin; L <= LMax; ++L) {
			ManyBodySpinSpace                       mbSpace(L, dimLoc);
			TransSector< decltype(mbSpace), Scalar> transSector(momentum, mbSpace);

			temp_t = getETtime();
			GPU::MatrixGPU< Eigen::MatrixX<Scalar> > dHtot
			    = construct_globalOp_onGPU(locH, transSector);
			T_pre += getETtime() - temp_t;

			temp_t = getETtime();
			GPU::SelfAdjointEigenSolver<decltype(dHtot)> dSolver(dHtot);
			T_diag += getETtime() - temp_t;

			auto const eigRange
			    = dSolver.eigenvalues().maxCoeff() - dSolver.eigenvalues().minCoeff();
			if(eigRange < 1.0e-10) {
				std::cerr << "Error : Diagonalization failed." << std::endl;
				std::exit(EXIT_FAILURE);
			}
			double const shWidth = eigRange * shellWidthParam / L;
			auto const   shellDim
			    = get_shellDims(dSolver.eigenvalues(), shWidth, dSolver.eigenvalues());
			thrust::device_vector<double> dEigVals(dSolver.eigenvalues().begin(),
			                                       dSolver.eigenvalues().end());

			Eigen::ArrayXXd ETHmeasure
			    = Eigen::ArrayXXd::Constant(dSolver.eigenvectors().cols(), L + 1, std::nan(""));
			temp_t = getETtime();
			for(auto m = MMin; m <= std::min(MMax, L); ++m) {
				mBodyOpSpace<decltype(mbSpace), Scalar> const       hOpSpace(m, L, dimLoc);
				ObjectOnGPU<std::decay_t<decltype(hOpSpace)>> const dOpSpace(hOpSpace);

				std::cout << "\n# SampleNo." << sampleNo << ", L = " << std::setw(2) << L
				          << ", m = " << std::setw(2) << m << ", dim = " << transSector.dim()
				          << ", opDim = " << hOpSpace.dim() << std::endl;
				auto                 execTime = getETtime();
				Eigen::ArrayXd const res      = StatMech::ETHMeasure2(
                    dEigVals, dSolver.eigenvectorsGPU(), hOpSpace, dOpSpace, transSector, shWidth);
				execTime = getETtime() - execTime;

				temp_t = getETtime();
				data_writer.load_Data(ETHmeasure, L);
				ETHmeasure.col(m) = res;
				data_writer.save_ResultsToFile(dSolver.eigenvalues(), ETHmeasure, L, shellDim,
				                               std::nan(""));
				T_IO += getETtime() - temp_t;
				// totExecTime += execTime;
				std::cout << "#\t exec time = " << execTime << " (sec)" << std::endl;
			}
			std::cout << std::endl;
			T_meas += getETtime() - temp_t;

			double maxDiff = std::nan("");
			{  // Verifying the results
				auto theory
				    = Eigen::ArrayXd::Ones(shellDim.size()) - shellDim.cast<double>().inverse();
				Eigen::ArrayXd resSum = Eigen::ArrayXd::Zero(ETHmeasure.rows());
				for(auto m = 1; m < ETHmeasure.cols(); ++m) { resSum += ETHmeasure(Eigen::all, m); }
				maxDiff = (resSum - theory).cwiseAbs().maxCoeff();
				if(!std::isnan(maxDiff) && maxDiff > 1.0e-10) {
					std::cout << "ETHmeasure.rowwise().sum():\n"
					          << ETHmeasure.rowwise().sum() << "\n"
					          << std::endl;
					std::cout << "theory:\n" << theory << "\n" << std::endl;
					std::cout << "ETHmeasure:\n" << ETHmeasure << "\n" << std::endl;
					std::cerr << "Error : maxDiff = " << maxDiff << " is too large." << std::endl;
					continue;
					// std::exit(EXIT_SUCCESS);
				}
			}

			temp_t = getETtime();
			data_writer.save_ResultsToFile(dSolver.eigenvalues(), ETHmeasure, L, shellDim, maxDiff);
			T_IO += getETtime() - temp_t;

			std::cout << "# SampleNo." << sampleNo << ", L=" << L << " T_pre: " << T_pre
			          << " T_diag: " << T_diag << " T_meas: " << T_meas << " T_IO: " << T_IO << "\n"
			          << std::endl;
		}
	}

	return EXIT_SUCCESS;
}