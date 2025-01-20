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
#include <iostream>
#include <sys/time.h>
#include <filesystem>
namespace fs = std::filesystem;

using Scalar = std::complex<double>;

template<class Array>
static inline Eigen::ArrayXi get_shellDims(Array const& eigVals, double const shWidth) {
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

static inline double getETtime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

double Jx;
double Bx;
double Bz;

template<>
fs::path IOManager::generate_filepath(Index sysSize) const {
	std::stringstream buff("");
	buff << std::setprecision(2) << std::defaultfloat << std::showpos;
	buff << "_Jx" << Jx << "_Bz" << Bz << "_Bx" << Bx;

	return this->rootDir() / "mBodyETH/PBC_TI" / ("Ising" + buff.str())
	       / ("SystemSize_L" + std::to_string(sysSize))
	       / ("mBody_quasiETHmeasureSq_dE" + this->shWith() + std::string(".txt"));
}

int main(int argc, char** argv) {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	Eigen::initParallel();

	if(argc != 10) {
		std::cerr << "Usage: 0.(This) 1.(LMax) 2.(LMin) 3.(MMax) 4.(MMin) 5.(Jx) 6.(Bz) 7.(Bx) "
		             "8.(shellWidthParam) "
		             "9.(OutDir)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	constexpr Index dimLoc       = 2;
	Index const     LMax         = std::atoi(argv[1]);
	Index const     LMin         = std::atoi(argv[2]);
	Index const     MMax         = std::atoi(argv[3]);
	Index const     MMin         = std::atoi(argv[4]);
	Jx                           = std::atof(argv[5]);
	Bz                           = std::atof(argv[6]);
	Bx                           = std::atof(argv[7]);
	double const shellWidthParam = std::atof(argv[8]);
	IOManager    data_writer(argv[9]);
	std::cout << "rootDir = " << data_writer.rootDir() << std::endl;
	{
		std::stringstream buff("");
		buff << "#Jx = " << Jx << "\n"
		     << "#Bz = " << Bz << "\n"
		     << "#Bx = " << Bx << "\n";
		data_writer.custom_header() = buff.str();
		data_writer.shWith()        = std::string(argv[8]);
	}

	// h_j = Jx \sigma^{x}_{j} \sigma^{x}_{j+1} + Bz sigma^{z}_{j} + Bx sigma^{x}_{j}
	Eigen::MatrixX<Scalar> locH = Eigen::MatrixXd::Zero(dimLoc * dimLoc, dimLoc * dimLoc);
	locH(0, 0)                  = Bz;
	locH(0, 1)                  = 0;
	locH(0, 2)                  = Bx;
	locH(0, 3)                  = Jx;
	locH(1, 0)                  = 0;
	locH(1, 1)                  = Bz;
	locH(1, 2)                  = Jx;
	locH(1, 3)                  = Bx;
	locH(2, 0)                  = Bx;
	locH(2, 1)                  = Jx;
	locH(2, 2)                  = -Bz;
	locH(2, 3)                  = 0;
	locH(3, 0)                  = Jx;
	locH(3, 1)                  = Bx;
	locH(3, 2)                  = 0;
	locH(3, 3)                  = -Bz;

	constexpr int momentum = 0;

	double temp_t;
	double T_pre = 0, T_diag = 0, T_meas = 0, T_IO = 0;
	for(Index L = LMin; L <= LMax; ++L) {
		ManyBodySpinSpace                       mbSpace(L, dimLoc);
		TransSector< decltype(mbSpace), Scalar> transSector(momentum, mbSpace);

		temp_t                      = getETtime();
		Eigen::MatrixX<Scalar> Htot = construct_globalOp(locH, transSector);
		T_pre += getETtime() - temp_t;

		temp_t = getETtime();
		Eigen::SelfAdjointEigenSolver< decltype(Htot) > eigSolver(std::move(Htot));
		T_diag += getETtime() - temp_t;

		auto const eigRange
		    = eigSolver.eigenvalues().maxCoeff() - eigSolver.eigenvalues().minCoeff();
		double const shWidth  = eigRange * shellWidthParam / L;
		auto const   shellDim = get_shellDims(eigSolver.eigenvalues(), shWidth);

		Eigen::ArrayXXd ETHmeasure
		    = Eigen::ArrayXXd::Constant(eigSolver.eigenvectors().cols(), L + 1, std::nan(""));
		temp_t = getETtime();
		for(Index m = MMin; m <= std::min(MMax, L); ++m) {
			mBodyOpSpace<decltype(mbSpace), Scalar> const opSpace(m, L, dimLoc);

			std::cout << "\n# L = " << std::setw(2) << L << ", m = " << std::setw(2) << m
			          << ", dim = " << transSector.dim() << ", opDim = " << opSpace.dim()
			          << std::endl;

			auto                 execTime = getETtime();
			Eigen::ArrayXd const res      = StatMech::ETHmeasure2Sq(
                eigSolver.eigenvectors(), eigSolver.eigenvalues(), opSpace, transSector, shWidth);
			execTime = getETtime() - execTime;

			temp_t = getETtime();
			data_writer.load_Data(ETHmeasure, L);
			ETHmeasure.col(m) = res;
			data_writer.save_ResultsToFile(eigSolver.eigenvalues(), ETHmeasure, L, shellDim,
			                               std::nan(""));
			T_IO += getETtime() - temp_t;
			// totExecTime += execTime;
			std::cout << "#\t exec time = " << execTime << " (sec)" << std::endl;
		}
		std::cout << std::endl;
		T_meas += getETtime() - temp_t;

		double maxDiff = std::nan("");
		{  // Verifying the results
			auto theory = Eigen::ArrayXd::Ones(shellDim.size()) - shellDim.cast<double>().inverse();
			maxDiff = (ETHmeasure(Eigen::all, Eigen::seq(1, Eigen::last)).rowwise().sum() - theory)
			              .cwiseAbs()
			              .maxCoeff();
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
		data_writer.save_ResultsToFile(eigSolver.eigenvalues(), ETHmeasure, L, shellDim, maxDiff);
		T_IO += getETtime() - temp_t;

		std::cout << "L=" << L << " T_pre: " << T_pre << " T_diag: " << T_diag
		          << " T_meas: " << T_meas << " T_IO: " << T_IO << std::endl;
	}

	return EXIT_SUCCESS;
}