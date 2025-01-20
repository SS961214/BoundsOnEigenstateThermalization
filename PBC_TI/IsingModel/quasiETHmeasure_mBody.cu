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
#include <unsupported/Eigen/KroneckerProduct>
#include <random>
#include <iostream>
#include <sys/time.h>
#include <filesystem>
namespace fs = std::filesystem;

using Scalar = cuda::std::complex<double>;

static inline Index powi(int base, Index expo) {
	int res = 1;
	for(Index j = 0; j != expo; ++j) res *= base;
	return res;
}

double Jx;
double Bx;
double Bz;

template<>
fs::path IOManager::generate_filepath(Index sysSize) const {
	std::stringstream buff("");
	buff << std::setprecision(4) << std::defaultfloat << std::showpos;
	buff << "_Jx" << Jx << "_Bz" << Bz << "_Bx" << Bx;

	return this->rootDir() / "mBodyETH/PBC_TI" / ("Ising" + buff.str())
	       / ("SystemSize_L" + std::to_string(sysSize))
	       / ("mBody_quasiETHmeasureSq_dE" + this->shWith() + std::string(".txt"));
}

int main(int argc, char** argv) {
	if(argc != 9) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(MMax) 3.(MMin) 4.(Jx) 5.(Bz) 6.(Bx) "
		             "7.(shellWidthParam) 8.(OutDir)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	GPU::MAGMA::get_controller();

	constexpr Index dimLoc            = 2;
	constexpr int   parity            = +1;
	Index const     L                 = std::atoi(argv[1]);
	Index const     N                 = L;
	Index const     MMax              = std::atoi(argv[2]);
	Index const     MMin              = std::atoi(argv[3]);
	Jx                                = std::atof(argv[4]);
	Bz                                = std::atof(argv[5]);
	Bx                                = std::atof(argv[6]);
	double const      shellWidthParam = std::atof(argv[7]);
	std::string const shellWidthStr(argv[7]);
	IOManager         data_writer(argv[8]);
	std::cout << "rootDir = " << data_writer.rootDir() << std::endl;
	data_writer.shWith()        = shellWidthStr;

	using StateSpace = ManyBodySpinSpace;
	// h_j = Jx \sigma^{x}_{j} \sigma^{x}_{j+1} + Bz sigma^{z}_{j} + Bx sigma^{x}_{j}
	// Eigen::MatrixX<Scalar> locH = Eigen::MatrixXd::Zero(dimLoc * dimLoc, dimLoc * dimLoc);
	// locH(0, 0)                  = Bz;
	// locH(0, 1)                  = 0;
	// locH(0, 2)                  = Bx;
	// locH(0, 3)                  = Jx;
	// locH(1, 0)                  = 0;
	// locH(1, 1)                  = Bz;
	// locH(1, 2)                  = Jx;
	// locH(1, 3)                  = Bx;
	// locH(2, 0)                  = Bx;
	// locH(2, 1)                  = Jx;
	// locH(2, 2)                  = -Bz;
	// locH(2, 3)                  = 0;
	// locH(3, 0)                  = Jx;
	// locH(3, 1)                  = Bx;
	// locH(3, 2)                  = 0;
	// locH(3, 3)                  = -Bz;

	Eigen::MatrixX<Scalar> locH;
	{
		Eigen::MatrixX<Scalar> sigmaI(2, 2);
		Eigen::MatrixX<Scalar> sigmaX(2, 2);
		Eigen::MatrixX<Scalar> sigmaZ(2, 2);
		sigmaI << 1, 0, 0, 1;
		sigmaX << 0, 1, 1, 0;
		sigmaZ << 1, 0, 0, -1;
		locH = Jx * Eigen::kroneckerProduct(sigmaX, sigmaX)
		       + Bz * Eigen::kroneckerProduct(sigmaZ, sigmaI)
		       + Bx * Eigen::kroneckerProduct(sigmaX, sigmaI);
	}
	Index const                 dimC = powi(dimLoc, L - 2);
	Eigen::SparseMatrix<Scalar> idC(dimC, dimC);
	idC.setIdentity();

	double temp_t;
	double T_pre = 0, T_diag = 0, T_meas = 0, T_IO = 0;
	{
		StateSpace const                            mbSpace(L, dimLoc);
		TransParitySector<StateSpace, Scalar> const sector(parity, mbSpace);

		temp_t = omp_get_wtime();
		Eigen::MatrixX<Scalar> H
		    = sector.basis().adjoint() * Eigen::kroneckerProduct(locH, idC) * sector.basis();
		T_pre += omp_get_wtime() - temp_t;

		temp_t = omp_get_wtime();
		GPU::SelfAdjointEigenSolver_mgpu<std::decay_t<decltype(H)>> const solver(
		    GPU::MAGMA::ngpus(), std::move(H));
		T_diag += omp_get_wtime() - temp_t;
		std::cout << "# L = " << L << ": Diagonalized the Hamiltonian.\t Elapsed = " << T_diag
		          << " (sec)" << std::endl;

		auto const   eigRange  = solver.eigenvalues().maxCoeff() - solver.eigenvalues().minCoeff();
		double const shWidth   = eigRange * shellWidthParam / L;
		auto const   shellDims = get_shellDims(solver.eigenvalues(), shWidth, solver.eigenvalues());
		double const lsRatio   = LevelSpacingRatio(solver.eigenvalues());
		{
			std::stringstream buff("");
			buff << "#Jx = " << Jx << "\n"
			     << "#Bz = " << Bz << "\n"
			     << "#Bx = " << Bx << "\n"
			     << "#parity = " << parity << "\n"
			     << "#LevelSpacingRatio = " << lsRatio << "\n";
			data_writer.custom_header() = buff.str();
		}

		thrust::device_vector<double>            const dEigVals(solver.eigenvalues().begin(),
		                                                  solver.eigenvalues().end());
		GPU::MatrixGPU< Eigen::MatrixX<Scalar> > const dEigVecs(solver.eigenvectors());

		Eigen::ArrayXXd ETHmeasure
		    = Eigen::ArrayXXd::Constant(solver.eigenvectors().cols(), N + 1, std::nan(""));
		temp_t = omp_get_wtime();
		for(auto m = MMin; m <= std::min(MMax, N); ++m) {
			mBodyOpSpace<StateSpace, Scalar> const             opSpace(m, L, dimLoc);
			ObjectOnGPU<std::decay_t<decltype(opSpace)>> const dOpSpace(opSpace);

			std::cout << "\n# L = " << std::setw(2) << L << ", N = " << std::setw(2) << N
			          << ", m = " << std::setw(2) << m << ", dim = " << sector.dim()
			          << ", dimTot = " << sector.dimTot() << ", opDim = " << opSpace.dim()
			          << std::endl;
			auto                 execTime = omp_get_wtime();
			Eigen::ArrayXd const res
			    = StatMech::ETHMeasure2(dEigVals, dEigVecs, opSpace, dOpSpace, sector, shWidth);
			execTime = omp_get_wtime() - execTime;

			temp_t = omp_get_wtime();
			data_writer.load_Data(ETHmeasure, L);
			ETHmeasure.col(m) = res;
			data_writer.save_ResultsToFile(solver.eigenvalues(), ETHmeasure, L, shellDims,
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
			Eigen::ArrayXd resSum = Eigen::ArrayXd::Zero(ETHmeasure.rows());
			for(auto m = 1; m < ETHmeasure.cols(); ++m) { resSum += ETHmeasure.col(m); }
			maxDiff = (resSum - theorySum).cwiseAbs().maxCoeff();
			if(!std::isnan(maxDiff) && maxDiff > precision) {
				std::cout << "ETHmeasure.rowwise().sum():\n"
				          << ETHmeasure.rowwise().sum() << "\n"
				          << std::endl;
				std::cout << "theorySum:\n" << theorySum << "\n" << std::endl;
				std::cout << "ETHmeasure:\n" << ETHmeasure << "\n" << std::endl;
				std::cerr << "Error : maxDiff = " << maxDiff << " is too large." << std::endl;
				// std::exit(EXIT_SUCCESS);
				std::exit(EXIT_FAILURE);
			}
		}

		temp_t = omp_get_wtime();
		data_writer.save_ResultsToFile(solver.eigenvalues(), ETHmeasure, L, shellDims, maxDiff);
		T_IO += omp_get_wtime() - temp_t;

		std::cout << "L=" << L << " T_pre: " << T_pre << " T_diag: " << T_diag
		          << " T_meas: " << T_meas << " T_IO: " << T_IO << std::endl;
	}

	return EXIT_SUCCESS;
}