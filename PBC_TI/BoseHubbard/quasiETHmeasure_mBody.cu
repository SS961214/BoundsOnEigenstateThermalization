/**
 * @file quasiETHmeasure_mBody.cu
 * @author Shoki Sugimoto (shoki.sugimoto@riken.jp)
 * @brief
 * @version 0.1
 * @date 2024-07-25
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

#include "Hamiltonian.hpp"
#include "dataHandler.hpp"
#include <HilbertSpace>
#include <StatMech>
#include <Eigen/Dense>
#include <iostream>
#include <omp.h>
#include <filesystem>
namespace fs = std::filesystem;

using RealScalar = double;
using Scalar     = cuda::std::complex<RealScalar>;

double J1, J2;
double U1, U2;

template<>
fs::path IOManager::generate_filepath(Index sysSize, Index N) const {
	std::stringstream buff("");
	buff << std::setprecision(2) << std::defaultfloat << std::showpos;
	buff << "_J1" << J1 << "_U1" << U1 << "_J2" << J2 << "_U2" << U2;

	return this->rootDir() / "mBodyETH/PBC_TI" / ("BoseHubbard" + buff.str())
	       / ("SystemSize_L" + std::to_string(sysSize) + "_N" + std::to_string(N))
	       / ("mBody_quasiETHmeasureSq_dE" + this->shWith() + std::string(".txt"));
}

int main(int argc, char** argv) {
	if(argc != 13) {
		std::cerr << "Usage: 0.(This) 1.(LMax) 2.(LMin) 3.(NMax) 4.(NMin) 5.(MMax) 6.(MMin) 7.(J1) "
		             "8.(U1) 9.(J2) 10.(U2) "
		             "11.(shellWidthParam) "
		             "12.(OutDir)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	GPU::MAGMA::get_controller();

	constexpr double precision = 1.0e-12;

	Index const LMax                  = std::atoi(argv[1]);
	Index const LMin                  = std::atoi(argv[2]);
	Index const NMax                  = std::atoi(argv[3]);
	Index const NMin                  = std::atoi(argv[4]);
	Index const MMax                  = std::atoi(argv[5]);
	Index const MMin                  = std::atoi(argv[6]);
	J1                                = std::atof(argv[7]);
	U1                                = std::atof(argv[8]);
	J2                                = std::atof(argv[9]);
	U2                                = std::atof(argv[10]);
	double const      shellWidthParam = std::atof(argv[11]);
	std::string const shellWidthStr   = std::string(argv[11]);
	IOManager         data_writer(argv[12]);
	std::cout << "rootDir = " << data_writer.rootDir() << std::endl;

	constexpr int parity = 1;

	double tempT;
	double T_pre = 0, T_diag = 0, T_meas = 0, T_IO = 0;
	for(auto L = LMin; L <= LMax; ++L)
		for(auto N = NMin; N <= NMax; ++N) {
			ManyBodyBosonSpace const                                           mbSpace(L, N);
			TransParitySector< std::decay_t<decltype(mbSpace)>, Scalar > const subSpace(parity,
			                                                                            mbSpace);

			tempT                       = omp_get_wtime();
			Eigen::MatrixX<Scalar> Htot = subSpace.basis().adjoint()
			                              * BoseHubbard(mbSpace, J1, U1, J2, U2, PBC)
			                              * subSpace.basis();
			T_pre += omp_get_wtime() - tempT;

			tempT = omp_get_wtime();
			GPU::SelfAdjointEigenSolver_mgpu< std::decay_t<decltype(Htot)> > solver(
			    GPU::MAGMA::ngpus(), std::move(Htot));
			// {
			// 	double const err = (Htot.selfadjointView<Eigen::Lower>() * solver.eigenvectors()
			// 	                    - solver.eigenvectors() * solver.eigenvalues().asDiagonal())
			// 	                       .cwiseAbs()
			// 	                       .maxCoeff();
			// 	if(err > precision) {
			// 		std::cerr << "# Error: Diagonalization failed with err = " << err << std::endl;
			// 		std::exit(EXIT_FAILURE);
			// 	}
			// }
			T_diag += omp_get_wtime() - tempT;
			std::cout << "# (L, N) = (" << L << ", " << N
			          << "): Diagonalized the Hamiltonian.\t Elapsed = " << T_diag << " (sec)"
			          << std::endl;

			auto const eigRange = solver.eigenvalues().maxCoeff() - solver.eigenvalues().minCoeff();
			double const         shWidth = eigRange * shellWidthParam / L;
			Eigen::ArrayXi const shellDims
			    = get_shellDims(solver.eigenvalues(), shWidth, solver.eigenvalues());
			double const lsRatio = LevelSpacingRatio(solver.eigenvalues());
			{
				std::stringstream buff("");
				buff << "#J1 = " << J1 << "\n"
				     << "#U1 = " << U1 << "\n"
				     << "#J2 = " << J2 << "\n"
				     << "#U2 = " << U2 << "\n"
				     << "#parity = " << parity << "\n"
				     << "#LevelSpacingRatio = " << lsRatio << "\n";
				data_writer.custom_header() = buff.str();
				data_writer.shWith()        = shellWidthStr;
			}

			Eigen::ArrayXXd ETHmeasure
			    = Eigen::ArrayXXd::Constant(subSpace.dim(), N + 1, std::nan(""));
			tempT = omp_get_wtime();
			for(auto m = MMin; m <= std::min(MMax, N); ++m) {
				mBodyOpSpace<std::decay_t<decltype(mbSpace)>, Scalar> opSpace(m, L, N);

				std::cout << "L = " << std::setw(2) << L << ", N = " << std::setw(2) << N
				          << ", m = " << std::setw(2) << m << ", dim = " << subSpace.dim()
				          << ", dimTot = " << subSpace.dimTot() << ", opDim = " << opSpace.dim()
				          << std::endl;
				auto                 execTime = omp_get_wtime();
				Eigen::ArrayXd const res
				    = StatMech::ETHmeasure2Sq(solver, opSpace, subSpace, shWidth);
				execTime = omp_get_wtime() - execTime;
				// std::cout << "# res:\n" << res << "\n" << std::endl;

				tempT = omp_get_wtime();
				data_writer.load_Data(ETHmeasure, L, N);
				ETHmeasure.col(m) = res;
				data_writer.save_ResultsToFile(solver.eigenvalues(), ETHmeasure, L, N, shellDims,
				                               std::nan(""));
				T_IO += omp_get_wtime() - tempT;
				// totExecTime += execTime;
				std::cout << "#\t exec time = " << execTime << " (sec)" << std::endl;
			}
			std::cout << std::endl;
			T_meas += omp_get_wtime() - tempT;

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

			tempT = omp_get_wtime();
			data_writer.save_ResultsToFile(solver.eigenvalues(), ETHmeasure, L, N, shellDims,
			                               maxDiff);
			T_IO += omp_get_wtime() - tempT;

			std::cout << "L=" << L << ", N=" << N << ", T_pre: " << T_pre << " T_diag: " << T_diag
			          << " T_meas: " << T_meas << " T_IO: " << T_IO << "\n"
			          << std::endl;
		}

	return EXIT_SUCCESS;
}