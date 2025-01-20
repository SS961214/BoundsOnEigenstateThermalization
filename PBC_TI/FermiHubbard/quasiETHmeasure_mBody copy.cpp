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

#include "Hamiltonian.hpp"
#include <StatMech/HilbertSpace/OpSpace/mBodyOpSpace_Fermion.hpp>
#include <StatMech/quasiETHmeasure.hpp>

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <filesystem>
namespace fs = std::filesystem;

using Scalar = std::complex<double>;

// Forward declarations
class IOManager {
	private:
		std::string m_outRootDir;
		fs::path    generate_filepath(Index sysSize) const;

	public:
		IOManager(std::string outRootDir) : m_outRootDir{outRootDir} {}

		fs::path outDir() const;

		template<class Vector_, class Matrix_, class iVector_>
		bool save_ResultsToFile(Vector_ const& eigVals, Matrix_ const& ETHmeasure, Index sysSize,
		                        iVector_ const& dimShell, double const maxDiff) const;
};

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

double t1, V1, t2, V2;

int main(int argc, char** argv) {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	Eigen::initParallel();

	if(argc != 11) {
		std::cerr << "Usage: 0.(This) 1.(LMax) 2.(LMin) 3.(MMax) 4.(MMin) 5.(t1) 6.(V1) 7.(t2) 8.(V2)"
		             "9.(shellWidth) "
		             "10.(OutDir)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	Index const     LMax      = std::atoi(argv[1]);
	Index const     LMin      = std::atoi(argv[2]);
	Index const     MMax      = std::atoi(argv[3]);
	Index const     MMin      = std::atoi(argv[4]);
	t1                        = std::atof(argv[5]);
	V1                        = std::atof(argv[6]);
	t2                        = std::atof(argv[7]);
	V2                        = std::atof(argv[8]);
	double const   shellWidth = std::atof(argv[9]);
	IOManager const data_writer(argv[10]);
	std::cout << "outDir = " << data_writer.outDir() << std::endl;

	constexpr int momentum = 0;

	double temp_t;
	double T_pre = 0, T_diag = 0, T_meas = 0, T_IO = 0;
	for(Index L = LMin; L <= LMax; ++L) {
		if(L % 3 != 0) continue;
		auto const N = L/3; // Consider the case of filling 1/3
		ManyBodyFermionSpace                      mbSpace(L, N);
		TransSector< decltype(mbSpace), Scalar> transSector(momentum, mbSpace);

		temp_t                = getETtime();
		Eigen::MatrixXcd Htot = transSector.basis().adjoint() * FermiHubbard(mbSpace, t1, V1, t2, V2, PBC) * transSector.basis();
		T_pre += getETtime() - temp_t;

		temp_t = getETtime();
		Eigen::SelfAdjointEigenSolver< decltype(Htot) > eigSolver(std::move(Htot));
		T_diag += getETtime() - temp_t;

		auto const eigRange
		    = *std::max_element(eigSolver.eigenvalues().begin(), eigSolver.eigenvalues().end())
		      - *std::min_element(eigSolver.eigenvalues().begin(), eigSolver.eigenvalues().end());
		auto const shellDim = get_shellDims(eigSolver.eigenvalues(), eigRange * shellWidth);

		Eigen::ArrayXXd ETHmeasure = Eigen::ArrayXXd::Zero(eigSolver.eigenvectors().cols(), L + 1);
		Eigen::ArrayXd  colVec;
		temp_t = getETtime();
		for(Index m = MMin; m <= std::min(MMax, L); ++m) {
			mBodyOpSpace< decltype(mbSpace), Scalar > opSpace(m, mbSpace);
			std::cout << "m=" << m << ", opSpace.dim() = " << opSpace.dim() << std::endl;
			colVec = StatMech::ETHmeasure2Sq(eigSolver.eigenvectors(), eigSolver.eigenvalues(),
			                                     opSpace, transSector, eigRange * shellWidth);

			ETHmeasure.col(m) += colVec;
		}
		std::cout << std::endl;
		T_meas += getETtime() - temp_t;

		double maxDiff;
		{  // Verifying the results
			auto theorySum = Eigen::ArrayXd::Ones(shellDim.size())
			                 - shellDim.template cast<double>().inverse();
			maxDiff = (ETHmeasure.rowwise().sum() - theorySum).maxCoeff();
			if(maxDiff > 1.0e-10) {
				std::cout << "ETHmeasure.rowwise().sum():\n"
				          << ETHmeasure.rowwise().sum() << "\n"
				          << std::endl;
				std::cout << "theorySum:\n" << theorySum << "\n" << std::endl;
				std::cout << "ETHmeasure:\n" << ETHmeasure << "\n" << std::endl;
				std::cerr << "Error : maxDiff = " << maxDiff << " is too large." << std::endl;
				// continue;
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

#include <fstream>
#include <algorithm>

#define QUOTE(x) Q(x)
#define Q(x)     #x
#ifndef DATA_PREFIX
	#define DATA_PREFIX "TypicalityOfETH"
#endif
#ifdef FLOAT
	#define PRECISION "C"
#else
	#define PRECISION "Z"
#endif

#define ENSEMBLE "PBC_TI/FermiHubbard"

fs::path IOManager::outDir() const {
	fs::path filename(m_outRootDir);
	filename.append(DATA_PREFIX);
	filename.append(ENSEMBLE);
	return filename;
}

fs::path IOManager::generate_filepath(Index sysSize) const {
	std::stringstream buff("");
	buff << std::setprecision(2) << std::defaultfloat << std::showpos;
	buff << "_t1" << t1 << "_V1" << V1 << "_t2" << t2 << "_V2" << V2 << "_n0.33";
	fs::path filename = this->outDir();
	filename.append("FermiHubbard" + buff.str());
	filename.append("SystemSize_L" + std::to_string(sysSize));
	filename.append(std::string("nBody_quasiETHmeasureSq") + std::string(PRECISION)
	                + std::string(".txt"));
	return filename;
}

template<class Vector_, class Matrix_, class iVector_>
bool IOManager::save_ResultsToFile(Vector_ const& eigVals, Matrix_ const& ETHmeasure,
                                   Index sysSize, iVector_ const& dimShell,
                                   double const maxDiff) const {
	auto const gE          = *std::min_element(eigVals.begin(), eigVals.end());
	auto const energyRange = *std::max_element(eigVals.begin(), eigVals.end()) - gE;

	fs::path outFilePath = this->generate_filepath(sysSize);
	fs::create_directories(outFilePath.parent_path());
	if(!fs::exists(outFilePath.parent_path())) {
		std::cerr << "Error: Failed to create a directory " << outFilePath.parent_path()
		          << std::endl;
		std::exit(EXIT_FAILURE);
	}
	std::ofstream resFile(outFilePath);
	if(!resFile) {
		std::cerr << "Error: Can't open a file (" << outFilePath << ")" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::stringstream headers("");
	headers << std::setprecision(6) << std::scientific << std::showpos;
	headers << "#energyRange = " << energyRange << "\n"
	        << "#gE          = " << gE << "\n"
	        << "#maxDiff     = " << maxDiff << "\n\n";
	resFile << headers.str();
	resFile << "# 1.(Normalized energy) 2.(shellDim) 3.4. ...(quasi ETH measures)\n";

	resFile << std::setprecision(6) << std::scientific << std::showpos;
	for(auto j = 0; j < eigVals.size(); ++j) {
		resFile << (eigVals[j] - gE) / energyRange << " " << dimShell[j] << " ";
		for(Index m = 1; m <= sysSize; ++m) { resFile << ETHmeasure(j, m) << " "; }
		resFile << "\n";
	}
	resFile.close();

	return true;
}
