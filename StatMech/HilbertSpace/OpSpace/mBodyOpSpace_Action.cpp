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

#include "mBodyOpSpace_Spin.hpp"
#include "mBodyOpSpace_Boson.hpp"
#include "mBodyOpSpace_Fermion.hpp"
#include <complex>
#include <iostream>
#include <iomanip>

#if defined(Spin)
using MBSpace = ManyBodySpinSpace;
#elif defined(Boson)
using MBSpace = ManyBodyBosonSpace;
#elif defined(Fermion)
using MBSpace = ManyBodyFermionSpace;
#endif

using Scalar = std::complex<double>;

int main(int argc, char** argv) {
#ifdef Spin
	if(argc < 3) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(m) 3.(opNum)" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	int const     L     = std::atoi(argv[1]);
	constexpr int N     = 2;
	int const     m     = std::atoi(argv[2]);
	int const     opNum = std::atoi(argv[3]);
#else
	if(argc < 4) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N) 3.(m) 4.(opNum)" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	int const L     = std::atoi(argv[1]);
	int const N     = std::atoi(argv[2]);
	int const m     = std::atoi(argv[3]);
	int const opNum = std::atoi(argv[4]);
#endif

	mBodyOpSpace<MBSpace, Scalar> opSpace(m, L, N);
	auto const&                   hSpace = opSpace.baseSpace();

	std::cout << "# hSpace.dim() = " << hSpace.dim() << ", opSpace.dim() = " << opSpace.dim()
	          << std::endl;
	Eigen::ArrayXi config(opSpace.sysSize());
	opSpace.ordinal_to_config(config, opNum);

	for(auto j = 0; j < hSpace.dim(); ++j) {
		auto [res, coeff] = opSpace.action(opNum, j);
		std::cout << "#\t coeff = " << coeff << std::endl;
		std::cout << "#\t               In: " << hSpace.ordinal_to_config(j) << std::endl;
#ifdef Spin
		std::cout << "#\t         opConfig: " << config.transpose() << std::endl;
#else
		std::cout
		    << "#\t  Annihilation Op: "
		    << config.unaryExpr([&](int x) { return x % (opSpace.maxOnSite() + 1); }).transpose()
		    << std::endl;
		std::cout
		    << "#\t      Creation Op: "
		    << config.unaryExpr([&](int x) { return x / (opSpace.maxOnSite() + 1); }).transpose()
		    << std::endl;
#endif
		std::cout << "#\t              Out: " << hSpace.ordinal_to_config(res) << std::endl;
	}

	return EXIT_SUCCESS;
}