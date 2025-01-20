#ifdef NEBUG
	#undef NDEBUG
#endif
#define EIGEN_USE_MKL_ALL

#include "constructGlobalOp.hpp"
#include <StatMech>
#include <HilbertSpace>
#include <random>
#include <iostream>

using Scalar = std::complex<double>;

int main(int argc, char* argv[]) {
	if(argc != 5) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N) 3.(ell) 4.(seed)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "# EIGEN_USE_MKL_ALL is set. Using Intel(R) MKL for BLAS/LAPACK." << std::endl;
#endif
	constexpr double precision = 1e-10;
	constexpr int    parity    = +1;
	constexpr int    m         = 2;
	int const        L         = std::atoi(argv[1]);
	int const        N         = std::atoi(argv[2]);
	int const        ell       = std::atoi(argv[3]);
	int const        seed      = std::atoi(argv[4]);
	std::cout << "#(ManyBodyFermionSpace) L = " << L << ", N = " << N << ", ell = " << ell
	          << ", m = " << m << ", seed = " << seed << std::endl;

	Combination const locOpConfig(ell, m);

	std::mt19937                     mt(seed);
	std::normal_distribution<double> Gaussian(0.0, 1.0);

	ManyBodyFermionSpace const                            stateSpace(L, N);
	mBodyOpSpace<ManyBodyFermionSpace, Scalar> const      opSpace(m, stateSpace);
	TransParitySector<ManyBodyFermionSpace, Scalar> const subSpace(parity, stateSpace);
	std::cout << "# locOpConfig.dim() = " << locOpConfig.dim()
	          << ", stateSpace.dim() = " << stateSpace.dim()
	          << ", opSpace.dim() = " << opSpace.dim() << ", subSpace.dim() = " << subSpace.dim()
	          << std::endl;
	for(int sample = 0; sample < 100; ++sample) {
		Eigen::ArrayX<Scalar> coeff = Eigen::ArrayX<Scalar>::NullaryExpr(
		    locOpConfig.dim(), [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });

		Eigen::MatrixX<Scalar> const H = construct_globalOp(coeff, locOpConfig, opSpace, subSpace);
		Eigen::SelfAdjointEigenSolver<std::decay_t<decltype(H)>> const solver(
		    H, Eigen::EigenvaluesOnly);
		double const lsratio = LevelSpacingRatio(solver.eigenvalues());
		std::cout << "# Sample = " << sample << ", Level Spacings Ratio = " << lsratio << std::endl;
	}

	return EXIT_SUCCESS;
}