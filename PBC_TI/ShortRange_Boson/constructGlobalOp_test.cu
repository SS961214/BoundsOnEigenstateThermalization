#ifdef NEBUG
	#undef NDEBUG
#endif
#define EIGEN_USE_MKL_ALL

#include "constructGlobalOp.hpp"
#include <StatMech>
#include <HilbertSpace>
#include <MatrixGPU>
#include <random>
#include <iostream>

using Scalar = std::complex<double>;

int main(int argc, char* argv[]) {
	if(argc != 6) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N) 3.(ell) 4.(m) 5.(seed)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "# EIGEN_USE_MKL_ALL is set. Using Intel(R) MKL for BLAS/LAPACK." << std::endl;
#endif
	GPU::MAGMA::get_controller();
	constexpr double precision = 1e-10;
	int const        L         = std::atoi(argv[1]);
	int const        N         = std::atoi(argv[2]);
	int const        ell       = std::atoi(argv[3]);
	int const        m         = std::atoi(argv[4]);
	int const        seed      = std::atoi(argv[5]);
	std::cout << "#(ManyBodyBosonSpace) L = " << L << ", N = " << N << ", ell = " << ell
	          << ", seed = " << seed << std::endl;

	mBodyOpSpace<ManyBodyBosonSpace, Scalar> const locOpSpace(m, ell, N);

	std::mt19937                     mt(seed);
	std::normal_distribution<double> Gaussian(0.0, 1.0);

	ManyBodyBosonSpace const                            stateSpace(L, N);
	mBodyOpSpace<ManyBodyBosonSpace, Scalar> const      opSpace(m, stateSpace);
	TransParitySector<ManyBodyBosonSpace, Scalar> const evenSector(+1, stateSpace);
	TransParitySector<ManyBodyBosonSpace, Scalar> const oddSector(-1, stateSpace);
	std::cout << "# locOpSpace.dim() = " << locOpSpace.dim()
	          << ", stateSpace.dim() = " << stateSpace.dim()
	          << ", opSpace.dim() = " << opSpace.dim()
	          << ", evenSector.dim() = " << evenSector.dim()
	          << ", oddSector.dim() = " << oddSector.dim() << std::endl;
	int const       Nsamples = 100;
	Eigen::ArrayXXd ratios(Nsamples, 2);
	for(int sample = 0; sample < Nsamples; ++sample) {
		Eigen::ArrayX<Scalar> coeff = Eigen::ArrayX<Scalar>::NullaryExpr(
		    locOpSpace.dim(), [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });

		double ratioEven = 0.0, ratioOdd = 0.0;
		{
			Eigen::MatrixX<Scalar> H = construct_globalOp(coeff, locOpSpace, opSpace, evenSector);
			GPU::SelfAdjointEigenSolver_mgpu<std::decay_t<decltype(H)>> const solver(
			    GPU::MAGMA::ngpus(), std::move(H), Eigen::EigenvaluesOnly);
			ratioEven = LevelSpacingRatio(solver.eigenvalues());
		}
		{
			Eigen::MatrixX<Scalar> H = construct_globalOp(coeff, locOpSpace, opSpace, oddSector);
			GPU::SelfAdjointEigenSolver_mgpu<std::decay_t<decltype(H)>> const solver(
			    GPU::MAGMA::ngpus(), std::move(H), Eigen::EigenvaluesOnly);
			ratioOdd = LevelSpacingRatio(solver.eigenvalues());
		}
		ratios(sample, 0) = ratioEven;
		ratios(sample, 1) = ratioOdd;
		std::cout << "# Sample = " << sample << ", Level Spacings Ratio = " << ratios.row(sample)
		          << std::endl;
	}
	std::cout << "# Mean level spacings ratio = " << ratios.colwise().mean() << std::endl;

	return EXIT_SUCCESS;
}