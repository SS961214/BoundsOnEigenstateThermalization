#define EIGEN_USE_MKL_ALL

#include "Hamiltonian.hpp"
#include <StatMech>
#include <HilbertSpace>
#include <omp.h>

using Scalar = std::complex<double>;

int main(int argc, char* argv[]) {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	if(argc != 9) {
		std::cerr << "Usage: 0.(This) 1.(LMax) 2.(LMin) 3.(NMax) 4.(NMin) 5.(t1) "
		             "6.(V1) 7.(t2) 8.(V2)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	Index const  LMax = std::atoi(argv[1]);
	Index const  LMin = std::atoi(argv[2]);
	Index const  NMax = std::atoi(argv[3]);
	Index const  NMin = std::atoi(argv[4]);
	double const t1   = std::atof(argv[5]);
	double const J1   = std::atof(argv[6]);
	double const t2   = std::atof(argv[7]);
	double const J2   = std::atof(argv[8]);

	constexpr int momentum = 0;
	double        startT, endT;
	for(auto L = LMin; L <= LMax; ++L)
		for(auto N = NMin; N <= std::min(NMax, L); ++N) {
			ManyBodyFermionSpace const mbSpace(L, N);

			startT                                 = omp_get_wtime();
			Eigen::SparseMatrix<double> const Htot = FermiHubbard(mbSpace, t1, J1, t2, J2, PBC);
			endT                                   = omp_get_wtime();
			// std::cout << "# Constructed the Hamiltonian: Nonzeros = " << H.nonZeros()
			//           << ", elapsed = " << endT - startT << " (sec)" << std::endl;
			std::cout << "# L = " << L << ", N = " << N << std::endl;
			{
				TransSector<std::decay_t<decltype(mbSpace)>, Scalar> const sector(momentum,
				                                                                  mbSpace);
				Eigen::MatrixXd H = (sector.basis().adjoint() * Htot * sector.basis()).real();
				Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> const solver(
				    std::move(H));
				double const ratio = LevelSpacingRatio(solver.eigenvalues());
				std::cout << "# ratio(TransSector)       = " << ratio
				          << ", sectorDim = " << sector.dim() << std::endl;
			}
			for(int parity = 1; parity >= -1; parity -= 2) {
				TransParitySector<std::decay_t<decltype(mbSpace)>, Scalar> const sector(parity,
				                                                                        mbSpace);
				if(sector.dim() <= 2) continue;
				Eigen::MatrixXd H = (sector.basis().adjoint() * Htot * sector.basis()).real();
				Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> const solver(
				    std::move(H));
				double const ratio = LevelSpacingRatio(solver.eigenvalues());
				std::cout << "# ratio(TransParitySector) = " << ratio << " (parity=" << parity
				          << ")" << ", sectorDim = " << sector.dim() << std::endl;
			}
		}

	return EXIT_SUCCESS;
}