#include "Hamiltonian.hpp"
#include <HilbertSpace>
#include <omp.h>

using Scalar = std::complex<double>;

int main(int argc, char* argv[]) {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	Eigen::initParallel();

	if(argc != 11) {
		std::cerr << "Usage: 0.(This) 1.(LMax) 2.(LMin) 3.(NMax) 4.(NMin) 5.(MMax) 6.(MMin) 7.(t1) "
		             "8.(V1) 9.(t2) 10.(V2)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	Index const  LMax = std::atoi(argv[1]);
	Index const  LMin = std::atoi(argv[2]);
	Index const  NMax = std::atoi(argv[3]);
	Index const  NMin = std::atoi(argv[4]);
	Index const  MMax = std::atoi(argv[5]);
	Index const  MMin = std::atoi(argv[6]);
	double const t1   = std::atof(argv[7]);
	double const J1   = std::atof(argv[8]);
	double const t2   = std::atof(argv[9]);
	double const J2   = std::atof(argv[10]);

	constexpr int momentum = 0;
	double startT;
	for(auto L = LMin; L <= LMax; ++L)
		for(auto N = NMin; N <= std::min(NMax, L); ++N) {
			ManyBodyFermionSpace mbSpace(L, N);

			startT = omp_get_wtime();
			TransSector< decltype(mbSpace), Scalar> transSector(momentum, mbSpace);
			std::cout << "# Constructed the TransSector: L = " << L << ", N = " << N
			         << ", dim = " << transSector.dim()
			         << ", elapsed = " << omp_get_wtime() - startT << " (sec)" << std::endl;

			startT                                 = omp_get_wtime();
			Eigen::SparseMatrix<double> const Hraw = FermiHubbard(mbSpace, t1, J1, t2, J2, PBC);
			std::cout << "# Constructed the raw Hamiltonian: Nonzeros = " << Hraw.nonZeros()
			          << ", elapsed = " << omp_get_wtime() - startT << " (sec)" << std::endl;

			startT = omp_get_wtime();
			Eigen::SparseMatrix<Scalar> const H
			    = transSector.basis().adjoint() * Hraw * transSector.basis();
			std::cout << "# Constructed the Hamiltonian in a sector: Nonzeros = " << H.nonZeros()
			          << ", elapsed = " << omp_get_wtime() - startT << " (sec)\n"
			          << std::endl;
		}

	return EXIT_SUCCESS;
}