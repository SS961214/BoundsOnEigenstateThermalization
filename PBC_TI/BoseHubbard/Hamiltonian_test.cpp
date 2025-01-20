#include "Hamiltonian.hpp"
#include "Hamiltonian_ref.hpp"
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
			ManyBodyBosonSpace mbSpace(L, N);

			startT                                 = omp_get_wtime();
			Eigen::SparseMatrix<double> const Href = BoseHubbard_ref(mbSpace, t1, J1, t2, J2, PBC);
			endT                                   = omp_get_wtime();
			std::cout << "# Constructed the reference Hamiltonian: Nonzeros = " << Href.nonZeros()
			          << ", elapsed = " << endT - startT << " (sec)" << std::endl;

			startT                              = omp_get_wtime();
			Eigen::SparseMatrix<double> const H = BoseHubbard(mbSpace, t1, J1, t2, J2, PBC);
			endT                                = omp_get_wtime();
			std::cout << "# Constructed the Hamiltonian: Nonzeros = " << H.nonZeros()
			          << ", elapsed = " << endT - startT << " (sec)" << std::endl;

			Eigen::SparseMatrix<double> diffMat = H - Href;
			diffMat.makeCompressed();
			Eigen::Map<Eigen::VectorXd> const diffCoeff(diffMat.valuePtr(), diffMat.nonZeros());
			double const                      diff = diffCoeff.cwiseAbs().maxCoeff();
			std::cout << "# Maximum difference = " << diff << std::endl;
			assert(diff < 1.0e-10);
			std::cout << std::endl;
		}

	return EXIT_SUCCESS;
}