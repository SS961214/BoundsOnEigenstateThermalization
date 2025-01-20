#define EIGEN_DEFAULT_IO_FORMAT \
	Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", "", "", "", "", 10)

#include "TransParitySector.hpp"
#include "ManyBodyFermionSpace.hpp"
#include <complex>
#include <iostream>
#include <iomanip>

using Scalar = std::complex<double>;

int main(int argc, char* argv[]) {
	if(argc < 3) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	int const L = std::atoi(argv[1]);
	int const N = std::atoi(argv[2]);

	ManyBodyFermionSpace const mbSpace(L, N);
	for(int parity = 1; parity >= -1; parity -= 2) {
		TransParitySector<std::decay_t<decltype(mbSpace)>, Scalar> sector(parity, mbSpace);
		std::cout << "# L = " << L << ", N = " << N << ", parity = " << parity
		          << ": dim = " << mbSpace.dim() << ", sectorDim = " << sector.dim() << std::endl;

		auto const basis = Eigen::MatrixXd::NullaryExpr(
		    mbSpace.dim(), sector.dim(),
		    [&](int i, int j) { return sector.basis().coeffRef(i, j).real(); });

		std::cout << "## Basis:\n";
		for(int state = 0; state < mbSpace.dim(); ++state) {
			std::cout << "# state = " << std::setw(5) << state << ": "
			          << mbSpace.ordinal_to_config(state) << ":\t";
			for(int col = 0; col < sector.dim(); ++col) {
				std::cout << std::setw(8) << basis(state, col) << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "# " << std::endl;
	}

	return EXIT_SUCCESS;
}