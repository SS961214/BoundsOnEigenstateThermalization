#include "mBodyOpSpace_Fermion.hpp"

using Scalar = std::complex<double>;
int main(int argc, char** argv) {
	if(argc != 5) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N) 3.(m) 4.(basisOpNum)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	int const L          = std::atoi(argv[1]);
	int const N          = std::atoi(argv[2]);
	int const m          = std::atoi(argv[3]);
	int const basisOpNum = std::atoi(argv[4]);

	ManyBodyFermionSpace                     mbHSpace(L, N);
	mBodyOpSpace<decltype(mbHSpace), Scalar> opSpace(m, mbHSpace);

	std::cout << "# State configurations" << std::endl;
	Eigen::RowVectorXi config(L);
	for(auto j = 0; j < mbHSpace.dim(); ++j) {
		mbHSpace.ordinal_to_config(config, j);
		std::cout << "# " << j << ": " << config << std::endl;
	}
	std::cout << std::endl;

	std::cout << "# Operator configurations" << std::endl;
	opSpace.ordinal_to_config(config, basisOpNum);
	std::cout << config.array() / 2 << std::endl;
	std::cout << config.unaryExpr([](const int x) { return (x & 1); }) << "\n" << std::endl;

	std::cout << Eigen::MatrixX<Scalar>(opSpace.basisOp(basisOpNum)) << std::endl;
	return EXIT_SUCCESS;
}