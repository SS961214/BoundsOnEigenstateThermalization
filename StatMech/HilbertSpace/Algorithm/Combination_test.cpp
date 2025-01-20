#include "Combination.hpp"
#include <Eigen/Dense>
#include <iomanip>

int main(int argc, char* argv[]) {
	if(argc < 2) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	int const L = std::atoi(argv[1]);
	int const N = std::atoi(argv[2]);

	Combination const comb(L, N);

	std::cout << "# L = " << L << ", N = " << N << ", comb.dim() = " << comb.dim() << std::endl;
#pragma omp parallel for ordered
	for(int idx = 0; idx < comb.dim(); ++idx) {
		auto const config  = comb.ordinal_to_config(idx);
		auto const ordinal = comb.config_to_ordinal(config);
		auto const bitString
		    = Eigen::ArrayXi::NullaryExpr(L, [&config](int i) { return (config & (1 << i)) >> i; });
		auto const reversedIdx = comb.reverse(idx);
		auto const reversed    = comb.ordinal_to_config(reversedIdx);
		auto const revString   = Eigen::ArrayXi::NullaryExpr(
            L, [&reversed](int i) { return (reversed & (1 << i)) >> i; });

#pragma omp ordered
		std::cout << "# idx = " << std::setw(6) << idx << ", config = " << std::setw(8) << config
		          << ", ordinal = " << std::setw(6) << ordinal
		          << ", bitString = " << bitString.transpose()
		          << ",\t revString = " << revString.transpose() << std::endl;
		assert(idx == ordinal);
		// assert(bitString == revString);
	}

	return EXIT_SUCCESS;
}
