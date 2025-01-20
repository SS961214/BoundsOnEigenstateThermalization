#define EIGEN_DEFAULT_IO_FORMAT \
	Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", ";\n", " ", "", "[", ";\n]")

#include "distributeBlocks.hpp"

using Scalar = std::complex<double>;

int main(int argc, char* argv[]) {
	if(argc != 5) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N) 3.(m) 4.(nDevs)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	int const L     = std::atoi(argv[1]);
	int const N     = std::atoi(argv[2]);
	int const m     = std::atoi(argv[3]);
	int const nDevs = std::atoi(argv[4]);
	std::cout << "# L: " << L << ", N: " << N << ", m: " << m << std::endl;

	using OpSpace_ = ManyBodyBosonSpace;
	mBodyOpSpace<OpSpace_, Scalar> const mbOpSpace(m, L, N);
	auto                                 blocks = blocksInGramMat(mbOpSpace);

	auto const [workloads, blockSizes] = StatMech::distributeBlocks(blocks, nDevs);

	auto const test = [&](auto const& func){
		Index sum = 0;
		// #pragma omp parallel for reduction(+ : sum)
		for(int dev = 0; dev < nDevs; ++dev) {
			Index const nBlocks = workloads[dev].period.size();
			for(Index b = 0; b < nBlocks; ++b) {
				Index part = 0;
				for(Index idx = workloads[dev].offset[b]; idx < workloads[dev].offset[b + 1];
				    ++idx) {
					part += func(workloads[dev].elems[idx]);
				}
				sum += part;
			}
		}
		// Reference calculation
		Index sumRef = 0;
#pragma omp parallel for reduction(+ : sumRef)
		for(uint b = 1; b < blocks.size(); ++b) {
			for(uint i = 0; i < blocks[b][0].size(); ++i) sumRef += func(blocks[b][0][i]);
		}
#pragma omp parallel for reduction(+ : sumRef)
		for(uint c = 0; c < blocks[0].size(); ++c) sumRef += func(blocks[0][c][0]);
		// (END) Reference calculation

		std::cout << "# sum = " << sum << ",\t sumRef = " << sumRef << std::endl;
		assert(sum == sumRef);
	};

	test([](auto const& x) { return x; });
	test([](auto const& x) { return x * x; });
	test([](auto const& x) { return (x * x) % 142; });

	return 0;
}