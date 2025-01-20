#include "mBodyOpSpace_Boson_GramMat.hpp"
#include <iostream>
#include <complex>
#include <algorithm>

using Scalar = std::complex<double>;

int main(int argc, char* argv[]) {
	if(argc != 4) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N) 3.(m)" << std::endl;
		return EXIT_FAILURE;
	}
	int const L = std::atoi(argv[1]);
	int const N = std::atoi(argv[2]);
	int const m = std::atoi(argv[3]);

	// ManyBodyBosonSpace space(L, N);
	mBodyOpSpace<ManyBodyBosonSpace, Scalar> opSpace(m, L, N);
	auto                                     blocks = blocksInGramMat(opSpace);

	std::cout << "   blocks.size() = " << blocks.size() << std::endl;
	std::cout << "blocks[0].size() = " << blocks[0].size() << std::endl;

	std::sort(blocks.begin() + 1, blocks.end(),
	          [](auto const& a, auto const& b) { return a[0].size() > b[0].size(); });

	// 1. Determine the block sizes and their multiplicities
	int nSizeBlocks = 2;
#pragma omp parallel for reduction(+ : nSizeBlocks)
	for(uint b = 2; b < blocks.size(); ++b)
		if(blocks[b][0].size() != blocks[b - 1][0].size()) nSizeBlocks += 1;
	Eigen::ArrayXXi blockSizes(nSizeBlocks, 3);
	{
		// Mark the position of the beggining of the blocks with the same size
		int head            = 0;
		blockSizes(head, 0) = blocks[1][0].size();
		blockSizes(head, 1) = 1;
		head += 1;
#pragma omp parallel for ordered
		for(uint b = 2; b < blocks.size(); ++b) {
			if(blocks[b][0].size() == blocks[b - 1][0].size()) continue;
#pragma omp ordered
			{
				blockSizes(head, 0) = blocks[b][0].size();
				blockSizes(head, 1) = b;
				head += 1;
			}
		}
		blockSizes(head, 1) = blocks.size();
		// (END) Mark the position of the beggining of the blocks with the same size
		// Calculate the number of blocks with the same size without considering translation orbits
		for(int row = 0; row < blockSizes.rows() - 1; ++row) {
			Index nBlocks = 0;
#pragma omp parallel for reduction(+ : nBlocks)
			for(int b = blockSizes(row, 1); b < blockSizes(row + 1, 1); ++b) {
				nBlocks += blocks[b].size();
			}
			blockSizes(row, 2) = nBlocks;
			blockSizes(row, 1) = blockSizes(row + 1, 1) - blockSizes(row, 1);
		}
		// (END) Calculate the number of blocks with the same size without considering translation orbits
		// Special treatment for the blocks with size 1
		blockSizes(head, 0) = 1;
		blockSizes(head, 1) = blocks[0].size();
		Index nBlocks       = 0;
#pragma omp parallel for reduction(+ : nBlocks)
		for(uint c = 0; c < blocks[0].size(); ++c) { nBlocks += blocks[0][c].size(); }
		blockSizes(head, 2) = nBlocks;
		// (END) Special treatment for the blocks with size 1
	}
	std::cout << "# Block sizes:\n" << blockSizes << std::endl;

	Index opDim = (blockSizes.col(0) * blockSizes.col(2)).sum();
	std::cout << "# opSpace.dim() = " << opSpace.dim() << ", opDim = " << opDim << std::endl;
	assert(opDim == opSpace.dim());

	return EXIT_SUCCESS;
}