#pragma once
#include <HilbertSpace>
#include <Eigen/Dense>
#include <vector>
#include <queue>

using Index = Eigen::Index;

namespace StatMech {

	class Assignment {
		public:
			std::vector<Index> elems;
			std::vector<Index> offset;
			std::vector<int>   period;
	};

	__host__ std::pair<std::vector<Assignment>, Eigen::ArrayXXi> distributeBlocks(
	    std::vector<std::vector<std::vector<Index>>>& blocks, int const ndev) {
		std::sort(blocks.begin() + 1, blocks.end(),
		          [](auto const& a, auto const& b) { return a[0].size() > b[0].size(); });

		auto const costFunc = [](Index const size) { return Index(std::pow(size, 1.5)); };

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
			// (END) Mark the position of the beggining of the blocks with the same size
			// Caclulate the number of blocks with the same size
			blockSizes(head, 1) = blocks.size();
			for(int j = 0; j < blockSizes.rows() - 1; ++j) {
				blockSizes(j, 1) = blockSizes(j + 1, 1) - blockSizes(j, 1);
			}
			// (END) Caclulate the number of blocks with the same size
			// Special treatment for the blocks with size 1
			blockSizes(head, 0) = 1;
			blockSizes(head, 1) = blocks[0].size();
			blockSizes(head, 2) = blockSizes(head, 1);
			// (END) Special treatment for the blocks with size 1
			blockSizes.col(2)
			    = Eigen::ArrayXi::NullaryExpr(blockSizes.rows(),
			                                  [&](int row) { return costFunc(blockSizes(row, 0)); })
			      * blockSizes.col(1);
		}
		std::cout << "# Block sizes:\n" << blockSizes << std::endl;
		// (END) 1. Determine the block sizes and their multiplicities

		// 2. Determine the number of blocks each device will deal with
		struct Part {
				Index current_cost = 0;
				int   devIdx       = 0;
				Part(Index cost, int idx) : current_cost(cost), devIdx(idx) {}
				bool operator>(const Part& other) const {
					return current_cost > other.current_cost;
				}
		};
		std::priority_queue<Part, std::vector<Part>, std::greater<Part>> part_sums;
		for(int i = 0; i < ndev; ++i) part_sums.push(Part(0, i));

		std::vector<Eigen::ArrayXXi> table(ndev, Eigen::ArrayXXi(blockSizes.rows(), 2));
		for(int i = 0; i < blockSizes.rows(); ++i) {
			for(int j = 0; j < ndev; ++j) {
				table[j](i, 0) = blockSizes(i, 0);
				table[j](i, 1) = 0;
			}
			Index const cost = costFunc(blockSizes(i, 0));
			for(int mult = blockSizes(i, 1); mult > 0;) {
				Part part1 = part_sums.top();
				part_sums.pop();
				Part part2 = part_sums.top();

				int assign = (part2.current_cost - part1.current_cost) / cost;
				assign     = std::min(mult, assign);
				assign     = std::max(assign, 1);

				part1.current_cost += cost * assign;
				mult -= assign;
				table[part1.devIdx](i, 1) += assign;
				part_sums.push(part1);
			}
		}
		for(int j = 0; j < ndev; ++j) {
			Index const cost
			    = (Eigen::ArrayXi::NullaryExpr(table[j].rows(),
			                                   [&](int row) { return costFunc(table[j](row, 0)); })
			       * table[j].col(1))
			          .sum();
			std::cout << "# Device " << j << ": cost = " << cost << "\n" << table[j] << std::endl;
		}
		// (END) 2. Determine the number of blocks each device will deal with

		// 3. Distribute the elements of the blocks to the devices
		std::vector<Assignment> res(ndev);
		std::vector<Index>      head(blockSizes.rows());
		head[0] = 1;
		for(int i = 1; i < blockSizes.rows() - 1; ++i) head[i] = head[i - 1] + blockSizes(i - 1, 1);
		head[blockSizes.rows() - 1] = 0;

		for(int dev = 0; dev < ndev; ++dev) {
			Index const nElems  = (table[dev].col(0) * table[dev].col(1)).sum();
			Index const nBlocks = table[dev].col(1).sum();
			res[dev].elems.resize(nElems);
			res[dev].offset.resize(nBlocks + 1);
			res[dev].period.resize(nBlocks);
			std::cout << "# Device " << dev << ": nElems = " << nElems << ", nBlocks = " << nBlocks
			          << std::endl;

			res[dev].offset[0] = 0;
			Index blockIdx     = 0;
			for(int j = 0; j < blockSizes.rows() - 1; ++j) {
				Index const cHead = head[j];
				head[j] += table[dev](j, 1);
				for(Index b = cHead; b < head[j]; ++b) {
					Index const elemHead = res[dev].offset[blockIdx];
					std::copy(blocks[b][0].begin(), blocks[b][0].end(), &res[dev].elems[elemHead]);
					res[dev].period[blockIdx]   = blocks[b].size();
					res[dev].offset[++blockIdx] = elemHead + blocks[b][0].size();

					assert(int(blocks[b][0].size()) == table[dev](j, 0));
				}
			}
			{
				int const   j     = blockSizes.rows() - 1;
				Index const cHead = head[j];
				head[j] += table[dev](j, 1);
				for(Index b = cHead; b < head[j]; ++b) {
					Index const elemHead        = res[dev].offset[blockIdx];
					res[dev].elems[elemHead]    = blocks[0][b][0];
					res[dev].period[blockIdx]   = blocks[0][b].size();
					res[dev].offset[++blockIdx] = elemHead + 1;
				}
			}
		}
		// (END) 3. Distribute the elements of the blocks to the devices

		return std::make_pair(res, blockSizes);
	}

}  // namespace StatMech
