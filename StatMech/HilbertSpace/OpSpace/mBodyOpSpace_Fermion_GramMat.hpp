#pragma once

#include "mBodyOpSpace_Fermion.hpp"
#include "../Algorithm/IntegerComposition.hpp"

template<typename Scalar_>
__host__ std::vector<std::vector<std::vector<Index>>> blocksInGramMat(
    mBodyOpSpace<ManyBodyFermionSpace, Scalar_> const& opSpace) {
	std::cout << __PRETTY_FUNCTION__ << std::endl;
	int const max = opSpace.maxOnSite();

	std::vector< std::vector<std::vector<Index>> > blocks(1);
	Eigen::ArrayX<bool> calculated = Eigen::ArrayX<bool>::Constant(opSpace.dim(), false);
#pragma omp parallel
	{
		Eigen::ArrayXi invConfig(opSpace.sysSize());
		Eigen::ArrayXi variation(opSpace.sysSize());
		Eigen::ArrayXi zeroPos(opSpace.sysSize());
#pragma omp for schedule(dynamic, 10)
		for(auto j = 0; j < opSpace.dim(); ++j) {
			if(calculated(j)) continue;
			calculated(j) = true;

			int nZeros = 0;
			// zeroPos    = Eigen::ArrayXi::Zero(opSpace.sysSize());
			int x = opSpace.m();
			{
				auto& config = invConfig;
				opSpace.ordinal_to_config(config, j);
				// std::cout << "\t" << config.transpose() << std::endl;
				for(auto pos = 0; pos < opSpace.sysSize(); ++pos) {
					int const var = config(pos) / (max + 1) - config(pos) % (max + 1);
					if(var == 0) {
						zeroPos(nZeros++) = pos;
						variation(pos)    = 0;
					}
					else if(var > 0) {
						x -= var;
						variation(pos) = var * (max + 1);
					}
					else { variation(pos) = -var; }
				}
			}
			// std::cout << "#\t nZeros = " << nZeros << std::endl;

			int period = 0;
			for(auto trans = 1; trans <= opSpace.sysSize(); ++trans) {
				auto const& translated = variation.NullaryExpr(variation.size(), [&](Index j) {
					return variation((j + trans) % variation.size());
				});
				if(translated.isApprox(variation)) {
					period = trans;
					break;
				}
			}
			// std::cout << "# period = " << period << std::endl;

			IntegerComposition const evenOpConf(x, nZeros, max);
			if(evenOpConf.dim() <= 1) {
				std::vector<Index> indices(period);
				auto&            config = invConfig;
				opSpace.ordinal_to_config(config, j);

				bool flag = false;
				for(auto trans = 0; trans < period; ++trans) {
					auto const& translated = config.NullaryExpr(config.size(), [&](Index pos) {
						return config((pos + trans) % config.size());
					});
					// std::cout << translated.transpose() << std::endl;
					auto const k = opSpace.config_to_ordinal(translated);
					if(k < j) {
						flag = true;
						break;
					}
					calculated(k)  = true;
					indices[trans] = k;
					// idMin          = (idMin < k ? idMin : k);
					// std::cout << "# ↑ trans = " << trans << std::endl;
				}
				// std::cout << "# " << __FILE__ << ":(" << __LINE__ << ")" << std::endl;
				if(flag) continue;
					// if(idMin != j) continue;

					// std::cout << "#(2) " << __FILE__ << ":(" << __LINE__ << ")" << std::endl;
#pragma omp critical
				blocks[0].push_back(indices);
				continue;
			}

			// zeroPos.conservativeResize(nZeros == 0 ? 1 : nZeros);
			invConfig = Eigen::ArrayXi::Zero(invConfig.size());
			class wrapper {
				private:
					Eigen::ArrayXi&       m_array;
					Eigen::ArrayXi const& m_index;

				public:
					wrapper(Eigen::ArrayXi& array, Eigen::ArrayXi const& index)
					    : m_array{array}, m_index{index} {};
					int& operator()(int idx) {
						assert(idx < m_index.size());
						assert(m_index(idx) < m_array.size());
						return m_array(m_index(idx));
					}
					Index size() { return m_index.size(); }
			};
			wrapper invConfig_wrapper(invConfig, zeroPos);

			// #pragma omp critical
			// 			std::cout << "#\t opSpace.maxOnSite() = " << opSpace.maxOnSite()
			// 			          << ", opSpace.sysSize() = " << opSpace.sysSize() << ", nZeros = " << nZeros
			// 			          << ", evenOpConf.value() = " << evenOpConf.value()
			// 			          << ", evenOpConf.length() = " << evenOpConf.length()
			// 			          << ", evenOpConf.max() = " << evenOpConf.max()
			// 			          << ", evenOpConf.dim() = " << evenOpConf.dim() << std::endl;
			// std::cout << "# zeroPos: " << zeroPos.transpose() << std::endl;

			// evenOpConf.ordinal_to_config(invConfig_wrapper, 0);
			// std::cout << "# invConfig: " << invConfig.transpose() << std::endl;

			std::vector<std::vector<Index>> indices(period);
			for(auto trans = 0; trans < period; ++trans) indices[trans].resize(evenOpConf.dim());

			// int idMin = opSpace.dim();
			bool flag = false;
			for(auto beta = 0; beta < evenOpConf.dim(); ++beta) {
				evenOpConf.ordinal_to_config(invConfig_wrapper, beta);
				auto const& config
				    = variation + invConfig.unaryExpr([&](const int x) { return x * (max + 2); });
				// std::cout << "#\t config.size() = " << config.size() << std::endl;

				for(auto trans = 0; trans < period; ++trans) {
					auto const& translated = config.NullaryExpr(config.size(), [&](Index j) {
						return config((j + trans) % config.size());
					});
					auto const  k          = opSpace.config_to_ordinal(translated);
					if(k < j) {
						flag = true;
						break;
					}
					calculated(k)        = true;
					indices[trans][beta] = k;
					// idMin                = (idMin < k ? idMin : k);
				}
				if(flag) break;
			}
			if(flag) continue;
			// if(idMin != j) continue;

#pragma omp critical
			blocks.push_back(indices);
		}
	}

	// std::cout << "# (END) " << __PRETTY_FUNCTION__ << std::endl;
	return blocks;
}