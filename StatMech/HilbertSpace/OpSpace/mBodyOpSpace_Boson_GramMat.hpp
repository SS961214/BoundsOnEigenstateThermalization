#pragma once

#include "mBodyOpSpace_Boson.hpp"

template<typename Scalar_>
__host__ std::vector<std::vector<std::vector<Index>>> blocksInGramMat(
    mBodyOpSpace<ManyBodyBosonSpace, Scalar_> const& opSpace) {
	int const m = opSpace.m();

	std::vector< std::vector<std::vector<Index>> > blocks(1);
	Eigen::ArrayX<bool> calculated = Eigen::ArrayX<bool>::Constant(opSpace.dim(), false);
#pragma omp parallel
	{
		Eigen::ArrayXi invConfig(opSpace.sysSize());
		Eigen::ArrayXi variation(opSpace.sysSize());
#pragma omp for schedule(dynamic, 10)
		for(auto j = 0; j < opSpace.dim(); ++j) {
			if(calculated(j)) continue;
			calculated(j) = true;

			int x = opSpace.m();
			{
				auto& config = invConfig;
				opSpace.ordinal_to_config(config, j);
				for(auto pos = 0; pos < opSpace.sysSize(); ++pos) {
					int const var = config(pos) / (m + 1) - config(pos) % (m + 1);
					if(var > 0) {
						x -= var;
						variation(pos) = var * (m + 1);
					}
					else { variation(pos) = -var; }
				}
			}
			IntegerComposition const evenOpConf(x, opSpace.sysSize(), x);

			evenOpConf.ordinal_to_config(invConfig, 0);
			int idMin;
			{
				auto const& config
				    = variation + invConfig.unaryExpr([&](const int x) { return x * (m + 2); });
				idMin = opSpace.config_to_ordinal(config);
			}
			int period = 0;
			for(auto trans = 1; trans <= opSpace.sysSize(); ++trans) {
				auto const& translated = variation.NullaryExpr(variation.size(), [&](Index j) {
					return variation((j + trans) % variation.size());
				});
				if(translated.isApprox(variation)) {
					period = trans;
					break;
				}
				auto const& config
				    = translated + invConfig.unaryExpr([&](const int x) { return x * (m + 2); });
				int const k = opSpace.config_to_ordinal(config);
				idMin       = (idMin < k ? idMin : k);
			}
			if(idMin != j) continue;

			std::vector<std::vector<Index>> indices;
			if(evenOpConf.dim() <= 1) {
				indices.resize(1);
				indices[0].resize(period);
				evenOpConf.ordinal_to_config(invConfig, 0);
				for(auto trans = 0; trans < period; ++trans) {
					auto const& translated = variation.NullaryExpr(variation.size(), [&](Index j) {
						return variation((j + trans) % variation.size());
					});
					auto const& config = translated + invConfig.unaryExpr([&](const int x) { return x * (m + 2); });
					auto const  k          = opSpace.config_to_ordinal(config);
					calculated(k)          = true;
					indices[0][trans]      = k;
				}
#pragma omp critical
				blocks[0].push_back(indices[0]);
				continue;
			}

			indices.resize(period);
			for(auto trans = 0; trans < period; ++trans) indices[trans].resize(evenOpConf.dim());

			for(auto beta = 0; beta < evenOpConf.dim(); ++beta) {
				evenOpConf.ordinal_to_config(invConfig, beta);
				for(auto trans = 0; trans < period; ++trans) {
					auto const& translated = variation.NullaryExpr(variation.size(), [&](Index j) {
						return variation((j + trans) % variation.size());
					});
					auto const& config     = translated + invConfig.unaryExpr([&](const int x) {
                        return x * (m + 2);
                    });
					auto const  k          = opSpace.config_to_ordinal(config);
					calculated(k)          = true;
					indices[trans][beta]   = k;
				}
			}

#pragma omp critical
			blocks.push_back(indices);
		}
	}

	return blocks;
}