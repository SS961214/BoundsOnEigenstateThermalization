#include "tests.hpp"
#include "IntegerComposition.hpp"
#include <Eigen/Dense>
#include <iostream>

void test_IntegerComposition(Index N, Index L, Index Max, Index dim) {
	std::cout << "# " << __PRETTY_FUNCTION__ << std::endl;
	std::cout << "##\t (N, Length, Max) = (" << N << ", " << L << ", " << Max << ")" << std::endl;
	IntegerComposition iComp(N, L, Max);
	std::cout << "##\t iComp.dim() = " << iComp.dim() << std::endl;
	REQUIRE(iComp.value() == N);
	REQUIRE(iComp.length() == L);
	REQUIRE(iComp.max() == Max);
	REQUIRE(iComp.dim() == dim);
#pragma omp parallel
	{
		Eigen::ArrayX<Index> config(iComp.length());
#pragma omp for
		for(Index ordinal = 0; ordinal < iComp.dim(); ++ordinal) {
			iComp.ordinal_to_config(config, ordinal);
			REQUIRE(iComp.value() == static_cast<Index>(config.sum()));
			REQUIRE(iComp.max() >= static_cast<Index>(config.maxCoeff()));
			REQUIRE(iComp.config_to_ordinal(config) == ordinal);
			for(auto pos = 0; pos < iComp.length(); ++pos) {
				REQUIRE(iComp.locNumber(ordinal, pos) == config(pos));
			}
		}
	}
	std::cout << "# Passed bijectiveness test.\n#" << std::endl;

	Eigen::ArrayXi state_to_rep(iComp.dim());
#pragma omp parallel for
	for(Index j = 0; j < iComp.dim(); ++j) {
		state_to_rep(j) = j;
		for(auto trans = 1; trans != iComp.length(); ++trans) {
			int const translated = iComp.translate(j, trans);
			if(j == translated) break;
			state_to_rep(j) = std::min(state_to_rep(j), translated);
		}
	}
	std::sort(state_to_rep.begin(), state_to_rep.end());
	Index transEqDim = std::unique(state_to_rep.begin(), state_to_rep.end()) - state_to_rep.begin();
	std::cout << "##\t transEqDim = " << transEqDim << std::endl;

	Eigen::ArrayXi appeared = Eigen::ArrayXi::Zero(iComp.dim());
#pragma omp parallel for
	for(auto j = 0; j < transEqDim; ++j) {
		auto const rep = state_to_rep(j);
#pragma omp atomic
		++appeared(rep);
		for(auto trans = 1; trans < iComp.length(); ++trans) {
			auto const translated = iComp.translate(rep, trans);
			if(translated == rep) break;
#pragma omp atomic
			++appeared(translated);
		}
	}
#pragma omp parallel for
	for(auto j = 0; j < appeared.size(); ++j) { REQUIRE(appeared(j) == 1); }
	std::cout << "# Passed tests for translation operations.\n#" << std::endl;

#pragma omp parallel
	{
		Eigen::ArrayX<Index> config(iComp.length());
#pragma omp for
		for(Index ordinal = 0; ordinal < iComp.dim(); ++ordinal) {
			iComp.ordinal_to_config(config, ordinal);
			auto const reversed = iComp.config_to_ordinal(config.reverse());
			REQUIRE(iComp.reverse(ordinal) == reversed);
		}
	}
	std::cout << "# Passed tests for reverse operations.\n#" << std::endl;
	std::cout << std::endl;
}

TEST_CASE("IntegerComposition", "test") {
	int                   LMax  = 24;
	int                   NMax  = 12;
	Eigen::ArrayXX<Index> binom = Eigen::ArrayXX<Index>::Zero(NMax + LMax + 1, NMax + LMax + 1);
	binom(0, 0)                 = 1;
	for(auto j = 1; j < binom.rows(); ++j) {
		binom(j, 0) = 1;
		for(auto m = 1; m <= j; ++m) binom(j, m) = binom(j - 1, m - 1) + binom(j - 1, m);
	}

	{
		test_IntegerComposition(0, 0, 0, 0);

		// Dimension check for hard-core boson case
		for(auto N = 1; N <= NMax; ++N)
			for(auto L = N; L <= LMax; ++L) { test_IntegerComposition(N, L, 1, binom(L, N)); }

		LMax = 18;
		NMax = 9;
		// Dimension check for soft-core boson case
		for(auto N = 1; N <= NMax; ++N)
			for(auto L = N; L <= LMax; ++L) {
				test_IntegerComposition(N, L, N, binom(N + L - 1, N));
			}
	}
}