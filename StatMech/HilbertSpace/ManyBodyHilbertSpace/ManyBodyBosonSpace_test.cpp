#include "tests.hpp"
#include "ManyBodySpaceBase_test.hpp"
#include "ManyBodyBosonSpace.hpp"
#include <iostream>

TEST_CASE("ManyBodyBosonSpace", "test") {
	constexpr int         NMin  = 4;
	constexpr int         NMax  = 11;
	auto const sysSize = [](int N) { return N; }; // unit filling
	int const LMax = sysSize(NMax);
	Eigen::ArrayXX<Index> binom = Eigen::ArrayXX<Index>::Zero(NMax + LMax + 1, NMax + LMax + 1);
	binom(0, 0)                 = 1;
	for(auto j = 1; j < binom.rows(); ++j) {
		binom(j, 0) = 1;
		for(auto m = 1; m <= j; ++m) binom(j, m) = binom(j - 1, m - 1) + binom(j - 1, m);
	}

	// test for class ManyBodyBosonSpace
	{
		// Default constructor
		ManyBodyBosonSpace mbSpace;
		test_ManyBodySpaceBase(mbSpace, 0);
	}
	// test Constructor1
	for(auto N = NMin; N <= NMax; ++N) {
		int const L = sysSize(N);
		{
			ManyBodyBosonSpace mbSpace(L, N);
			REQUIRE(mbSpace.dim() == binom(L + N - 1, N));
			test_ManyBodySpaceBase(mbSpace, L);
		}
		{
			ManyBodyBosonSpace mbSpace(L, N, 1);
			REQUIRE(mbSpace.dim() == binom(L, N));
			test_ManyBodySpaceBase(mbSpace, L);
		}
	}
}