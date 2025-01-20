#include "tests.hpp"
#include "ManyBodySpaceBase_test.hpp"
#include "ManyBodyFermionSpace.hpp"
#include <iostream>

TEST_CASE("ManyBodyFermionSpace", "test") {
	constexpr int         NMin  = 4;
	constexpr int         NMax  = 11;
	auto const sysSize = [](int N) { return 2*N; }; // half filling
	int const LMax = sysSize(NMax);
	Eigen::ArrayXX<Index> binom = Eigen::ArrayXX<Index>::Zero(LMax + 1, NMax + 1);
	binom(0, 0)                 = 1;
	for(auto j = 1; j < binom.rows(); ++j) {
		binom(j, 0) = 1;
		for(auto m = 1; m <= std::min(j, NMax); ++m)
			binom(j, m) = binom(j - 1, m - 1) + binom(j - 1, m);
	}

	// test for class ManyBodyFermionSpace
	{
		// Default constructor
		ManyBodyFermionSpace mbSpace;
		test_ManyBodySpaceBase(mbSpace, 0);
	}
	// test Constructor1
	for(auto N = NMin; N <= NMax; ++N) {
		int const L = sysSize(N);
		std::cout << "# L = " << L << ", N = " << N << std::endl;
		ManyBodyFermionSpace const mbSpace(L, N);
		REQUIRE(mbSpace.dim() == binom(L, N));
		test_ManyBodySpaceBase(mbSpace, L);
	}
}