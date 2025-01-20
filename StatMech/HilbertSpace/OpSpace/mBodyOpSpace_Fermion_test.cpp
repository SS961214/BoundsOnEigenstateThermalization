#include "tests.hpp"
#include "mBodyOpSpace_Fermion.hpp"
#include "../ManyBodySpaceBase_test.hpp"
#include "../OpSpaceBase_test.hpp"
#include <iostream>
#include <complex>

using Scalar = std::complex<double>;

TEST_CASE("mBodyOpSpace_Fermion", "test") {
	constexpr int         NMax  = 16;
	constexpr int         LMax  = 16;
	Eigen::ArrayXX<Index> binom = Eigen::ArrayXX<Index>::Zero(LMax + 1, NMax + 1);
	binom(0, 0)                 = 1;
	for(auto j = 1; j < binom.rows(); ++j) {
		binom(j, 0) = 1;
		for(auto m = 1; m <= j; ++m) binom(j, m) = binom(j - 1, m - 1) + binom(j - 1, m);
	}

	// test for class mBodyOpSpace<ManyBodyFermionSpace>
	{
		// Default constructor
		ManyBodyFermionSpace                    mbSpace;
		mBodyOpSpace<decltype(mbSpace), Scalar> opSpace;
		test_ManyBodySpaceBase(opSpace, 0);
		test_OpSpace(opSpace);
	}
	{
		for(auto L = 2; L <= LMax; ++L)
			for(auto N = 1; N <= std::min(NMax, L); ++N) {
				ManyBodyFermionSpace mbSpace(L, N);
				for(auto m = 1; m <= N; ++m) {
					std::cout << "# L = " << L << ", N = " << N << ", m = " << m << std::endl;
					mBodyOpSpace<decltype(mbSpace), Scalar> opSpace(m, mbSpace);
					auto const                              n = std::min(m, L - N);
					REQUIRE(opSpace.dim() == binom(L, n) * binom(L, n));
					test_ManyBodySpaceBase(opSpace, L);
					test_OpSpace(opSpace);
				}
			}
	}
}