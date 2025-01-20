#include "tests.hpp"
#include "mBodyOpSpace_Spin.hpp"
#include "../ManyBodySpaceBase_test.hpp"
#include "../OpSpaceBase_test.hpp"
#include <iostream>
#include <complex>

using Scalar = std::complex<double>;

TEST_CASE("mBodyOpSpace_Spin", "test") {
	constexpr int LMax = 18;
	constexpr int LMin = 6;
	constexpr int dLoc = 2;

	// test for class ManyBodySpinSpace
	{
		// Default constructor
		ManyBodySpinSpace                       mbSpace;
		mBodyOpSpace<decltype(mbSpace), Scalar> opSpace;
		test_ManyBodySpaceBase(opSpace, 0);
		// test_OpSpace(opSpace);
	}
	{
		// test Constructor1
		ManyBodySpinSpace                       mbSpace(0, dLoc);
		mBodyOpSpace<decltype(mbSpace), Scalar> opSpace(0, mbSpace);
		test_ManyBodySpaceBase(opSpace, 0);
		// test_OpSpace(opSpace);
		for(auto sysSize = LMin; sysSize <= LMax; ++sysSize) {
			ManyBodySpinSpace mbSpace(sysSize, dLoc);
			for(auto m = 1; m <= sysSize; ++m) {
				std::cout << "sysSize = " << sysSize << ", m = " << m << std::endl;
				mBodyOpSpace<decltype(mbSpace), Scalar> opSpace(m, mbSpace);
				if(m > 4 && opSpace.dim() > 100000000) {
					std::cout << "opSpace.dim() = " << opSpace.dim()
					          << " is so large. Skipping test..." << std::endl;
					continue;
				}
				test_ManyBodySpaceBase(opSpace, sysSize);
				// test_OpSpace(opSpace);
			}
		}
	}
}