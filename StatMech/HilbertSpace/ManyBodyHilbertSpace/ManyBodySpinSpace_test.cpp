#include "tests.hpp"
#include "ManyBodySpaceBase_test.hpp"
#include "ManyBodySpinSpace.hpp"
#include <iostream>

TEST_CASE("ManyBodySpinSpace", "test") {
	constexpr int dLoc = 2;

	// test for class ManyBodySpinSpace
	{
		// Default constructor
		ManyBodySpinSpace mbSpace;
		test_ManyBodySpaceBase(mbSpace, 0);
	}
	{
		// test Constructor1
		ManyBodySpinSpace mbSpace(0, dLoc);
		test_ManyBodySpaceBase(mbSpace, 0);
		for(auto sysSize = 1; sysSize <= 20; ++sysSize) {
			std::cout << "# L = " << sysSize << std::endl;
			ManyBodySpinSpace mbSpace(sysSize, dLoc);
			test_ManyBodySpaceBase(mbSpace, sysSize);
		}
	}
}