#include "tests.hpp"
#include "../SubSpace_test.hpp"
#include "TransSector.hpp"
#include "ManyBodySpinSpace.hpp"
#include <complex>
#include <iostream>

using Scalar = std::complex<double>;

TEST_CASE("TransSector_Spin", "test") {
	constexpr int   k    = 0;
	constexpr Index LMax = 20;
	constexpr Index dLoc = 2;

	// test for class ManyBodySpinSpace
	{
		// Default constructor
		TransSector<ManyBodySpinSpace, Scalar> transSector;
	}
	{
		ManyBodySpinSpace                      mbSpace;
		TransSector<decltype(mbSpace), Scalar> transSector(k, mbSpace);
		test_SubSpace(transSector);
	}
	{
		// test Constructor1
		ManyBodySpinSpace                      mbSpace(0, dLoc);
		TransSector<decltype(mbSpace), Scalar> transSector(k, mbSpace);
		test_SubSpace(transSector);

		for(auto sysSize = 1; sysSize <= LMax; ++sysSize) {
			std::cout << "##\t L = " << sysSize << std::endl;
			ManyBodySpinSpace                      mbSpace(sysSize, dLoc);
			TransSector<decltype(mbSpace), Scalar> transSector(k, mbSpace);
			test_SubSpace(transSector);
		}
	}
}