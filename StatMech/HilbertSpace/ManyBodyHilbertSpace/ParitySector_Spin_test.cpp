#include "tests.hpp"
#include "../SubSpace_test.hpp"
#include "ParitySector.hpp"
#include "ManyBodySpinSpace.hpp"
#include <complex>
#include <iostream>

using Scalar = std::complex<double>;

TEST_CASE("ParitySector_Spin", "test") {
	constexpr Index LMax = 20;
	constexpr Index dLoc = 2;

	// test for class ManyBodySpinSpace
	{
		// Default constructor
		ParitySector<ManyBodySpinSpace, Scalar> paritySector;
	}
	{
		ManyBodySpinSpace mbSpace;
		Index             dimTot = 0;
		for(auto parity = 1; parity >= -1; parity -= 2) {
			ParitySector<decltype(mbSpace), Scalar> paritySector(parity, mbSpace);
			test_SubSpace(paritySector);
			dimTot += paritySector.dim();
		}
		REQUIRE(dimTot == mbSpace.dim());
	}
	{
		// test Constructor1
		ManyBodySpinSpace mbSpace(0, dLoc);
		Index             dimTot = 0;
		for(auto parity = 1; parity >= -1; parity -= 2) {
			ParitySector<decltype(mbSpace), Scalar> paritySector(parity, mbSpace);
			test_SubSpace(paritySector);
			dimTot += paritySector.dim();
		}
		REQUIRE(dimTot == mbSpace.dim());

		for(Index sysSize = 1; sysSize <= LMax; ++sysSize) {
			ManyBodySpinSpace mbSpace(sysSize, dLoc);
			Index             dimTot = 0;
			for(auto parity = 1; parity >= -1; parity -= 2) {
				std::cout << "##\t L = " << sysSize << ", parity = " << parity << std::endl;
				ParitySector<decltype(mbSpace), Scalar> paritySector(parity, mbSpace);
				test_SubSpace(paritySector);
				dimTot += paritySector.dim();
			}
			REQUIRE(dimTot == mbSpace.dim());
		}
	}
}