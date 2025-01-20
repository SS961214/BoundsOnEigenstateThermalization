#include "tests.hpp"
#include "../SubSpace_test.hpp"
#include "TransParitySector.hpp"
#include "ManyBodySpinSpace.hpp"
#include <complex>
#include <iostream>

using Scalar = std::complex<double>;

TEST_CASE("TransParitySector_Spin", "test") {
	constexpr int dLoc = 2;
	constexpr int LMax = 20;

	// test for class ManyBodySpinSpace
	using OpSpace = ManyBodySpinSpace;
	{
		// Default constructor
		TransParitySector<OpSpace, Scalar> sector;
	}
	for(int parity = 1; parity >= -1; parity -= 2) {
		OpSpace                                      mbSpace;
		TransParitySector<decltype(mbSpace), Scalar> sector(parity, mbSpace);
		test_SubSpace(sector);
	}
	// test Constructor1
	for(int L = 0; L <= LMax; ++L)
		for(int parity = 1; parity >= -1; parity -= 2) {
			OpSpace const                                              mbSpace(L, dLoc);
			TransParitySector<std::decay_t<decltype(mbSpace)>, Scalar> sector(parity, mbSpace);
			std::cout << "L = " << L << ", S = " << (dLoc - 1) / 2.0 << ", parity = " << parity
			          << ": dim = " << mbSpace.dim() << ", sectorDim = " << sector.dim()
			          << std::endl;
			test_SubSpace(sector);

			if(L <= 1) continue;
			if(sector.dim() <= 0) continue;

			{
				Eigen::SparseMatrix<Scalar> shiftOp(mbSpace.dim(), mbSpace.dim());
				for(int in = 0; in < mbSpace.dim(); ++in) {
					auto const out = mbSpace.translate(in, 1);
					assert(out < mbSpace.dim());
					shiftOp.insert(out, in) = 1.0;
				}
				shiftOp.makeCompressed();
				double const diff = (shiftOp * sector.basis() - sector.basis()).norm();
				REQUIRE(diff < 1.0E-14);
				std::cout << "# Checked the invatiance under the translation operator: diff = "
				          << diff << std::endl;
			}
			{
				Eigen::SparseMatrix<Scalar> reflectOp(mbSpace.dim(), mbSpace.dim());
				for(int in = 0; in < mbSpace.dim(); ++in) {
					auto const out = mbSpace.reverse(in);
					assert(out < mbSpace.dim());
					reflectOp.insert(out, in) = 1.0;
				}
				reflectOp.makeCompressed();
				double const diff
				    = (reflectOp * sector.basis() - sector.parity() * sector.basis()).norm();
				REQUIRE(diff < 1.0E-14);
				std::cout << "# Checked the invatiance under the parity operator: diff = " << diff
				          << std::endl;
			}

			std::cout << "# " << std::endl;
		}
}