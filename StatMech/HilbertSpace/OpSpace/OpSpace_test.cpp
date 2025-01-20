#include "tests.hpp"
#include "OpSpaceBase_test.hpp"
#include "OpSpace.hpp"
#include <iostream>
#include <complex>

using Scalar = std::complex<double>;

TEST_CASE("OpSpace", "test") {
	using Matrix       = Eigen::MatrixX<Scalar>;
	constexpr Scalar I = Scalar(0, 1);

	{
		for(Index dim = 2; dim < 64; dim *= 2) {
			std::cout << "dim = " << dim << std::endl;
			HilbertSpace<int> baseSpace(dim);
			OpSpace<Scalar>   opSpace(baseSpace);
			test_OpSpace(opSpace);
			for(auto rep = 0; rep != 100; ++rep)
				REQUIRE(Matrix(opSpace.basisOp(0)) == Eigen::MatrixX<Scalar>::Identity(dim, dim));
		}
	}
	{
		Index             dim = 3;
		HilbertSpace<int> baseSpace(dim);
		OpSpace<Scalar>   opSpace(baseSpace);
		test_OpSpace(opSpace);

		for(auto rep = 0; rep != 100; ++rep)
			REQUIRE(Matrix(opSpace.basisOp(0)) == Eigen::Matrix3<Scalar>::Identity());

		Eigen::Matrix3<Scalar> op;
		op << sqrt(2), 0, 0, 0, -1 / sqrt(2), 0, 0, 0, -1 / sqrt(2);
		for(auto rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(1)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 0, 0, 0, sqrt(1.5), 0, 0, 0, -sqrt(1.5);
		for(auto rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(2)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 1, 0, 1, 0, 0, 0, 0, 0;
		for(auto rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(3)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 0, 1, 0, 0, 0, 1, 0, 0;
		for(auto rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(4)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 0, 0, 0, 0, 1, 0, 1, 0;
		for(auto rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(5)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, -I, 0, I, 0, 0, 0, 0, 0;
		for(auto rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(6)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 0, -I, 0, 0, 0, I, 0, 0;
		for(auto rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(7)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 0, 0, 0, 0, -I, 0, I, 0;
		for(auto rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(8)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
	}
}