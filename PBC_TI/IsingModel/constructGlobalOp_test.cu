#include <cuda/std/complex>
#include <iostream>
template<typename Real_>
std::ostream& operator<<(std::ostream& os, cuda::std::complex<Real_> const& z) {
	os << "(" << double(z.real()) << ", " << double(z.imag()) << ")" << std::endl;
	return os;
};
#include <catch2/catch_test_macros.hpp>
#include "constructGlobalOp.hpp"
#include <HilbertSpace/ManyBodyHilbertSpace/ManyBodySpinSpace.hpp>
#include <HilbertSpace/ManyBodyHilbertSpace/TransSector.hpp>

#include <random>

using Scalar = cuda::std::complex<double>;
std::random_device               seed_gen;
std::mt19937                     mt(seed_gen());
std::normal_distribution<double> Gaussian(0.0, 1.0);

TEST_CASE("constructGlobalOp_onGPU", "test") {
	static_assert(std::is_convertible_v<magma_int_t, Index>);
	constexpr double precision = 1.0E-12;
	constexpr int    dimLoc    = 2;
	int const        dLocOp    = dimLoc * dimLoc;

	Eigen::MatrixX<Scalar> locOp = Eigen::MatrixX<Scalar>::NullaryExpr(
	    dLocOp, dLocOp, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
	locOp = (locOp + locOp.adjoint()).eval() / 2.0;
	REQUIRE((locOp - locOp.adjoint()).cwiseAbs().maxCoeff() < precision);

	constexpr int LMin = 2;
	constexpr int LMax = 18;
	for(int L = LMin; L <= LMax; ++L, std::cout << std::endl)
		for(int momentum = 0; momentum < L; ++momentum) {
			std::cout << "# L = " << L << ", momentum = " << momentum << std::endl;
			ManyBodySpinSpace                      mbSpace(L, dimLoc);
			TransSector<ManyBodySpinSpace, Scalar> sector(momentum, mbSpace);

			Eigen::MatrixX<Scalar>                   globOp = construct_globalOp(locOp, sector);
			GPU::MatrixGPU< Eigen::MatrixX<Scalar> > dGlobOp
			    = construct_globalOp_onGPU(locOp, sector);
			Eigen::MatrixX<Scalar> hGlobOp(dGlobOp.rows(), dGlobOp.cols());
			dGlobOp.copyTo(hGlobOp);

			// std::cout << globOp.cast<std::complex<double>>() << "\n" << std::endl;
			// std::cout << hGlobOp.cast<std::complex<double>>() << std::endl;
			REQUIRE((globOp - hGlobOp).cwiseAbs().maxCoeff() < precision);
			REQUIRE((hGlobOp - hGlobOp.adjoint()).cwiseAbs().maxCoeff() < precision);
			// assert((globOp - globOp.adjoint()).cwiseAbs().maxCoeff() < precision);
		}
}