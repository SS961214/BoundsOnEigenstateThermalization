#define EIGEN_DEFAULT_IO_FORMAT \
	Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", ";\n", " ", "", "[", ";\n]")

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <complex>
#include <cuda/std/complex>

using Eigen::Index;
using Scalar = std::complex<double>;

int main(int argc, char** argv) {
	if(argc != 2) {
		std::cerr << "Usage: 0.(This) 1.(SampleNo)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	constexpr Index dimLoc   = 2;
	int const       sampleNo = std::atoi(argv[1]);

	Eigen::MatrixX<Scalar>           locH;
	constexpr int                    seed = 0;
	std::mt19937                     mt(seed);
	std::normal_distribution<double> Gaussian(0.0, 1.0);
	for(int j = 0; j < sampleNo; ++j) {
		locH = Eigen::MatrixX<Scalar>::NullaryExpr(
		    dimLoc * dimLoc, dimLoc * dimLoc, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
	}
	locH = Eigen::MatrixX<Scalar>::NullaryExpr(
	    dimLoc * dimLoc, dimLoc * dimLoc, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
	locH = (locH + locH.adjoint()).eval() / 2.0;

	std::cout << locH.cast<std::complex<double>>() << std::endl;

	return EXIT_SUCCESS;
}