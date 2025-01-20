#ifdef NEBUG
	#undef NDEBUG
#endif

#include "constructGlobalOp.hpp"
#include <HilbertSpace>
#include <unsupported/Eigen/KroneckerProduct>
#include <random>
#include <iostream>

using Scalar = std::complex<double>;

int main(int argc, char* argv[]) {
	if(argc != 3) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(seed)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	constexpr double precision = 1e-10;
	constexpr int    dLoc      = 2;
	int const        L         = std::atoi(argv[1]);
	int const        seed      = std::atoi(argv[2]);
	std::cout << "#(ManyBodySpinSpace) L = " << L << ", dLoc = " << dLoc << ", seed = " << seed
	          << std::endl;

	std::mt19937                     mt(seed);
	std::normal_distribution<double> Gaussian(0.0, 1.0);
	Eigen::MatrixX<Scalar>           locH = Eigen::MatrixX<Scalar>::NullaryExpr(
        dLoc * dLoc, dLoc * dLoc, [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
	locH = (locH + locH.adjoint()).eval() / 2.0;

	ManyBodySpinSpace const     mbSpace(L, dLoc);
	Eigen::SparseMatrix<Scalar> I(mbSpace.dim() / locH.rows(), mbSpace.dim() / locH.cols());
	I.setIdentity();
	for(int momentum = 0; momentum < L; ++momentum) {
		TransSector<ManyBodySpinSpace, Scalar> const subSpace(momentum, mbSpace);

		Eigen::MatrixX<Scalar> const      H    = construct_globalOp(locH, subSpace);
		Eigen::SparseMatrix<Scalar> const Htot = Eigen::kroneckerProduct(locH, I);
		Eigen::MatrixX<Scalar> const Href = subSpace.basis().adjoint() * Htot * subSpace.basis();

		double const diff = (H - Href).cwiseAbs().maxCoeff();
		std::cout << "# k = " << momentum << ", diff = " << diff << std::endl;
		if(diff > precision) {
			std::cerr << "Error: diff = " << diff << std::endl;
			std::exit(EXIT_FAILURE);
		}
		assert(diff < precision);
	}

	return EXIT_SUCCESS;
}