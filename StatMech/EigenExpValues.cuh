#pragma once

#include "EigenExpValues.hpp"
#include <MatrixGPU>
#include <complex>

template<class Matrix1_, class Matrix2_>
Eigen::ArrayXd EigenExpValues(GPU::MatrixGPU<Matrix1_> const& obs,
                              GPU::MatrixGPU<Matrix2_> const& eigVecs) {
	using Scalar = typename Matrix2_::Scalar;
	GPU::MatrixGPU<Eigen::MatrixX<Scalar>> temp(obs.rows(), eigVecs.cols());
	magma_hemm(MagmaLeft, MagmaLower, obs.rows(), eigVecs.cols(), Scalar(1.0, 0), obs.data(),
	           obs.LD(), eigVecs.data(), eigVecs.LD(), Scalar(0), temp.data(), temp.LD(),
	           GPU::MAGMA::queue());

	// Eigen::MatrixX<Scalar> mat;
	// temp.copyTo(mat);
	// std::cout << mat.template cast< std::complex<double> >() << std::endl;

	Eigen::ArrayXd res(eigVecs.cols());
#pragma omp parallel for
	for(Index alpha = 0; alpha < eigVecs.cols(); ++alpha) {
		res(alpha) = real(magma_dotc(eigVecs.rows(), eigVecs.data() + alpha * eigVecs.LD(), 1,
		                             temp.data() + alpha * temp.LD(), 1,
		                             GPU::MAGMA::queue(omp_get_thread_num())));
	}

	// std::cout << res << std::endl;
	return res;
}
