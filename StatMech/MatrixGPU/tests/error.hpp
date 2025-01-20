#pragma once
#include "debug.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <mkl.h>

namespace GPU::internal {
	template<typename Scalar_, typename RealScalar_>
	double diagError(Eigen::MatrixX<Scalar_> const& mat, Eigen::MatrixX<Scalar_> const& eigvecs,
	                 Eigen::VectorX<RealScalar_> const& eigvals) {
		using RealScalar = typename Eigen::NumTraits<Scalar_>::Real;

		Eigen::MatrixX<Scalar_> res = eigvecs * eigvals.asDiagonal();
		MKL_INT const           dim = mat.rows();
		if constexpr(std::is_same_v<Scalar_, std::complex<float>>) {
			MKL_Complex8 const alpha = {1.0, 0};
			MKL_Complex8 const beta  = {-1.0, 0};
			chemm("L", "U", &dim, &dim, &alpha, reinterpret_cast<MKL_Complex8 const*>(mat.data()),
			      &dim, reinterpret_cast<MKL_Complex8 const*>(eigvecs.data()), &dim, &beta,
			      reinterpret_cast<MKL_Complex8*>(res.data()), &dim);
		}
		else {
			MKL_Complex16 const alpha = {1.0, 0};
			MKL_Complex16 const beta  = {-1.0, 0};
			zhemm("L", "U", &dim, &dim, &alpha, reinterpret_cast<MKL_Complex16 const*>(mat.data()),
			      &dim, reinterpret_cast<MKL_Complex16 const*>(eigvecs.data()), &dim, &beta,
			      reinterpret_cast<MKL_Complex16*>(res.data()), &dim);
		}

		double norm = 0;
#pragma omp parallel for reduction(+ : norm)
		for(Eigen::Index j = 0; j < res.size(); ++j) norm += std::norm(res(j));

		return std::sqrt(norm);
	}
}  // namespace GPU::internal