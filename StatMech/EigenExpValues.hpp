#pragma once

#include <Eigen/Dense>

template<class Matrix1_, class Matrix2_>
Eigen::ArrayXd EigenExpValues(Matrix1_ const& obs, Matrix2_ const& eigVecs) {
	return Eigen::ArrayXd::NullaryExpr(eigVecs.cols(), [&](Index alpha) {
		return (eigVecs.col(alpha).adjoint() * obs * eigVecs.col(alpha)).real()(0);
	});
}

template<class Matrix1_, class Matrix2_>
Eigen::MatrixXcd EigenMatElems(Matrix1_ const& obs, Matrix2_ const& eigVecs) {
	Eigen::MatrixXcd res = eigVecs.adjoint() * obs * eigVecs;
	return (res + res.adjoint()) / 2.0;
}

#ifdef __NVCC__
	#include "EigenExpValues.cuh"
#endif