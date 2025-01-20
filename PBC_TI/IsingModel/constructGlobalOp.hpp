#pragma once

#include <HilbertSpace/ManyBodyHilbertSpace/TransSector.hpp>
#include <Eigen/Dense>

template<typename Scalar, class TotalSpace_>
Eigen::MatrixX<Scalar> construct_globalOp(Eigen::MatrixX<Scalar> const&           locOp,
                                          TransSector<TotalSpace_, Scalar> const& subSpace) {
	auto const& basis    = subSpace.basis();
	auto const  dimLocOp = locOp.rows();

	Eigen::MatrixX<Scalar> res = Eigen::MatrixX<Scalar>::Zero(subSpace.dim(), subSpace.dim());
#pragma omp parallel for schedule(dynamic, 10)
	for(Index idx1 = 0; idx1 != subSpace.dim(); ++idx1)
		for(Index idx2 = idx1; idx2 != subSpace.dim(); ++idx2) {
			Scalar coeff = 0.0;
			for(Index pos2 = basis.outerIndexPtr()[idx2]; pos2 < basis.outerIndexPtr()[idx2 + 1];
			    ++pos2) {
				Index const  inBasis    = basis.innerIndexPtr()[pos2];
				Scalar const basisElem2 = basis.valuePtr()[pos2];
				Index const  locIdx2    = inBasis % dimLocOp;
				Index const  residual2  = inBasis / dimLocOp;

				for(Index pos1 = basis.outerIndexPtr()[idx1];
				    pos1 < basis.outerIndexPtr()[idx1 + 1]; ++pos1) {
					Index const  outBasis   = basis.innerIndexPtr()[pos1];
					Scalar const basisElem1 = basis.valuePtr()[pos1];
					Index const  locIdx1    = outBasis % dimLocOp;
					Index const  residual1  = outBasis / dimLocOp;
					if(residual1 != residual2) continue;

					coeff += conj(basisElem1) * locOp(locIdx1, locIdx2) * basisElem2;
				}
			}
			res(idx1, idx2) = coeff;
			if(idx1 == idx2)
				res(idx1, idx2) = real(res(idx1, idx2));
			else
				res(idx2, idx1) = conj(res(idx1, idx2));
		}
	return res;
}

#ifdef __NVCC__
	#include "constructGlobalOp.cuh"
#endif