#pragma once

#include <HilbertSpace>
#include <Eigen/Dense>

template<class Array_, typename Scalar_, class Sector_>
Eigen::MatrixX<Scalar_> construct_globalOp(
    Array_ const& coeffs, mBodyOpSpace<ManyBodyBosonSpace, Scalar_> const& locOpSpace,
    mBodyOpSpace<ManyBodyBosonSpace, Scalar_> const& globOpSpace, Sector_ const& subSpace) {
	static_assert(std::is_same_v<Sector_, TransSector<ManyBodyBosonSpace, Scalar_>>
	              || std::is_same_v<Sector_, TransParitySector<ManyBodyBosonSpace, Scalar_>>);
	if(coeffs.size() < locOpSpace.dim()) {
		std::cerr << "Error: construct_globalOp: coeffs.size() < locOpSpace.dim()" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	if(locOpSpace.m() != globOpSpace.m()) {
		std::cerr << "Error: construct_globalOp: locOpSpace.m() != globOpSpace.m(): "
		          << locOpSpace.m() << " != " << globOpSpace.m() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	Index const                  dim = globOpSpace.baseSpace().dim();
	Eigen::SparseMatrix<Scalar_> Htot(dim, dim);
	Htot.setZero();
	Eigen::ArrayXi config = Eigen::ArrayXi::Zero(globOpSpace.sysSize());
	for(auto i = 0; i < locOpSpace.dim(); ++i) {
		locOpSpace.ordinal_to_config(config.head(locOpSpace.sysSize()), i);
		Index const globOpNum = globOpSpace.config_to_ordinal(config);
		Htot += coeffs[i] * globOpSpace.basisOp(globOpNum);
	}
	{
		Eigen::SparseMatrix<Scalar_> adj = Htot.adjoint();
		Htot += adj;
	}
	Htot *= 0.5;
	Htot.makeCompressed();

	Eigen::MatrixX<Scalar_> res = subSpace.basis().adjoint() * Htot * subSpace.basis();
	return res;
}