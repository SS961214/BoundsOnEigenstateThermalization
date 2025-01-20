#pragma once

#include <HilbertSpace>
#include <Eigen/Dense>

template<class Array_, typename Scalar_, class Sector_>
Eigen::MatrixX<Scalar_> construct_globalOp(
    Array_ const& coeffs, Combination const& locOpConfig,
    mBodyOpSpace<ManyBodyFermionSpace, Scalar_> const& globOpSpace, Sector_ const& subSpace) {
	static_assert(std::is_same_v<Sector_, TransSector<ManyBodyFermionSpace, Scalar_>>
	              || std::is_same_v<Sector_, TransParitySector<ManyBodyFermionSpace, Scalar_>>);
	if(coeffs.size() < locOpConfig.dim()) {
		std::cerr << "Error: construct_globalOp: coeffs.size() < locOpConfig.dim()" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	if(locOpConfig.N() != globOpSpace.m()) {
		std::cerr << "Error: construct_globalOp: locOpConfig.N() != globOpSpace.m(): "
		          << locOpConfig.N() << " != " << globOpSpace.m() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	Index const                  dim = globOpSpace.baseSpace().dim();
	Eigen::SparseMatrix<Scalar_> Htot(dim, dim);
	Htot.setZero();
	for(auto i = 0; i < locOpConfig.dim() * locOpConfig.dim(); ++i) {
		auto const  cConf     = locOpConfig.ordinal_to_config(i / locOpConfig.dim());
		auto const  aConf     = locOpConfig.ordinal_to_config(i % locOpConfig.dim());
		auto const  config    = (cConf << 32) | aConf;
		Index const globOpNum = globOpSpace.config_to_ordinal(config);
		Htot += coeffs[i] * globOpSpace.basisOp(globOpNum);
		// std::cout << "i = " << i << ", config = " << globOpSpace.ordinal_to_config(globOpNum) << std::endl;
		// << ", "
		//           << Eigen::RowVectorXi::NullaryExpr(
		//                  64, [&](int pos) { return ((config >> pos) & 1); })
		//           << ", cConf = " << cConf << ", aConf = " << aConf << ", globOpNum = " << globOpNum
		//           << std::endl;
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