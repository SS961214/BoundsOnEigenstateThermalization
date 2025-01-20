#pragma once

#include <Eigen/Dense>

template<class Array_>
Eigen::ArrayXi MCEnergyShell(Array_ const& eigVals, int NDIV) {
	double const   range = eigVals(eigVals.size() - 1) - eigVals(0);
	double const   gE    = eigVals(0);
	Eigen::ArrayXi res(NDIV + 1);
	res(0) = 0;
	int id = 0;
	for(int j = 1; j <= NDIV; ++j) {
		double const div = j / double(NDIV);
		for(; id < eigVals.size() && (eigVals[id] - gE) / range < div; ++id) {};
		res(j) = id;
	}
	return res;
}

template<class Array1_, class Array2_>
Eigen::ArrayXi get_shellDims(Array1_ const& shCenter, double const shWidth,
                             Array2_ const& eigVals) {
	Eigen::ArrayXi res(shCenter.size());

#pragma omp parallel for
	for(auto j = 0; j < res.rows(); ++j) {
		int idMin, idMax;
		for(idMin = j; idMin >= 0; --idMin) {
			if(shCenter(j) - eigVals(idMin) > shWidth) break;
		}
		++idMin;
		for(idMax = j; idMax < eigVals.size(); ++idMax) {
			if(eigVals(idMax) - shCenter(j) > shWidth) break;
		}
		--idMax;

		res(j) = idMax - idMin + 1;
	}

	return res;
}

template<class Matrix_>
Eigen::MatrixX<typename Matrix_::Scalar> MCAverages(Eigen::VectorX<double> const& shCenter,
                                                    double const                  shWidth,
                                                    Eigen::VectorX<double> const& eigVals,
                                                    Matrix_ const&                vals) {
	assert(eigVals.size() == vals.rows());
	using Scalar_ = typename Matrix_::Scalar;
	Eigen::MatrixX<Scalar_> res(shCenter.size(), vals.cols());

#pragma omp parallel for
	for(auto j = 0; j < res.rows(); ++j) {
		int idMin, idMax;
		for(idMin = eigVals.size() - 1; idMin >= 0; --idMin) {
			if(shCenter(j) - eigVals(idMin) > shWidth) break;
		}
		++idMin;
		for(idMax = 0; idMax < eigVals.size(); ++idMax) {
			if(eigVals(idMax) - shCenter(j) > shWidth) break;
		}
		--idMax;

		res.row(j) = vals(Eigen::seq(idMin, idMax), Eigen::all).colwise().mean();
		// res.row(j) = Eigen::VectorXd::Constant(vals.cols(), idMax - idMin + 1);

		// std::cout << "#\t idMin = " << idMin << ", idMax = " << idMax << std::endl;
		// std::cout << res.row(j) << std::endl;
	}
	// std::cout << res << std::endl;

	return res;
}