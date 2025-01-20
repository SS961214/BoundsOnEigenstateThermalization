#pragma once

#include <Eigen/Dense>

template<class Vector_>
double LevelSpacingRatio(Vector_ const& eigVals) {
	Index const dim = eigVals.size();
	double       res = 0;
	assert(dim >= 3);
#pragma omp parallel for reduction(+: res)
	for(Index j = 0; j < dim - 2; ++j) {
		double ratio = (eigVals[j + 1] - eigVals[j]) / (eigVals[j + 2] - eigVals[j + 1]);
		res += (ratio < 1 ? ratio : 1 / ratio);
	}
	return res / double(dim - 1);
}