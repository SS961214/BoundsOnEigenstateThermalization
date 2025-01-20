#pragma once
#include "debug.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <random>
#include <sys/time.h>

namespace GPU::internal {
	static inline double getETtime() {
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return tv.tv_sec + (double)tv.tv_usec * 1e-6;
	}

	template<typename Scalar>
	void generateRandomMatrix(Eigen::MatrixX<Scalar>& mat, int dim) {
		assert(mat.rows() == dim);
		assert(mat.cols() == dim);
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

		DEBUG(std::cout << "## Preparing a matrix on CPU." << std::endl);
		double T_pre = getETtime();
#pragma omp parallel
		{
			std::random_device                   seed_gen;
			std::mt19937                         engine(seed_gen());
			std::normal_distribution<RealScalar> dist(0.0, 1.0);
#pragma omp for
			for(Eigen::Index j = 0; j < mat.size(); ++j) {
				mat(j) = Scalar(dist(engine), dist(engine));
			}
		}
		DEBUG(std::cout << "## Generated random numbers." << std::endl);
		double norm = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+ : norm)
		for(auto j = 0; j < mat.rows(); ++j) {
			mat(j, j) = 2 * std::real(mat(j, j));
			norm += std::norm(mat(j, j));
			for(auto k = 0; k < j; ++k) {
				mat(j, k) = mat(j, k) + std::conj(mat(k, j));
				mat(k, j) = std::conj(mat(j, k));
				norm += 2 * std::norm(mat(j, k));
			}
		}
		norm = std::sqrt(norm);
#pragma omp parallel for
		for(Eigen::Index j = 0; j < mat.size(); ++j) { mat(j) /= norm; }
		T_pre = getETtime() - T_pre;
		std::cout << "## Prepared a matrix on CPU.\t norm = " << std::scientific << norm << "\t T_pre = " << std::defaultfloat << T_pre
		          << " (sec)" << std::endl;
	}
}  // namespace GPU::internal