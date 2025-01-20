#pragma once

#include <HilbertSpace/ManyBodyHilbertSpace/ManyBodyBosonSpace.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// enum BoundaryCondition { OBC, PBC };

Eigen::SparseMatrix<double> BoseHubbard_ref(ManyBodyBosonSpace const& mbSpace, double t1, double U1, double t2 = 0, double U2 = 0, BoundaryCondition bc = PBC) {
	std::cout << "# Hamiltonian: " << __FILE__ << "\tdim = " << mbSpace.dim() << std::endl;
	Eigen::MatrixXd res = Eigen::MatrixXd::Zero(mbSpace.dim(), mbSpace.dim());
	int const sysSize = mbSpace.sysSize();

	auto const kineticTerm = [&](int site1, int site2, Eigen::RowVectorXi& config) {
		double    res = 0;
		int const n   = config(site1);
		if(n > 0) {
			config(site1) -= 1;
			config(site2) += 1;
			res = -t1 * std::sqrt(double(config(site2) * n));
		}
		return res;
	};

	// std::cout << "# Hamiltonian: " << __FILE__ << "\tdim = " << mbSpace.dim() << std::endl;
#pragma omp parallel
	{
		double             coeff = 0;
		Eigen::RowVectorXi config(mbSpace.sysSize());
// Kinetic term
#pragma omp for
		for(auto j = 0; j < mbSpace.dim(); ++j) {
			mbSpace.ordinal_to_config(config, j);
			for(auto l = 1; l < sysSize; ++l) {
				coeff = kineticTerm(l - 1, l, config);
				if(coeff != 0) {
					auto const outIdx = mbSpace.config_to_ordinal(config);
					res(outIdx, j) += coeff;
					config(l - 1) += 1;
					config(l) -= 1;
				}
				coeff = kineticTerm(l, l - 1, config);
				if(coeff != 0) {
					auto const outIdx = mbSpace.config_to_ordinal(config);
					res(outIdx, j) += coeff;
					config(l) += 1;
					config(l - 1) -= 1;
				}
			}
			if(bc == PBC) {
				coeff = kineticTerm(sysSize - 1, 0, config);
				if(coeff != 0) {
					auto const outIdx = mbSpace.config_to_ordinal(config);
					res(outIdx, j) += coeff;
					config(sysSize - 1) += 1;
					config(0) -= 1;
				}
				coeff = kineticTerm(0, sysSize - 1, config);
				if(coeff != 0) {
					auto const outIdx = mbSpace.config_to_ordinal(config);
					res(outIdx, j) += coeff;
					config(0) += 1;
					config(sysSize - 1) -= 1;
				}
			}
		}
	}

	// std::cout << "# Hamiltonian: " << __FILE__ << "\tdim = " << mbSpace.dim() << std::endl;
#pragma omp parallel
	{
		Eigen::RowVectorXi config(mbSpace.sysSize());
// Interaction and Potential terms
#pragma omp for
		for(auto j = 0; j < mbSpace.dim(); ++j) {
			mbSpace.ordinal_to_config(config, j);
			// #pragma omp critical
			// 			std::cout << config << std::endl;
			for(auto l = 0; l < config.size(); ++l) {
				res(j, j) += U1 * config(l) * (config(l) - 1) / 2.0;
			}
		}
	}
	Eigen::SparseMatrix<double> spMat = res.sparseView();
	std::cout << "#(END) Hamiltonian: " << __FILE__ << "\tdim = " << mbSpace.dim() << "\tNonZeros=" << spMat.nonZeros() << std::endl;
	spMat.makeCompressed();
	return spMat;
}
