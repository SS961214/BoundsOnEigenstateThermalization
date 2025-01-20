#pragma once

#include <HilbertSpace/ManyBodyHilbertSpace/ManyBodyBosonSpace.hpp>
#include <HilbertSpace/OpSpace/mBodyOpSpace_Boson.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

enum BoundaryCondition { OBC, PBC };

Eigen::SparseMatrix<double> BoseHubbard(ManyBodyBosonSpace const& mbSpace, double J1, double U1,
                                        double J2 = 0, double U2 = 0, BoundaryCondition bc = PBC) {
	if(bc != PBC) {
		std::cerr << "Open boundary condition is not implemented." << std::endl;
		std::exit(EXIT_FAILURE);
	}
	std::cout << "# Hamiltonian: " << __FILE__ << ":\tL=" << mbSpace.sysSize()
	          << ",\tN=" << mbSpace.N() << ",\tdim = " << mbSpace.dim() << std::endl;
	Eigen::SparseMatrix<double> res(mbSpace.dim(), mbSpace.dim());
	res.reserve(Eigen::ArrayXi::Constant(mbSpace.dim(), 4 * mbSpace.sysSize()));
	res.setZero();
	if(mbSpace.dim() <= 1) return res;

	int const sysSize = mbSpace.sysSize();
	using Scalar      = double;

	double const startT = omp_get_wtime();
	// Kinetic term
	std::cout << "##\t " << __FILE__ << ": Calculating Kinetic terms..." << std::endl;
	{
		Eigen::SparseMatrix<double> adj;
		adj.reserve(Eigen::ArrayXi::Constant(mbSpace.dim(), 4 * mbSpace.sysSize()));
		mBodyOpSpace<ManyBodyBosonSpace, Scalar> const opSpace(1, mbSpace);
		Eigen::VectorXi                                config = Eigen::VectorXi::Zero(sysSize);
		for(auto pos = 0; pos < sysSize; ++pos) {
			// Nearest neighbor hopping
			config(pos)                 = 1;
			config((pos + 1) % sysSize) = 1 * (opSpace.maxOnSite() + 1);
			auto opNum                  = opSpace.config_to_ordinal(config);
			// std::cout << "# opNum = " << opNum << ": " << opSpace.ordinal_to_config(opNum)
			//           << std::endl;
			res += -J1 * opSpace.basisOp(opNum);
			config(pos)                 = 0;
			config((pos + 1) % sysSize) = 0;

			// Next-nearest neighbor hopping
			config(pos)                 = 1;
			config((pos + 2) % sysSize) = 1 * (opSpace.maxOnSite() + 1);
			opNum                       = opSpace.config_to_ordinal(config);
			// std::cout << "# opNum = " << opNum << ": " << opSpace.ordinal_to_config(opNum)
			//           << std::endl;
			res += -J2 * opSpace.basisOp(opNum);
			config(pos)                 = 0;
			config((pos + 2) % sysSize) = 0;
		}
		adj = res.adjoint();
		res += adj;
	}

	std::cout << "##\t " << __FILE__ << ": Calculating interaction terms..." << std::endl;
	Eigen::VectorXd diag = Eigen::VectorXd::Zero(mbSpace.dim());
#pragma omp parallel
	{
		Eigen::RowVectorXi config(mbSpace.sysSize());
// Interaction and Potential terms
#pragma omp for
		for(auto j = 0; j < mbSpace.dim(); ++j) {
			mbSpace.ordinal_to_config(config, j);
			// #pragma omp critical
			// 			std::cout << config << std::endl;
			double term = 0.0;
			for(auto l = 0; l < config.size(); ++l) {
				term += U1 * config(l) * (config(l) - 1) / 2.0
				        + U2 * config(l) * config((l + 1) % sysSize);
			}
			diag(j) = term;
		}
	}
	// Eigen::SparseMatrix<double> spMat = res.sparseView();
	res += diag.asDiagonal();
	res.makeCompressed();
	std::cout << "#(END) Hamiltonian: " << __FILE__ << "\tdim = " << mbSpace.dim()
	          << "\tNonZeros=" << res.nonZeros() << ",\t elapsed = " << omp_get_wtime() - startT
	          << " (sec)" << std::endl;
	// spMat.makeCompressed();
	// return spMat;
	return res;
}
