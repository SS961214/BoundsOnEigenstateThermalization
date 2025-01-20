#pragma once

#include <HilbertSpace/ManyBodyHilbertSpace/ManyBodyFermionSpace.hpp>
#include <HilbertSpace/OpSpace/mBodyOpSpace_Fermion.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <omp.h>

enum BoundaryCondition { OBC, PBC };

Eigen::SparseMatrix<double> FermiHubbard(ManyBodyFermionSpace const& mbSpace, double t1, double V1,
                                         double t2, double V2, BoundaryCondition bc = PBC) {
	if(bc != PBC) {
		std::cerr << "Open boundary condition is not implemented." << std::endl;
		std::exit(EXIT_FAILURE);
	}
	std::cout << "# Hamiltonian: " << __FILE__ << ":\tL=" << mbSpace.sysSize()
	          << ",\tN=" << mbSpace.N() << ",\tdim = " << mbSpace.dim() << std::endl;
	Eigen::SparseMatrix<double> res(mbSpace.dim(), mbSpace.dim());
	res.reserve(Eigen::ArrayXi::Constant(mbSpace.dim(), 4 * mbSpace.sysSize()));
	if(mbSpace.dim() <= 1) return res;
	// Eigen::MatrixXd res = Eigen::MatrixXd::Zero(mbSpace.dim(), mbSpace.dim());
	// if(mbSpace.dim() <= 1) return res.sparseView();

	int const sysSize = mbSpace.sysSize();
	using Scalar      = double;

	double const startT = omp_get_wtime();
	// Kinetic term
	std::cout << "##\t " << __FILE__ << ": Calculating Kinetic terms..." << std::endl;
	{
		Eigen::SparseMatrix<double> adj;
		adj.reserve(Eigen::ArrayXi::Constant(mbSpace.dim(), 2 * mbSpace.sysSize()));
		mBodyOpSpace<ManyBodyFermionSpace, Scalar> const opSpace(1, mbSpace);
		Eigen::VectorXi                                  config = Eigen::VectorXi::Zero(sysSize);
		for(auto pos = 0; pos < sysSize; ++pos) {
			config(pos)                 = 1;
			config((pos + 1) % sysSize) = 2;
			auto opNum                  = opSpace.config_to_ordinal(config);
			// std::cout << "# opNum = " << opNum << ": " << opSpace.ordinal_to_config(opNum)
			//           << std::endl;
			res += -t1 * opSpace.basisOp(opNum);
			config(pos)                 = 0;
			config((pos + 1) % sysSize) = 0;

			config(pos)                 = 1;
			config((pos + 2) % sysSize) = 2;
			opNum                       = opSpace.config_to_ordinal(config);
			// std::cout << "# opNum = " << opNum << ": " << opSpace.ordinal_to_config(opNum)
			//           << std::endl;
			res += -t2 * opSpace.basisOp(opNum);
			config(pos)                 = 0;
			config((pos + 2) % sysSize) = 0;
		}
		adj = res.adjoint();
		res += adj;
	}

	std::cout << "##\t " << __FILE__ << ": Calculating interaction terms..." << std::endl;
#pragma omp parallel
	{
		Eigen::RowVectorXi config(mbSpace.sysSize());
// Interaction terms
#pragma omp for
		for(auto j = 0; j < mbSpace.dim(); ++j) {
			mbSpace.ordinal_to_config(config, j);
			// #pragma omp critical
			// 			std::cout << config << std::endl;
			double term = 0.0;
			for(auto l = 0; l < config.size(); ++l) {
				term += V1 * config(l) * config((l + 1) % sysSize)
				        + V2 * config(l) * config((l + 2) % sysSize);
			}
			res.coeffRef(j, j) += term;
		}
	}
	// Eigen::SparseMatrix<double> spMat = res.sparseView();
	res.makeCompressed();
	std::cout << "#(END) Hamiltonian: " << __FILE__ << "\tdim = " << mbSpace.dim()
	          << "\tNonZeros=" << res.nonZeros() << ",\t elapsed = " << omp_get_wtime() - startT
	          << " (sec)" << std::endl;
	// spMat.makeCompressed();

	// std::exit(EXIT_SUCCESS);
	// return spMat;
	return res;
}

//
// WITHOUT PRERESERVATION OF MEMORIES
//
// root@992cb4ef060d # for N in $(seq 9 18);do L=$((2*N)); ./PBC_TI/FermiHubbard/quasiETHmeasure_mBody ${L} ${L} ${N} ${N} 1 1 1 1 1 1 0.2 $HOME/temp; done                                                                  (work)-[GPU]
// EIGEN_USE_MKL_ALL is set
// rootDir = "/root/temp"
// # Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp:  L=18,   N=9,    dim = 48620
// #(END) Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp      dim = 48620     NonZeros=473330,         elapsed = 0.093039 (sec)
// EIGEN_USE_MKL_ALL is set
// rootDir = "/root/temp"
// # Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp:  L=20,   N=10,   dim = 184756
// #(END) Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp      dim = 184756    NonZeros=1983696,        elapsed = 0.547483 (sec)
// EIGEN_USE_MKL_ALL is set
// rootDir = "/root/temp"
// # Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp:  L=22,   N=11,   dim = 705432
// #(END) Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp      dim = 705432    NonZeros=8280428,        elapsed = 1.50401 (sec)
// EIGEN_USE_MKL_ALL is set
// rootDir = "/root/temp"
// # Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp:  L=24,   N=12,   dim = 2704156
// #(END) Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp      dim = 2704156   NonZeros=34448596,       elapsed = 6.43522 (sec)
// EIGEN_USE_MKL_ALL is set
// rootDir = "/root/temp"
// # Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp:  L=26,   N=13,   dim = 10400600
// #(END) Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp      dim = 10400600  NonZeros=142904244,      elapsed = 28.0819 (sec)
// EIGEN_USE_MKL_ALL is set
// rootDir = "/root/temp"
// # Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp:  L=28,   N=14,   dim = 40116600
// #(END) Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp      dim = 40116600  NonZeros=591348400,      elapsed = 141.405 (sec)
// EIGEN_USE_MKL_ALL is set
// rootDir = "/root/temp"
// # Hamiltonian: /root/work/PBC_TI/FermiHubbard/Hamiltonian.hpp:  L=30,   N=15,   dim = 155117520
// terminate called after throwing an instance of 'std::bad_alloc'
//   what():  std::bad_alloc