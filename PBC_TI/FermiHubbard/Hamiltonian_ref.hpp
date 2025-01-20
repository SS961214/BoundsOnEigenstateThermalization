#pragma once

#include <HilbertSpace/ManyBodyHilbertSpace/ManyBodyFermionSpace.hpp>
#include <HilbertSpace/OpSpace/mBodyOpSpace_Fermion.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <omp.h>

// enum BoundaryCondition { OBC, PBC };

Eigen::SparseMatrix<double> FermiHubbard_ref(ManyBodyFermionSpace const& mbSpace, double t1, double V1,
                                         double t2, double V2, BoundaryCondition bc = PBC) {
	assert(bc == PBC && "Open boundary condition is not implemented.");
	std::cout << "# Hamiltonian: " << __FILE__ << ":\tL=" << mbSpace.sysSize()
	          << ",\tN=" << mbSpace.N() << ",\tdim = " << mbSpace.dim() << std::endl;
	Eigen::SparseMatrix<double> res(mbSpace.dim(), mbSpace.dim());
	if(mbSpace.dim() <= 1) return res;
	std::cout << "##\t Reserving memories for non-zero elements..." << std::endl;
	res.reserve(Eigen::ArrayXi::Constant(mbSpace.dim(), 4 * mbSpace.sysSize()));
	std::cout << "##\t Reserved memories for non-zero elements." << std::endl;

	int const sysSize = mbSpace.sysSize();
	using Scalar      = double;

	double const startT = omp_get_wtime();
	// 	// Kinetic term
	mBodyOpSpace<ManyBodyFermionSpace, Scalar> opSpace(2, mbSpace);
#pragma omp parallel
	{
		Eigen::VectorXi opConf = Eigen::VectorXi::Zero(sysSize);
		Eigen::ArrayXi  work(opSpace.actionWorkSize());
		Index           k, opNum;
		double          coeff;
#pragma omp for
		for(auto j = 0; j < mbSpace.dim(); ++j) {
			for(auto pos = 0; pos < sysSize; ++pos) {
				// Nearest neighbor hopping
				{
					opConf(pos)                 = 1;
					opConf((pos + 1) % sysSize) = 2;
					opNum                       = opSpace.config_to_ordinal(opConf);
					opSpace.action(k, coeff, opNum, j, work);
					assert(0 <= k && k < mbSpace.dim());
					res.coeffRef(k, j) += -t1 * coeff;

					opConf(pos)                 = 2;
					opConf((pos + 1) % sysSize) = 1;
					opNum                       = opSpace.config_to_ordinal(opConf);
					opSpace.action(k, coeff, opNum, j, work);
					assert(0 <= k && k < mbSpace.dim());
					res.coeffRef(k, j) += -t1 * coeff;

					opConf(pos)                 = 0;
					opConf((pos + 1) % sysSize) = 0;
				}

				// Next-nearest neighbor hopping
				{
					opConf(pos)                 = 1;
					opConf((pos + 2) % sysSize) = 2;
					opNum                       = opSpace.config_to_ordinal(opConf);
					opSpace.action(k, coeff, opNum, j, work);
					assert(0 <= k && k < mbSpace.dim());
					res.coeffRef(k, j) += -t2 * coeff;

					opConf(pos)                 = 2;
					opConf((pos + 2) % sysSize) = 1;
					opNum                       = opSpace.config_to_ordinal(opConf);
					opSpace.action(k, coeff, opNum, j, work);
					assert(0 <= k && k < mbSpace.dim());
					res.coeffRef(k, j) += -t2 * coeff;

					opConf(pos)                 = 0;
					opConf((pos + 2) % sysSize) = 0;
				}
			}
		}
		// for(auto pos = 0; pos < sysSize; ++pos) {
		// 	config(pos)                 = 1;
		// 	config((pos + 1) % sysSize) = 2;
		// 	auto opNum                  = opSpace.config_to_ordinal(config);
		// 	res += -t1 * opSpace.basisOp(opNum);
		// 	config(pos)                 = 0;
		// 	config((pos + 1) % sysSize) = 0;

		// 	config(pos)                 = 1;
		// 	config((pos + 2) % sysSize) = 2;
		// 	opNum                       = opSpace.config_to_ordinal(config);
		// 	res += -t2 * opSpace.basisOp(opNum);
		// 	config(pos)                 = 0;
		// 	config((pos + 2) % sysSize) = 0;
		// }
		// Eigen::SparseMatrix<double> adj = res.adjoint();
		// res += adj;
	}

// 	// std::cout << "# Hamiltonian: " << __FILE__ << "\tdim = " << mbSpace.dim() << std::endl;
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
			for(auto l = 0; l < sysSize; ++l) {
				term += V1 * config(l) * config((l + 1) % sysSize)
				        + V2 * config(l) * config((l + 2) % sysSize);
			}
			res.coeffRef(j, j) += term;
		}
	}
	std::cout << "#(END) Hamiltonian: " << __FILE__ << "\tdim = " << mbSpace.dim()
	          << "\tNonZeros=" << res.nonZeros() << ",\t elapsed = " << omp_get_wtime() - startT
	          << " (sec)" << std::endl;
	res.makeCompressed();
	std::exit(EXIT_SUCCESS);
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