
#pragma once

#include "macros.hpp"
#include "MatrixGPU.cuh"
#include "magma_overloads.cuh"
#include <vector>

namespace GPU {
	template<class Matrix_>
	class SelfAdjointEigenSolver_mgpu;

	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	class SelfAdjointEigenSolver_mgpu<
	    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> > {
		public:
			using MatrixCPU  = Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
			using Index      = typename MatrixCPU::Index;
			using ScalarCPU  = typename MatrixCPU::Scalar;
			using RealScalar = typename MatrixCPU::RealScalar;
			using VectorCPU  = Eigen::VectorX<RealScalar>;

		private:
			MatrixCPU   m_eigvecs;
			VectorCPU   m_eigvals;
			magma_int_t m_info;

		public:
			SelfAdjointEigenSolver_mgpu() = default;
			SelfAdjointEigenSolver_mgpu(int const ngpus, MatrixCPU const& hmat,
			                            Eigen::DecompositionOptions option
			                            = Eigen::ComputeEigenvectors) {
				DEBUG(std::cout
				      << "# GPU::SelfAdjointEigenSolver_mgpu from CPU: Copy constructor. (ngpus = "
				      << ngpus << ")" << std::endl);
				this->compute(ngpus, hmat, option);
			}
			SelfAdjointEigenSolver_mgpu(int const ngpus, MatrixCPU&& hmat,
			                            Eigen::DecompositionOptions option
			                            = Eigen::ComputeEigenvectors)
			    : m_eigvecs(std::move(hmat)) {
				DEBUG(std::cout
				      << "# GPU::SelfAdjointEigenSolver_mgpu from CPU: Move constructor. (ngpus = "
				      << ngpus << ")" << std::endl);
				this->compute(ngpus, m_eigvecs, option);
			}

			MatrixCPU const&             eigenvectors() const { return m_eigvecs; }
			VectorCPU const&             eigenvalues() const { return m_eigvals; }
			SelfAdjointEigenSolver_mgpu& compute(int const ngpus, MatrixCPU const& hmat,
			                                     Eigen::DecompositionOptions option
			                                     = Eigen::ComputeEigenvectors) {
				DEBUG(std::cerr << "# " << __func__ << std::endl);

				magma_vec_t  jobz = (option == Eigen::ComputeEigenvectors ? MagmaVec : MagmaNoVec);
				magma_uplo_t uplo = MagmaLower;

				if(&m_eigvecs != &hmat) {
					m_eigvecs.resize(hmat.rows(), hmat.cols());
#pragma omp parallel for
					for(Eigen::Index j = 0; j < hmat.size(); ++j) m_eigvecs(j) = hmat(j);
				}
				m_eigvals.resize(hmat.rows());
				magma_int_t const dim = hmat.rows();

				DEBUG(std::cout << "## Diagonalizing a matrix on n = " << ngpus << " GPUs."
				                << std::endl);
				magma_int_t              info = 0;
				std::vector<ScalarCPU>   work(1);
				std::vector<RealScalar>  rwork(1);
				std::vector<magma_int_t> iwork(1);
				magma_heevd_m(ngpus, jobz, uplo, dim, m_eigvecs.data(), dim, m_eigvals.data(),
				              work.data(), -1, rwork.data(), -1, iwork.data(), -1, &info);
				magma_int_t const lwork  = magma_int_t(real(work[0]));
				magma_int_t const lrwork = magma_int_t(rwork[0]);
				magma_int_t const liwork = iwork[0];
				DEBUG(std::cout << "#\t  lwork = " << lwork << "\n"
				                << "#\t lrwork = " << lrwork << "\n"
				                << "#\t liwork = " << liwork << std::endl);
				work.resize(lwork);
				rwork.resize(lrwork);
				iwork.resize(liwork);
				magma_heevd_m(ngpus, jobz, uplo, dim, m_eigvecs.data(), dim, m_eigvals.data(),
				              work.data(), lwork, rwork.data(), lrwork, iwork.data(), liwork,
				              &info);
				DEBUG(std::cout << "#\t  info = " << info << std::endl);

				if(info != 0) {
					std::cerr << "# Error: Diagonalization failed with info = " << info
					          << std::endl;
					std::exit(EXIT_FAILURE);
				}

				return *this;
			}
	};
}  // namespace GPU