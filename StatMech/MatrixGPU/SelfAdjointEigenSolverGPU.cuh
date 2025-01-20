
#pragma once

#include "macros.hpp"
#include "MatrixGPU.cuh"
#include "magma_overloads.cuh"
#include <vector>

namespace GPU {
	template<class Matrix_>
	class SelfAdjointEigenSolver;

	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	class SelfAdjointEigenSolver<
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
			SelfAdjointEigenSolver() = default;
			SelfAdjointEigenSolver(MatrixCPU const& hmat, Eigen::DecompositionOptions option
			                                              = Eigen::ComputeEigenvectors) {
				std::cout << "# GPU::SelfAdjointEigenSolver from CPU: Copy constructor."
				          << std::endl;
				this->compute(hmat, option);
			}
			SelfAdjointEigenSolver(MatrixCPU&&                 hmat,
			                       Eigen::DecompositionOptions option = Eigen::ComputeEigenvectors)
			    : m_eigvecs(std::move(hmat)) {
				std::cout << "# GPU::SelfAdjointEigenSolver from CPU: Move constructor."
				          << std::endl;
				this->compute(m_eigvecs, option);
			}

			MatrixCPU const&        eigenvectors() const { return m_eigvecs; }
			VectorCPU const&        eigenvalues() const { return m_eigvals; }
			SelfAdjointEigenSolver& compute(MatrixCPU const&            hmat,
			                                Eigen::DecompositionOptions option
			                                = Eigen::ComputeEigenvectors) {
				DEBUG(std::cerr << "# " << __func__ << std::endl);
				magma_vec_t  jobz = (option == Eigen::ComputeEigenvectors ? MagmaVec : MagmaNoVec);
				magma_uplo_t uplo = MagmaLower;
				std::vector<ScalarCPU>   work(1);
				std::vector<RealScalar>  rwork(1);
				std::vector<magma_int_t> iwork(1);

				if(&m_eigvecs != &hmat) m_eigvecs = hmat;
				m_eigvals.resize(hmat.rows());
				magma_heevd(jobz, uplo, m_eigvecs.rows(), m_eigvecs.data(), m_eigvecs.rows(),
				            m_eigvals.data(), work.data(), -1, rwork.data(), -1, iwork.data(), -1,
				            &m_info);
				DEBUG(std::cerr << "#           m_eigvecs.rows() = " << m_eigvecs.rows()
				                << std::endl);
				DEBUG(std::cerr << "#           m_eigvecs.cols() = " << m_eigvecs.cols()
				                << std::endl);
				DEBUG(std::cerr << "# magma_int_t(real(work[0])) = " << magma_int_t(real(work[0]))
				                << std::endl);
				DEBUG(std::cerr << "#      magma_int_t(rwork[0]) = " << magma_int_t(rwork[0])
				                << std::endl);
				DEBUG(std::cerr << "#                   iwork[0] = " << iwork[0] << std::endl);
				work.resize(magma_int_t(real(work[0])));
				rwork.resize(magma_int_t(rwork[0]));
				iwork.resize(iwork[0]);
				DEBUG(std::cerr << "#        work.size() = " << work.size() << std::endl);
				DEBUG(std::cerr << "#       rwork.size() = " << rwork.size() << std::endl);
				DEBUG(std::cerr << "#       iwork.size() = " << iwork.size() << std::endl);
				magma_heevd(jobz, uplo, m_eigvecs.rows(), m_eigvecs.data(), m_eigvecs.rows(),
				            m_eigvals.data(), work.data(), magma_int_t(work.size()), rwork.data(),
				            magma_int_t(rwork.size()), iwork.data(), magma_int_t(iwork.size()),
				            &m_info);
				DEBUG(std::cerr << "# info = " << m_info << "\n" << std::endl);
				if(m_info != MAGMA_SUCCESS) {
					std::cerr << "# Error: " << __FILE__ << ":" << __LINE__ << "\t"
					          << magma_strerror(m_info) << std::endl;
					std::exit(EXIT_FAILURE);
				}
				return *this;
			}
	};

	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	class SelfAdjointEigenSolver<
	    MatrixGPU<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> > {
		public:
			using MatrixCPU  = Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
			using Index      = typename MatrixCPU::Index;
			using ScalarCPU  = typename MatrixCPU::Scalar;
			using RealScalar = typename MatrixCPU::RealScalar;
			using VectorCPU  = Eigen::VectorX<RealScalar>;

			using MatrixGPU = MatrixGPU<MatrixCPU>;
			using ScalarGPU = typename MatrixGPU::Scalar;

		private:
			MatrixGPU   m_eigvecs;
			VectorCPU   m_eigvals;
			magma_int_t m_info;

		public:
			SelfAdjointEigenSolver() = default;
			SelfAdjointEigenSolver(MatrixGPU const& dmat, Eigen::DecompositionOptions option
			                                              = Eigen::ComputeEigenvectors) {
				this->compute(dmat, option);
			}
			MatrixGPU const& eigenvectorsGPU() const { return m_eigvecs; }
			MatrixCPU        eigenvectors() const {
                MatrixCPU res(m_eigvecs.rows(), m_eigvecs.cols());
                m_eigvecs.copyTo(res);
                return res;
			}
			VectorCPU const&        eigenvalues() const { return m_eigvals; }
			SelfAdjointEigenSolver& compute(MatrixGPU const&            dmat,
			                                Eigen::DecompositionOptions option
			                                = Eigen::ComputeEigenvectors) {
				DEBUG(std::cerr << "# " << __func__ << std::endl);
				magma_vec_t  jobz = (option == Eigen::ComputeEigenvectors ? MagmaVec : MagmaNoVec);
				magma_uplo_t uplo = MagmaLower;
				magma_int_t const        ldwa = dmat.rows();
				std::vector<ScalarGPU>   wA(ldwa * dmat.cols());
				std::vector<ScalarGPU>   work(1);
				std::vector<RealScalar>  rwork(1);
				std::vector<magma_int_t> iwork(1);

				m_eigvecs = dmat;
				m_eigvals.resize(dmat.rows());
				magma_heevd_gpu(jobz, uplo, m_eigvecs.rows(), m_eigvecs.data(), m_eigvecs.LD(),
				                m_eigvals.data(), wA.data(), ldwa, work.data(), -1, rwork.data(),
				                -1, iwork.data(), -1, &m_info);
				DEBUG(std::cerr << "#           m_eigvecs.rows() = " << m_eigvecs.rows()
				                << std::endl);
				DEBUG(std::cerr << "#           m_eigvecs.cols() = " << m_eigvecs.cols()
				                << std::endl);
				DEBUG(std::cerr << "# magma_int_t(real(work[0])) = " << magma_int_t(real(work[0]))
				                << std::endl);
				DEBUG(std::cerr << "#      magma_int_t(rwork[0]) = " << magma_int_t(rwork[0])
				                << std::endl);
				DEBUG(std::cerr << "#                   iwork[0] = " << iwork[0] << std::endl);
				work.resize(magma_int_t(real(work[0])));
				rwork.resize(magma_int_t(rwork[0]));
				iwork.resize(iwork[0]);
				DEBUG(std::cerr << "#        work.size() = " << work.size() << std::endl);
				DEBUG(std::cerr << "#       rwork.size() = " << rwork.size() << std::endl);
				DEBUG(std::cerr << "#       iwork.size() = " << iwork.size() << std::endl);
				magma_heevd_gpu(jobz, uplo, m_eigvecs.rows(), m_eigvecs.data(), m_eigvecs.LD(),
				                m_eigvals.data(), wA.data(), ldwa, work.data(),
				                magma_int_t(work.size()), rwork.data(), magma_int_t(rwork.size()),
				                iwork.data(), magma_int_t(iwork.size()), &m_info);
				DEBUG(std::cerr << "# info = " << m_info << "\n" << std::endl);
				if(m_info != MAGMA_SUCCESS) {
					std::cerr << "# Error: " << __FILE__ << ":" << __LINE__ << "\t"
					          << magma_strerror(m_info) << std::endl;
					std::exit(EXIT_FAILURE);
				}
				return *this;
			}
	};

}  // namespace GPU