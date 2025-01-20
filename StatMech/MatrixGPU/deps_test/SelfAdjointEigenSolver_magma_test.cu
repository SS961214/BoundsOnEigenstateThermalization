#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <magma_v2.h>
#include <magma_operators.h>
#include <Eigen/Dense>
#include <random>
#include <complex>

#define MAGMA_CHECK(call)                                                  \
	{                                                                      \
		const magma_int_t error = call;                                    \
		if(error != MAGMA_SUCCESS) {                                       \
			printf("MAGMA_CHECK Error: %s:%d,  ", __FILE__, __LINE__);     \
			printf("code:%d, reason: %s\n", error, magma_strerror(error)); \
			assert(error == MAGMA_SUCCESS);                                \
		}                                                                  \
	};

using RealScalar = double;
using Scalar     = std::complex<RealScalar>;

TEST_CASE("SelfAdjointEigenSolver_eigen", "test") {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "# EIGEN_USE_MKL_ALL is set." << std::endl;
#else
	std::cout << "# EIGEN_USE_MKL_ALL is NOT set." << std::endl;
#endif
	Eigen::initParallel();
	constexpr double precision = 1.0e-12;
	constexpr int    dim       = 10000;
	constexpr int    LD        = (dim / 32 + (dim % 32 == 0 ? 0 : 1)) * 32;
	constexpr int    Nsample   = 10;
	// Total Test time (real) = 1304.81 sec/10 samples: with MKL, zheev, zhemm
	// Total Test time (real) = 1283.73 sec/10 samples: with MKL, zheev, zhemm, and OpenMP
	// with MKL, zheev, without zhemm
	static_assert(sizeof(Scalar) == sizeof(magmaDoubleComplex));

	std::random_device                   seed_gen;
	std::mt19937                         engine(seed_gen());
	std::normal_distribution<RealScalar> dist(0.0, 1.0);
	auto const                           RME = [&](int dim) {
        Eigen::MatrixX<Scalar> mat = Eigen::MatrixX<Scalar>::NullaryExpr(
            dim, dim, [&]() { return Scalar(dist(engine), dist(engine)); });
        mat = (mat + mat.adjoint()).eval();
        mat /= mat.norm();
        return mat;
	};
	// std::cout << "# Sample 1:\n" << RME(dim) << "\n\n" << "# Sample 2:\n" << RME(dim) << std::endl;

	magma_init();
	std::cout << "# Initialized MAGMA." << std::endl;
	magma_queue_t queue = NULL;
	magma_queue_create(0, &queue);

	Scalar *dMatData = nullptr, *hMatData = nullptr;
	MAGMA_CHECK(magma_malloc((void**)&dMatData, LD * dim * sizeof(Scalar)));
	MAGMA_CHECK(magma_malloc_pinned((void**)&hMatData, LD * dim * sizeof(Scalar)));
	Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::Stride<LD, 1>> hmat(hMatData, dim, dim);

	RealScalar* hSpecData = nullptr;
	MAGMA_CHECK(magma_malloc_pinned((void**)&hSpecData, dim * sizeof(RealScalar)));
	Eigen::Map<Eigen::VectorX<RealScalar>> hSpec(hSpecData, dim);
	// Check whether the memory is allocated correctly.
	hSpec = Eigen::VectorX<RealScalar>::Random(dim);

	Scalar* hEigVecsData = nullptr;
	MAGMA_CHECK(magma_malloc_pinned((void**)&hEigVecsData, LD * dim * sizeof(Scalar)));
	Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::Stride<LD, 1>> heigVecs(hEigVecsData, dim, dim);
	// Check whether the memory is allocated correctly.
	heigVecs = Eigen::MatrixX<Scalar>::NullaryExpr(
	    dim, dim, [&]() { return Scalar(dist(engine), dist(engine)); });

	magmaDoubleComplex* wA = nullptr;
	MAGMA_CHECK(magma_malloc_pinned((void**)&wA, LD * dim * sizeof(decltype(wA[0]))););
	// Check whether the memory is allocated correctly.
#pragma omp parallel for collapse(2)
	for(auto i = 0; i < dim; ++i) {
		for(auto j = 0; j < dim; ++j) {
			wA[i + j * LD] = MAGMA_Z_MAKE(heigVecs(i, j).real(), heigVecs(i, j).imag());
		}
	}
	std::cout << "# Allocated memories." << std::endl;

	magma_vec_t        jobz = MagmaVec;
	magma_uplo_t       uplo = MagmaLower;
	magmaDoubleComplex lwork, *work   = nullptr;
	RealScalar         lrwork, *rwork = nullptr;
	magma_int_t        liwork, *iwork = nullptr;
	int                info = 0;
	MAGMA_CHECK(
	    magma_zheevd_gpu(jobz, uplo, dim, reinterpret_cast<magmaDoubleComplex_ptr>(dMatData), LD,
	                     hSpec.data(), wA, LD, &lwork, -1, &lrwork, -1, &liwork, -1, &info));
	std::cout << "# Work sizes: lwork=" << int(real(lwork)) << ", lrwork=" << int(lrwork)
	          << ", liwork=" << liwork << std::endl;
	// std::cout << "# Work sizes: sizeof(decltype(work[0]))=" << sizeof(decltype(work[0]))
	//           << ", sizeof(decltype(rwork[0]))=" << sizeof(decltype(rwork[0]))
	//           << ", sizeof(decltype(iwork[0]))=" << sizeof(decltype(iwork[0])) << std::endl;
	MAGMA_CHECK(magma_malloc_pinned((void**)&work, int(real(lwork)) * sizeof(decltype(work[0]))));
	MAGMA_CHECK(magma_malloc_pinned((void**)&rwork, int(lrwork) * sizeof(decltype(rwork[0]))));
	MAGMA_CHECK(magma_malloc_pinned((void**)&iwork, liwork * sizeof(decltype(iwork[0]))));
	for(auto n = 0; n < Nsample; ++n) {
		hmat = RME(dim);
		magma_setmatrix(dim, dim, sizeof(Scalar), hmat.data(), LD, dMatData, LD, queue);
		// std::cout << "# Prepared a random matrix. hmat.cwiseAbs().maxCoeff() = "
		//           << hmat.cwiseAbs().maxCoeff() << std::endl;

		MAGMA_CHECK(magma_zheevd_gpu(
		    jobz, uplo, dim, reinterpret_cast<magmaDoubleComplex_ptr>(dMatData), LD, hSpec.data(),
		    wA, LD, work, int(real(lwork)), rwork, int(lrwork), iwork, liwork, &info));
		std::cout << "# Diagonalized a random matrix." << std::endl;

		magma_getmatrix(dim, dim, sizeof(Scalar), dMatData, LD, heigVecs.data(), LD, queue);
		// cudaMemcpy(heigVecs.data(), dMatData, LD * dim * sizeof(Scalar), cudaMemcpyDeviceToHost);
		std::cout << "# Copied eigenvectors to CPU." << std::endl;

		Eigen::MatrixX<Scalar> const temp = hmat.selfadjointView<Eigen::Lower>() * heigVecs;
		// std::cout << "# Calculated temp.\t temp.cwiseAbs().maxCoeff() = "
		//           << temp.cwiseAbs().maxCoeff()
		//           << ",\t hSpec.cwiseAbs().maxCoeff() = " << hSpec.cwiseAbs().maxCoeff()
		//           << ",\t heigVecs.cwiseAbs().maxCoeff() = " << heigVecs.cwiseAbs().maxCoeff()
		//           << std::endl;
		double diff = 0;
#pragma omp parallel for reduction(max : diff) collapse(2)
		for(auto i = 0; i < dim; ++i)
			for(auto j = 0; j < dim; ++j) {
				diff = std::max(diff, std::abs(temp(i, j) - heigVecs(i, j) * hSpec(j)));
			}
		std::cout << "# Sample(" << n << ")\t Error = " << diff << std::endl;
		REQUIRE(diff < precision);
	}

	MAGMA_CHECK(magma_free(dMatData));
	MAGMA_CHECK(magma_free_pinned(hMatData));
	MAGMA_CHECK(magma_free_pinned(hSpecData));
	MAGMA_CHECK(magma_free_pinned(hEigVecsData));
	MAGMA_CHECK(magma_free_pinned(wA));
	MAGMA_CHECK(magma_free_pinned(work));
	MAGMA_CHECK(magma_free_pinned(rwork));
	MAGMA_CHECK(magma_free_pinned(iwork));
	magma_finalize();
}