#include "CholeskyFactorization.cuh"
#include <Eigen/Dense>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

#define cuCHECK(call)                                                          \
	{                                                                          \
		const cudaError_t error = call;                                        \
		if(error != cudaSuccess) {                                             \
			printf("cuCHECK Error: %s:%d,  ", __FILE__, __LINE__);             \
			printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
			assert(error == cudaSuccess);                                      \
		}                                                                      \
	};

template<typename Scalar>
__global__ void utuFactorization_basic_kernel(Scalar* A, int LD, int n) {
	Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::OuterStride<>> mat(A, n, n,
	                                                                Eigen::OuterStride<>(LD));
	utuFactorization_basic(mat);
}
template<typename Scalar>
__global__ void utduFactorization_basic_kernel(Scalar* A, int LD, int n) {
	Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::OuterStride<>> mat(A, n, n,
	                                                                Eigen::OuterStride<>(LD));
	utduFactorization_basic(mat);
}

template<typename Scalar>
__global__ void utuFactorization_kernel(Scalar* A, int LD, int n) {
	extern __shared__ Scalar shMem[];
	unsigned                 shMemSize;
	asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(shMemSize));
	if(threadIdx.x == 0) printf("# %s: shMemSize = %d\n", __func__, int(shMemSize));
	Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::OuterStride<>> mat(A, n, n,
	                                                                Eigen::OuterStride<>(LD));
	utuFactorization(mat, shMem, shMemSize);
}

template<typename Scalar>
__global__ void utduFactorization_kernel(Scalar* A, int LD, int n) {
	extern __shared__ Scalar shMem[];
	unsigned                 shMemSize;
	asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(shMemSize));
	if(threadIdx.x == 0) printf("# %s: shMemSize = %d\n", __func__, int(shMemSize));
	Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::OuterStride<>> mat(A, n, n,
	                                                                Eigen::OuterStride<>(LD));
	utduFactorization(mat, shMem, shMemSize);
}

template<typename Scalar>
__global__ void forwardSubstitution_kernel(Scalar* A, int LDA, int n, Scalar* B, int LDB, int m) {
	// Suppose U to be an upper triangular matrix
	Eigen::Map<Eigen::MatrixX<Scalar> const, 0, Eigen::OuterStride<>> matU(A, n, n,
	                                                                 Eigen::OuterStride<>(LDA));
	Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::OuterStride<>> matB(B, n, m,
	                                                                 Eigen::OuterStride<>(LDB));
	forwardSubstitution(matU.transpose(), matB);
}

template<typename Scalar>
__global__ void backSubstitution_kernel(Scalar* A, int LDA, int n, Scalar* B, int LDB, int m) {
	// Suppose U to be an upper triangular matrix
	Eigen::Map<Eigen::MatrixX<Scalar> const, 0, Eigen::OuterStride<>> matU(A, n, n,
	                                                                 Eigen::OuterStride<>(LDA));
	Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::OuterStride<>> matB(B, n, m,
	                                                                 Eigen::OuterStride<>(LDB));
	backSubstitution(matU, matB);
}

template<typename Scalar>
__global__ void solveForUTU_kernel(Scalar* A, int LDA, int n, Scalar* B, int LDB, int m) {
	// Suppose U to be an upper triangular matrix
	Eigen::Map<Eigen::MatrixX<Scalar> const, 0, Eigen::OuterStride<>> matU(A, n, n,
	                                                                 Eigen::OuterStride<>(LDA));
	Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::OuterStride<>> matB(B, n, m,
	                                                                 Eigen::OuterStride<>(LDB));
	forwardSubstitution(matU.transpose(), matB);
	backSubstitution(matU, matB);
}

using Scalar = double;
// using Scalar = float;

int main(int argc, char* argv[]) {
	if(argc < 2) {
		std::cerr << "Usage: 0.(This) 1.(n) \n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	const int n = std::atoi(argv[1]);  // Example size, ensure it fits within a single block
	if constexpr(std::is_same_v<Scalar, double>) { std::cout << "# Using double precision\n"; }
	else { std::cout << "# Using single precision\n"; }
	constexpr Scalar precision = (std::is_same_v<Scalar, double> ? 1e-10 : 1e-5);

	cudaEvent_t start, stop;
	float       elapsedTime;

	// Initialize host matrix using Eigen
	std::random_device         rd;
	std::mt19937               gen(rd());
	std::normal_distribution<> dis(0, 1);
	Eigen::MatrixX<Scalar>     temp
	    = Eigen::MatrixX<Scalar>::NullaryExpr(n, n, [&]() { return dis(gen); });
	temp = (temp * temp.transpose()).eval();  // Ensure symmetric positive definite
	temp /= temp.norm();                      // Normalize
	Eigen::MatrixX<Scalar> const h_A = temp;

	{  // Test utuFactorization_basic
		// Copy host matrix to device vector
		thrust::device_vector<Scalar> d_A(h_A.data(), h_A.data() + h_A.size());
		// Launch kernel
		cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, utuFactorization_basic_kernel<Scalar>));
		int const nThreads  = attr.maxThreadsPerBlock;
		int const shMemSize = attr.maxDynamicSharedSizeBytes;
		std::cout << "# utuFactorization_basic_kernel: nThreads = " << nThreads
		          << ", shMemSize = " << shMemSize << std::endl;
		cudaEventCreate(&start);
		cudaEventRecord(start, 0);
		utuFactorization_basic_kernel<<<1, nThreads>>>(d_A.data().get(), n, n);
		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		// Check the result
		thrust::host_vector<Scalar>        h_Res(d_A);
		Eigen::Map<Eigen::MatrixX<Scalar>> h_ResMat(h_Res.data(), n, n);
		Eigen::MatrixX<Scalar>             resU = Eigen::MatrixX<Scalar>::Zero(n, n);
		for(int i = 0; i < n; ++i)
			for(int j = i; j < n; ++j) resU(i, j) = h_ResMat(i, j);
		Eigen::MatrixX<Scalar> const resRecA = resU.transpose() * resU;
		// std::cout << "Reconstructed matrix A:\n" << resRecA << std::endl;
		Scalar const error = (h_A - resRecA).norm();
		std::cout << "# utuFactorization_basic: Reconstruction error = " << error
		          << ",\t Elapsed time = " << elapsedTime << " (ms)" << std::endl;
		assert(error < precision);
		std::cout << std::endl;
	}

	{  // Test utduFactorization_basic
		// Copy host matrix to device vector
		thrust::device_vector<Scalar> d_A(h_A.data(), h_A.data() + h_A.size());
		// Launch kernel
		cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, utduFactorization_basic_kernel<Scalar>));
		int const nThreads  = attr.maxThreadsPerBlock;
		int const shMemSize = attr.maxDynamicSharedSizeBytes;
		std::cout << "# utduFactorization_basic_kernel: nThreads = " << nThreads
		          << ", shMemSize = " << shMemSize << std::endl;
		cudaEventCreate(&start);
		cudaEventRecord(start, 0);
		utduFactorization_basic_kernel<<<1, nThreads>>>(d_A.data().get(), n, n);
		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		// Check the result
		thrust::host_vector<Scalar>        h_Res(d_A);
		Eigen::Map<Eigen::MatrixX<Scalar>> h_ResMat(h_Res.data(), n, n);
		Eigen::MatrixX<Scalar>             resU = Eigen::MatrixX<Scalar>::Zero(n, n);
		for(int i = 0; i < n; ++i) {
			resU(i, i) = 1;
			for(int j = i + 1; j < n; ++j) resU(i, j) = h_ResMat(i, j);
		}
		Eigen::VectorX<Scalar> const resD    = h_ResMat.diagonal();
		Eigen::MatrixX<Scalar> const resRecA = resU.transpose() * resD.asDiagonal() * resU;
		// std::cout << "Reconstructed matrix A:\n" << resRecA << std::endl;
		Scalar const error = (h_A - resRecA).norm();
		std::cout << "# utduFactorization_basic: Reconstruction error = " << error
		          << ",\t Elapsed time = " << elapsedTime << " (ms)" << std::endl;
		assert(error < precision);
		std::cout << std::endl;
	}

	{  // Test utuFactorization
		// Copy host matrix to device vector
		thrust::device_vector<Scalar> d_A(h_A.data(), h_A.data() + h_A.size());

		// Launch kernel
		cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, utuFactorization_kernel<Scalar>));
		int const nThreads  = attr.maxThreadsPerBlock;
		int const shMemSize = attr.maxDynamicSharedSizeBytes;
		std::cout << "# utuFactorization_kernel: nThreads = " << nThreads
		          << ", shMemSize = " << shMemSize << std::endl;
		cudaEventCreate(&start);
		cudaEventRecord(start, 0);
		utuFactorization_kernel<<<1, nThreads, shMemSize>>>(d_A.data().get(), n, n);
		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		thrust::host_vector<Scalar>        h_Res(d_A);
		Eigen::Map<Eigen::MatrixX<Scalar>> h_ResMat(h_Res.data(), n, n);
		Eigen::MatrixX<Scalar>             resU = Eigen::MatrixX<Scalar>::Zero(n, n);
		for(int i = 0; i < n; ++i)
			for(int j = i; j < n; ++j) resU(i, j) = h_ResMat(i, j);
		Eigen::MatrixX<Scalar> const resRecA = resU.transpose() * resU;
		// std::cout << "Reconstructed matrix A:\n" << resRecA << std::endl;
		Scalar const error = (h_A - resRecA).norm();
		std::cout << "# utuFactorization: Reconstruction error = " << error
		          << ",\t Elapsed time = " << elapsedTime << " (ms)" << std::endl;
		assert(error < precision);
		std::cout << std::endl;
	}

	{  // Test utduFactorization
		// Copy host matrix to device vector
		thrust::device_vector<Scalar> d_A(h_A.data(), h_A.data() + h_A.size());

		// Launch kernel
		cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, utduFactorization_kernel<Scalar>));
		int const nThreads  = attr.maxThreadsPerBlock;
		int const shMemSize = attr.maxDynamicSharedSizeBytes;
		std::cout << "# utduFactorization_kernel: nThreads = " << nThreads
		          << ", shMemSize = " << shMemSize << std::endl;
		cudaEventCreate(&start);
		cudaEventRecord(start, 0);
		utduFactorization_kernel<<<1, nThreads, shMemSize>>>(d_A.data().get(), n, n);
		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		thrust::host_vector<Scalar>        h_Res(d_A);
		Eigen::Map<Eigen::MatrixX<Scalar>> h_ResMat(h_Res.data(), n, n);
		Eigen::MatrixX<Scalar>             resU = Eigen::MatrixX<Scalar>::Zero(n, n);
		for(int i = 0; i < n; ++i) {
			resU(i, i) = 1;
			for(int j = i + 1; j < n; ++j) resU(i, j) = h_ResMat(i, j);
		}
		Eigen::VectorX<Scalar> const resD    = h_ResMat.diagonal();
		Eigen::MatrixX<Scalar> const resRecA = resU.transpose() * resD.asDiagonal() * resU;
		// std::cout << "Reconstructed matrix A:\n" << resRecA << std::endl;
		Scalar const error = (h_A - resRecA).norm();
		std::cout << "# utduFactorization: Reconstruction error = " << error
		          << ",\t Elapsed time = " << elapsedTime << " (ms)" << std::endl;
		assert(error < precision);
		std::cout << std::endl;
	}

	// Test forwardSubstitution and backSubstitution
	{
		Eigen::LLT<Eigen::MatrixX<Scalar>> llt(h_A);
		Eigen::MatrixX<Scalar> const       U = llt.matrixL().transpose();
		{
			Scalar const err = (U.transpose() * U - h_A).cwiseAbs().maxCoeff();
			assert(err < precision);
		}
		if(n <= 10) std::cout << "U:\n" << U << std::endl;

		Eigen::MatrixX<Scalar> temp
		    = Eigen::MatrixX<Scalar>::NullaryExpr(h_A.rows(), n, [&]() { return dis(gen); });
		for(int i = 0; i < n; ++i) temp.col(i) /= temp.col(i).norm();
		Eigen::MatrixX<Scalar> const B        = temp;
		Eigen::MatrixX<Scalar> const solution = llt.solve(B);
		{
			Scalar const err = (h_A * solution - B).cwiseAbs().maxCoeff();
			std::cout << "# Eigen LLT solve: err = " << err << std::endl;
			// assert(err < precision);
		}

		thrust::device_vector<Scalar> d_U(U.data(), U.data() + U.size());
		{  // Test forwardSubstitution
			thrust::device_vector<Scalar> d_B(B.data(), B.data() + B.size());
			// Launch kernel
			cudaFuncAttributes attr;
			cuCHECK(cudaFuncGetAttributes(&attr, forwardSubstitution_kernel<Scalar>));
			int const nThreads  = attr.maxThreadsPerBlock;
			int const shMemSize = attr.maxDynamicSharedSizeBytes;
			forwardSubstitution_kernel<<<1, nThreads>>>(d_U.data().get(), n, n, d_B.data().get(), n,
			                                            n);
			thrust::host_vector<Scalar>        h_X(d_B);
			Eigen::Map<Eigen::MatrixX<Scalar>> X(h_X.data(), n, n);
			Scalar const error = (U.transpose() * X - B).cwiseAbs().maxCoeff();
			std::cout << "# forwardSubstitution: error = " << error << std::endl;
			//   << ",\t Elapsed time = " << elapsedTime << " (ms)" << std::endl;
			assert(error < precision);
		}
		{  // Test backSubstitution
			thrust::device_vector<Scalar> d_B(B.data(), B.data() + B.size());
			// Launch kernel
			cudaFuncAttributes attr;
			cuCHECK(cudaFuncGetAttributes(&attr, backSubstitution_kernel<Scalar>));
			int const nThreads  = attr.maxThreadsPerBlock;
			int const shMemSize = attr.maxDynamicSharedSizeBytes;
			backSubstitution_kernel<<<1, nThreads>>>(d_U.data().get(), n, n, d_B.data().get(), n,
			                                         n);
			thrust::host_vector<Scalar>        h_X(d_B);
			Eigen::Map<Eigen::MatrixX<Scalar>> X(h_X.data(), n, n);
			Scalar const                       error = (U * X - B).cwiseAbs().maxCoeff();
			std::cout << "# backSubstitution: error = " << error << std::endl;
			//   << ",\t Elapsed time = " << elapsedTime << " (ms)" << std::endl;
			assert(error < precision);
		}
		{  // Test the combitation of forwardSubstitution and backSubstitution to solve U^T U x = B
			thrust::device_vector<Scalar> d_B(B.data(), B.data() + B.size());
			// Launch kernel
			cudaFuncAttributes attr;
			cuCHECK(cudaFuncGetAttributes(&attr, solveForUTU_kernel<Scalar>));
			int const nThreads  = attr.maxThreadsPerBlock;
			int const shMemSize = attr.maxDynamicSharedSizeBytes;
			solveForUTU_kernel<<<1, nThreads>>>(d_U.data().get(), n, n, d_B.data().get(), n, n);
			thrust::host_vector<Scalar>        h_X(d_B);
			Eigen::Map<Eigen::MatrixX<Scalar>> X(h_X.data(), n, n);
			Scalar const                       error = (h_A * X - B).cwiseAbs().maxCoeff();
			std::cout << "# solveForUTU: error = " << error << std::endl;
			//   << ",\t Elapsed time = " << elapsedTime << " (ms)" << std::endl;
			assert(error < precision);
		}
	}

	return EXIT_SUCCESS;
}