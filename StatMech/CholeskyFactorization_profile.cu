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

using Scalar = double;
// using Scalar = float;

int main(int argc, char* argv[]) {
	if(argc < 1) {
		std::cerr << "Usage: 0.(This) 1.(n) \n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	const int n = std::atoi(argv[1]);  // Example size, ensure it fits within a single block
	if constexpr(std::is_same_v<Scalar, double>) { std::cout << "# Using double precision\n"; }
	else { std::cout << "# Using single precision\n"; }
	// constexpr Scalar precision = (std::is_same_v<Scalar, double> ? 1e-10 : 1e-5);

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

	return EXIT_SUCCESS;
}