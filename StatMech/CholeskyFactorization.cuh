#pragma once
#include <Eigen/Dense>
#include <cuda/std/complex>

// TODO: Use fused maulitply-add operation for better performance and accuracy.

/**
 * @brief Do in-place forward substitution to solve the linear system Lx = b, where L is a lower-triangular matrix.
 *
 * @param L Square Lower-triangular matrix
 * @param b Right-hand side vectors
 */
template<class Matrix_, class Vector_>
__device__ void forwardSubstitution(Matrix_ const& L, Vector_&& b) {
	using Scalar        = typename std::decay_t<Vector_>::Scalar;
	using RealScalar    = typename Eigen::NumTraits<Scalar>::Real;
	using MatScalar     = typename std::decay_t<Matrix_>::Scalar;
	using MatRealScalar = typename Eigen::NumTraits<MatScalar>::Real;
	assert(L.rows() == L.cols());
	assert(L.cols() == b.rows());
	for(int col = threadIdx.x; col < b.cols(); col += blockDim.x) {
		for(int i = 0; i < L.rows(); ++i) {
			Scalar sum = b(i, col);
			for(int j = 0; j < i; ++j) {
				if constexpr(Eigen::NumTraits<Scalar>::IsComplex
				             && !Eigen::NumTraits<MatScalar>::IsComplex) {
					RealScalar const x
					    = __fma_rn(-L(i, j), cuda::std::real(b(j, col)), cuda::std::real(sum));
					RealScalar const y
					    = __fma_rn(-L(i, j), cuda::std::imag(b(j, col)), cuda::std::imag(sum));
					sum = Scalar(x, y);
				}
				else { sum = __fma_rn(-L(i, j), b(j, col), sum); }
				// sum -= L(i, j) * b(j, col);
			}
			b(i, col) = sum / L(i, i);
		}
	}
	__syncthreads();
}

/**
 * @brief Do in-place forward substitution to solve the linear system Lx = b, where L is a lower-triangular matrix.
 *
 * @param L Square Lower-triangular matrix
 * @param b Right-hand side vectors
 */
template<class Matrix_, class Vector_>
__device__ void forwardSubstitution_unitDiag(Matrix_ const& L, Vector_& b) {
	using Scalar = typename Vector_::Scalar;
	assert(L.rows() == L.cols());
	assert(L.cols() == b.rows());
	for(int col = threadIdx.x; col < b.cols(); col += blockDim.x) {
		for(int i = 0; i < L.rows(); ++i) {
			Scalar sum = b(i, col);
			for(int j = 0; j < i; ++j) sum -= L(i, j) * b(j, col);
			b(i, col) = sum;
		}
	}
	__syncthreads();
}

/**
 * @brief Do in-place back substitution to solve the linear system Ux = b, where U is an upper-triangular matrix.
 *
 * @param U Square Upper-triangular matrix
 * @param b Right-hand side vectors
 */
template<class Matrix_, class Vector_>
__device__ void backSubstitution(Matrix_ const& U, Vector_&& b) {
	using Scalar     = typename std::decay_t<Vector_>::Scalar;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	assert(U.rows() == U.cols());
	assert(U.cols() == b.rows());
	for(int col = threadIdx.x; col < b.cols(); col += blockDim.x) {
		for(int i = U.rows() - 1; i >= 0; --i) {
			Scalar sum = b(i, col);
			// Scalar res = 0.0, prev = sum;
			// for(int j = i + 1; j < U.rows(); ++j) sum -= U(i, j) * b(j, col);
			for(int j = U.rows() - 1; j > i; --j) {
				if constexpr(Eigen::NumTraits<Scalar>::IsComplex) {
					RealScalar const x
					    = __fma_rn(-U(i, j), cuda::std::real(b(j, col)), cuda::std::real(sum));
					RealScalar const y
					    = __fma_rn(-U(i, j), cuda::std::imag(b(j, col)), cuda::std::imag(sum));
					sum = Scalar(x, y);
				}
				else { sum = __fma_rn(-U(i, j), b(j, col), sum); }
				// sum -= U(i, j) * b(j, col);
				// res += (prev - sum) - U(i, j) * b(j, col);
				// prev = sum;
			}
			// sum += res;
			b(i, col) = sum / U(i, i);
		}
	}
	__syncthreads();
}

template<class Matrix_>
__device__ void utuFactorization_basic(Matrix_& A) {
	using Scalar = typename Matrix_::Scalar;
	assert(A.rows() == A.cols());
	int const n = A.rows();
	for(int i = 0; i < n; ++i) {  // Must be sequensial.
		for(int j = i + threadIdx.x; j < n; j += blockDim.x) {
			Scalar sum = A(i, j);
			for(int k = 0; k < i; ++k) {
				sum = __fma_rn(-A(k, i), A(k, j), sum);
				// sum -= A(k, i) * A(k, j);
			}
			A(i, j) = sum;
		}
		if(threadIdx.x == 0) A(i, i) = sqrt(A(i, i));
		__syncthreads();
		for(int j = i + 1 + threadIdx.x; j < n; j += blockDim.x) A(i, j) /= A(i, i);
		__syncthreads();
	}
}

/**
 * @brief Perform the UTU factorization of a matrix, where U is an upper-triangular matrix.
 *
 * @tparam Matrix_
 * @param mat
 * @param shMem
 * @param shMemSize
 */
template<class Matrix_>
__device__ void utuFactorization(Eigen::MatrixBase<Matrix_>& mat, typename Matrix_::Scalar* shMem,
                                 int shMemSize) {
	// mat: The matrix to be decomposed. It is decomposed as below:
	// mat = [A, B; B^T, C]
	using Scalar           = typename Matrix_::Scalar;
	constexpr int warpSize = 32;
	assert(shMemSize >= warpSize * warpSize);
	int const temp    = sqrt(shMemSize / sizeof(Scalar));
	int const shMemLD = int(temp / warpSize) * warpSize;
	assert(shMemLD * shMemLD <= shMemSize);
	using EigenMap = Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::OuterStride<>>;

	int head = 0;
	while(true) {
		int const dim      = mat.rows() - head;
		int const shMemDim = (dim < shMemLD ? dim : shMemLD);
		EigenMap  shMat(shMem, shMemDim, shMemDim, Eigen::OuterStride<>(shMemLD));
		int const dimA = shMat.rows();
		int const dimC = dim - dimA;
		// Decompose the matrix into blocks
		auto A = mat.block(head, head, dimA, dimA);
		auto B = mat.block(0 + head, dimA + head, dimA, dimC);
		auto C = mat.block(dimA + head, dimA + head, dimC, dimC);
		// Load the block "A" of "mat" to shared memory
		for(int i = threadIdx.x; i < dimA * dimA; i += blockDim.x) {
			int const row   = i % dimA;
			int const col   = i / dimA;
			shMat(row, col) = A(row, col);
		}
		__syncthreads();
		// Compute the utuFactorization of the block
		utuFactorization_basic(shMat);
		// Write the result back to "A"
		for(int i = threadIdx.x; i < dimA * dimA; i += blockDim.x) {
			int const row = i % dimA;
			int const col = i / dimA;
			A(row, col)   = shMat(row, col);
		}
		// __syncthreads();
		if(dimC <= 0) break;
		// Calculate the off-diagonal blocks
		forwardSubstitution(shMat.transpose(), B);
		// Update the "C" block
		for(int i = threadIdx.x; i < dimC * dimC; i += blockDim.x) {
			int const row = i % dimC;
			int const col = i / dimC;
			for(int k = 0; k < dimA; ++k) { C(row, col) -= B(k, row) * B(k, col); }
		}
		__syncthreads();
		head += dimA;
	}
}

template<class Matrix_>
__device__ void utduFactorization_basic(Matrix_& A) {
	using Scalar = typename Matrix_::Scalar;
	assert(A.rows() == A.cols());
	int const n = A.rows();
	for(int i = 0; i < n; ++i) {  // Must be sequensial.
		for(int j = i + threadIdx.x; j < n; j += blockDim.x) {
			Scalar sum = A(i, j);
			for(int k = 0; k < i; ++k) { sum -= A(k, i) * A(k, k) * A(k, j); }
			A(i, j) = sum;
		}
		__syncthreads();
		for(int j = i + 1 + threadIdx.x; j < n; j += blockDim.x) A(i, j) /= A(i, i);
		__syncthreads();
	}
}

template<class Matrix_>
__device__ void utduFactorization(Eigen::MatrixBase<Matrix_>& mat, typename Matrix_::Scalar* shMem,
                                  int shMemSize) {
	// mat: The matrix to be decomposed. It is decomposed as below:
	// mat = [A, B; B^T, C]
	using Scalar           = typename Matrix_::Scalar;
	constexpr int warpSize = 32;
	assert(shMemSize >= warpSize * warpSize);
	int const temp    = sqrt(shMemSize / sizeof(Scalar));
	int const shMemLD = int(temp / warpSize) * warpSize;
	assert(shMemLD * shMemLD <= shMemSize);
	using EigenMap = Eigen::Map<Eigen::MatrixX<Scalar>, 0, Eigen::OuterStride<>>;

	int head = 0;
	while(true) {
		int const dim      = mat.rows() - head;
		int const shMemDim = (dim < shMemLD ? dim : shMemLD);
		EigenMap  shMat(shMem, shMemDim, shMemDim, Eigen::OuterStride<>(shMemLD));
		int const dimA = shMat.rows();
		int const dimC = dim - dimA;
		// Decompose the matrix into blocks
		auto A = mat.block(head, head, dimA, dimA);
		auto B = mat.block(0 + head, dimA + head, dimA, dimC);
		auto C = mat.block(dimA + head, dimA + head, dimC, dimC);
		// Load the block "A" of "mat" to shared memory
		for(int i = threadIdx.x; i < dimA * dimA; i += blockDim.x) {
			int const row   = i % dimA;
			int const col   = i / dimA;
			shMat(row, col) = A(row, col);
		}
		__syncthreads();
		// Compute the utuFactorization of the block
		utduFactorization_basic(shMat);
		// Write the result back to "A"
		for(int i = threadIdx.x; i < dimA * dimA; i += blockDim.x) {
			int const row = i % dimA;
			int const col = i / dimA;
			A(row, col)   = shMat(row, col);
		}
		// __syncthreads();
		if(dimC <= 0) break;
		// Calculate the off-diagonal blocks
		forwardSubstitution_unitDiag(shMat.transpose(), B);
		// Scale the B block
		for(int i = threadIdx.x; i < dimC; i += blockDim.x) {
			for(int j = 0; j < dimA; ++j) B(j, i) /= shMat(j, j);
		}
		__syncthreads();
		// Update the "C" block
		for(int i = threadIdx.x; i < dimC * dimC; i += blockDim.x) {
			int const row = i % dimC;
			int const col = i / dimC;
			for(int k = 0; k < dimA; ++k) { C(row, col) -= B(k, row) * shMat(k, k) * B(k, col); }
		}
		__syncthreads();
		head += dimA;
	}
}