#pragma once

#include "../HilbertSpace/ObjectOnGPU.cuh"
#include <Eigen/Sparse>
#include <thrust/device_vector.h>

template<typename Scalar, typename Index = Eigen::Index>
class SparseMatrix {
	public:
		Index   m_outerSize  = 0;
		Index   m_innerSize  = 0;
		Index   m_nnz        = 0;  // number of non zeros
		Index*  m_outerIndex = nullptr;
		Index*  m_innerIndex = nullptr;
		Scalar* m_values     = nullptr;

	public:
		__host__ __device__ Index         cols() const { return m_outerSize; }
		__host__ __device__ Index         rows() const { return m_innerSize; }
		__host__ __device__ Index*        outerIndexPtr() { return m_outerIndex; }
		__host__ __device__ Index const*  outerIndexPtr() const { return m_outerIndex; }
		__host__ __device__ Index*        innerIndexPtr() { return m_innerIndex; }
		__host__ __device__ Index const*  innerIndexPtr() const { return m_innerIndex; }
		__host__ __device__ Scalar*       valuePtr() { return m_values; }
		__host__ __device__ Scalar const* valuePtr() const { return m_values; }
};

template<typename Scalar, typename Index>
__global__ void construct_SparseMatrix_kernel(SparseMatrix<Scalar, Index>* dptr,
                                              Index const outerSize, Index const innerSize,
                                              Index const nnz, Index* outerIndexPtr,
                                              Index* innerIndexPtr, Scalar* valuePtr) {
#ifndef NDEBUG
	int dev;
	cuCHECK(cudaGetDevice(&dev));
	printf("# (Device %d) %s: outerSize = %d,\tinnerSize = %d,\tnnz = %d\n", dev,
	       __PRETTY_FUNCTION__, int(outerSize), int(innerSize), int(nnz));
#endif
	new(dptr) SparseMatrix<Scalar, Index>();
	dptr->m_outerSize  = outerSize;
	dptr->m_innerSize  = innerSize;
	dptr->m_nnz        = nnz;
	dptr->m_outerIndex = outerIndexPtr;
	dptr->m_innerIndex = innerIndexPtr;
	dptr->m_values     = valuePtr;
}

template<typename Scalar, typename Index>
class ObjectOnGPU<SparseMatrix<Scalar, Index>> {
	private:
		using T   = SparseMatrix<Scalar, Index>;
		T*  m_ptr = nullptr;
		int m_dev = -1;

		thrust::device_vector<Index>  m_outerIndices;
		thrust::device_vector<Index>  m_innerIndices;
		thrust::device_vector<Scalar> m_values;

	public:
		ObjectOnGPU()                              = default;
		ObjectOnGPU(ObjectOnGPU const&)            = delete;
		ObjectOnGPU& operator=(ObjectOnGPU const&) = delete;
		ObjectOnGPU(ObjectOnGPU&& other)
		    : m_ptr(other.m_ptr),
		      m_dev(other.m_dev),
		      m_outerIndices(std::move(other.m_outerIndices)),
		      m_innerIndices(std::move(other.m_innerIndices)),
		      m_values(std::move(other.m_values)) {
			other.m_ptr = nullptr;
			other.m_dev = -1;
		}
		ObjectOnGPU& operator=(ObjectOnGPU&& other) {
			if(this != &other) {
				m_ptr          = other.m_ptr;
				m_dev          = other.m_dev;
				m_outerIndices = std::move(other.m_outerIndices);
				m_innerIndices = std::move(other.m_innerIndices);
				m_values       = std::move(other.m_values);
				other.m_ptr    = nullptr;
				other.m_dev    = -1;
			}
			return *this;
		}

		ObjectOnGPU(Eigen::SparseMatrix<Scalar, 0, Index> const& spmat) {
			assert(spmat.isCompressed());
			// Eigen::SparseMatrix<std::complex<double>> spmat2(spmat.template cast<std::complex<double>>());
			// spmat2.makeCompressed();
			// std::cout << spmat2 << std::endl;

			m_outerIndices = thrust::device_vector<Index>(
			    spmat.outerIndexPtr(), spmat.outerIndexPtr() + spmat.outerSize() + 1);
			m_innerIndices = thrust::device_vector<Index>(spmat.innerIndexPtr(),
			                                              spmat.innerIndexPtr() + spmat.nonZeros());
			m_values       = thrust::device_vector<Scalar>(spmat.valuePtr(),
			                                               spmat.valuePtr() + spmat.nonZeros());

			// std::cout << spmat.outerSize() << std::endl;
			// std::cout << spmat.nonZeros() << std::endl;
			// std::cout << "outerIndices.size() = " << outerIndices.size() << std::endl;
			// std::cout << "innerIndices.size() = " << innerIndices.size() << std::endl;
			// std::cout << "      values.size() = " << values.size() << std::endl;
			cuCHECK(cudaGetDevice(&m_dev));
			cuCHECK(cudaMalloc((void**)&m_ptr, sizeof(T)));
			construct_SparseMatrix_kernel<<<1, 1, 0, 0>>>(
			    m_ptr, spmat.outerSize(), spmat.innerSize(), spmat.nonZeros(),
			    m_outerIndices.data().get(), m_innerIndices.data().get(), m_values.data().get());
			cuCHECK(cudaGetLastError());
			cuCHECK(cudaDeviceSynchronize());
		}
		~ObjectOnGPU() {
			if(m_ptr != nullptr) {
				cuCHECK(cudaSetDevice(m_dev));
				destruct_object_kernel<<<1, 1>>>(m_ptr);
				cuCHECK(cudaGetLastError());
				cuCHECK(cudaDeviceSynchronize());
				cuCHECK(cudaFree(m_ptr));
			}
		}

		T*       ptr() { return m_ptr; }
		T const* ptr() const { return m_ptr; }
};