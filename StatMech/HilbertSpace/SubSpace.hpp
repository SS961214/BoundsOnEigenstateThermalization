#pragma once

#include "typedefs.hpp"
#include "HilbertSpace.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>

template<class TotalSpace_, typename ScalarType_>
class SubSpace : public HilbertSpace< SubSpace<TotalSpace_, ScalarType_> > {
	public:
		using TotalSpace = TotalSpace_;
		using Scalar     = ScalarType_;
		using Real       = typename Eigen::NumTraits<Scalar>::Real;

	private:
		using SparseMatrix = typename Eigen::SparseMatrix<Scalar, 0, Eigen::Index>;
		TotalSpace   m_totalSpace;
		SparseMatrix m_basis;

	public:
		__host__ SubSpace(TotalSpace const& totalSpace)
		    : m_totalSpace{totalSpace}, m_basis(totalSpace.dim(), 0) {}
		__host__ SubSpace(TotalSpace&& totalSpace)
		    : m_totalSpace{std::move(totalSpace)}, m_basis(totalSpace.dim(), 0) {}

		SubSpace()                                 = default;
		SubSpace(SubSpace const& other)            = default;
		SubSpace& operator=(SubSpace const& other) = default;
		SubSpace(SubSpace&& other)                 = default;
		SubSpace& operator=(SubSpace&& other)      = default;
		~SubSpace()                                = default;

		__host__ __device__ TotalSpace const& totalSpace() const { return m_totalSpace; }
		__host__ __device__ Index             dimTot() const { return m_totalSpace.dim(); }

		__host__ SparseMatrix&       basis() { return m_basis; }
		__host__ SparseMatrix const& basis() const { return m_basis; }

	private:
		friend HilbertSpace< SubSpace<TotalSpace, Scalar> >;
		__host__ __device__ Index dim_impl() const { return m_basis.cols(); }
};