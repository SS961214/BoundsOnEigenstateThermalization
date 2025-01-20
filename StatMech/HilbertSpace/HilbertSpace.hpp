#pragma once

#include "typedefs.hpp"

template<class Derived>
class HilbertSpace;

template<>
class HilbertSpace<int> {
	private:
		Index m_dim;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param dim Dimension of the Hilbert space
		 */
		__host__ __device__ HilbertSpace(Index dim = 0) : m_dim{dim} {}

		HilbertSpace(HilbertSpace const& other)            = default;
		HilbertSpace& operator=(HilbertSpace const& other) = default;
		HilbertSpace(HilbertSpace&& other)                 = default;
		HilbertSpace& operator=(HilbertSpace&& other)      = default;
		~HilbertSpace()                                    = default;

		/*! @name Operator overloads */
		/* @{ */
		__host__ __device__ bool operator==(HilbertSpace const& other) const {
			return this->dim() == other.dim();
		}
		/* @} */

		__host__ __device__ Index dim() const { return m_dim; };
};

template<class Derived>
class HilbertSpace {
	public:
		__host__ __device__ Index dim() const {
			return static_cast<Derived const*>(this)->dim_impl();
		};

		/*! @name Operator overloads */
		/* @{ */
		__host__ __device__ bool operator==(HilbertSpace const& other) const {
			return this->dim() == other.dim();
		}
		/* @} */
};