#pragma once

#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"
#include "Algorithm/IntegerComposition.hpp"

class ManyBodyBosonSpace;
template<>
struct ManyBodySpaceTraits<ManyBodyBosonSpace> {
		using LocalSpace = HilbertSpace<int>;
};

class ManyBodyBosonSpace : public ManyBodySpaceBase<ManyBodyBosonSpace> {
	private:
		using Base = ManyBodySpaceBase<ManyBodyBosonSpace>;

	public:
		using LocalSpace = typename Base::LocalSpace;

	private:
		IntegerComposition m_iComp;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param sysSize
		 * @param nBosons
		 */
		__host__ __device__ ManyBodyBosonSpace(Index sysSize = 0, Index nBosons = 0)
		    : Base(sysSize, LocalSpace(nBosons + 1)), m_iComp{nBosons, sysSize, nBosons} {}

		/**
		 * @brief Custom constructor 1
		 *
		 * @param sysSize
		 * @param max
		 * @param locSpace
		 */
		__host__ __device__ ManyBodyBosonSpace(Index sysSize, Index nBosons, Index max)
		    : Base(sysSize, LocalSpace(nBosons + 1)), m_iComp{nBosons, sysSize, max} {}

		__host__ __device__ Index N() const { return m_iComp.value(); }
		__host__ __device__ int   max() const { return m_iComp.max(); }

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace<ManyBodyBosonSpace>;
		__host__ __device__ Index dim_impl() const { return m_iComp.dim(); }
		/* @} */

		/*! @name Implementation for methods of parent class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase<ManyBodyBosonSpace>;
		__host__ __device__ Index locState_impl(Index stateNum, int pos) const {
			assert(stateNum < this->dim());
			assert(0 <= pos && static_cast<Index>(pos) < this->sysSize());
			return m_iComp.locNumber(stateNum, pos);
		}

		template<class Array>
		__host__ __device__ void ordinal_to_config_impl(Array&& config, Index stateNum) const {
			assert(stateNum < this->dim());
			return m_iComp.ordinal_to_config(config, stateNum);
		}

		template<class Array>
		__host__ __device__ Index config_to_ordinal_impl(Array const& config) const {
			assert(static_cast<Index>(config.size()) >= this->sysSize());
			return m_iComp.config_to_ordinal(config);
		}

		template<typename... Args>
		__host__ __device__ Index translate_impl(Index stateNum, int trans, Args&&... args) const {
			assert(stateNum < this->dim());
			if(!(0 <= trans && static_cast<Index>(trans) < this->sysSize())) {
				printf("# trans = %d, this->sysSize() = %d\n", int(trans), int(this->sysSize()));
			}
			assert(0 <= trans && static_cast<Index>(trans) < this->sysSize());
			return m_iComp.translate(stateNum, trans, std::forward<Args>(args)...);
		}

		__host__ Index reverse_impl(Index stateNum) const {
			Eigen::ArrayXi work(this->sysSize());
			return this->reverse_impl(stateNum, work);
		}
		template<class Array_>
		__host__ __device__ Index reverse_impl(Index stateNum, Array_& work) const {
			assert(stateNum < this->dim());
			return m_iComp.reverse(stateNum, work);
		}
		/* @} */
};