#pragma once

#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"
#include "Algorithm/Combination.hpp"

class ManyBodyFermionSpace;
template<>
struct ManyBodySpaceTraits<ManyBodyFermionSpace> {
		using LocalSpace = HilbertSpace<int>;
};

class ManyBodyFermionSpace : public ManyBodySpaceBase<ManyBodyFermionSpace> {
	private:
		using Base = ManyBodySpaceBase<ManyBodyFermionSpace>;

	public:
		using LocalSpace = typename Base::LocalSpace;
		using BIT        = typename Combination::BIT;

	private:
		Combination m_Comb;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param sysSize
		 * @param nFermions
		 */
		__host__ __device__ ManyBodyFermionSpace(Index sysSize = 0, Index nFermions = 0)
		    : Base(sysSize, LocalSpace(2)), m_Comb(sysSize, nFermions) {}

		__host__ __device__ Index N() const { return m_Comb.N(); }
		__host__ __device__ int   max() const { return 1; }

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace<ManyBodyFermionSpace>;
		__host__ __device__ Index dim_impl() const { return m_Comb.dim(); }
		/* @} */

		/*! @name Implementation for methods of parent class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase<ManyBodyFermionSpace>;
		__host__ __device__ Index locState_impl(Index stateNum, int pos) const {
			assert(stateNum < this->dim());
			assert(0 <= pos && static_cast<Index>(pos) < this->sysSize());
			return m_Comb.locNumber(stateNum, pos);
		}

		template<class Array>
		__host__ __device__ void ordinal_to_config_impl(Array&& config, Index stateNum) const {
			assert(stateNum < this->dim());
			if constexpr(std::is_same_v<std::decay_t<Array>, BIT>) {
				config = m_Comb.ordinal_to_config(stateNum);
				return;
			}
			else {
				auto const configBit = m_Comb.ordinal_to_config(stateNum);
				for(int i = 0; i < this->sysSize(); ++i) config(i) = (configBit >> i) & 1;
				return;
			}
			return;
		}

		template<class Array>
		__host__ __device__ Index config_to_ordinal_impl(Array const& config) const {
			if constexpr(std::is_same_v<std::decay_t<Array>, BIT>) { return m_Comb.config_to_ordinal(config); }
			else {
				assert(static_cast<Index>(config.size()) >= this->sysSize());
				BIT configBit = 0;
				for(int i = 0; i < this->sysSize(); ++i) configBit |= (config(i) << i);
				return m_Comb.config_to_ordinal(configBit);
			}
			return 0;
		}

		template<typename... Args>
		__host__ __device__ Index translate_impl(Index stateNum, int trans, Args&&... args) const {
			assert(stateNum < this->dim());
			assert(0 <= trans && static_cast<Index>(trans) < this->sysSize());
			return m_Comb.translate(stateNum, trans);
		}

		__host__ __device__ Index reverse_impl(Index stateNum) const { return m_Comb.reverse(stateNum); }
		template<class Array_>
		__host__ __device__ Index reverse_impl(Index stateNum, Array_& work) const {
			assert(stateNum < this->dim());
			return this->reverse_impl(stateNum);
		}
		/* @} */
};