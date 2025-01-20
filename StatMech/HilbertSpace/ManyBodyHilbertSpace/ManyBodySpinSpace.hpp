#pragma once

#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"

class ManyBodySpinSpace;
template<>
struct ManyBodySpaceTraits<ManyBodySpinSpace> {
		using LocalSpace = HilbertSpace<int>;
};
class ManyBodySpinSpace : public ManyBodySpaceBase<ManyBodySpinSpace> {
	private:
		using Base = ManyBodySpaceBase<ManyBodySpinSpace>;

	public:
		using LocalSpace = typename Base::LocalSpace;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param sysSize
		 * @param dimLoc dimension of a local spin (= 2S+1)
		 */
		__host__ __device__ ManyBodySpinSpace(Index sysSize = 0, Index dimLoc = 0)
		    : Base(sysSize, LocalSpace(dimLoc)) {}

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace<ManyBodySpinSpace>;
		__host__ __device__ Index dim_impl() const {
			if(this->sysSize() == 0) return 0;
			Index res = 1;
			for(int l = 0; l < this->sysSize(); ++l) { res *= this->dimLoc(); }
			return res;
		}
		/* @} */

		/*! @name Implementation for methods of parent class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase<ManyBodySpinSpace>;
		__host__ __device__ Index locState_impl(Index stateNum, int pos) const {
			assert(stateNum < this->dim());
			assert(static_cast<Index>(pos) < this->sysSize());
			for(auto l = 0; l != pos; ++l) stateNum /= this->dimLoc();
			return stateNum % this->dimLoc();
		}

		template<class Array>
		__host__ __device__ void ordinal_to_config_impl(Array&& config, Index stateNum) const {
			assert(stateNum < this->dim());
			config.resize(this->sysSize());
			for(int l = 0; l < this->sysSize(); ++l, stateNum /= this->dimLoc()) {
				config(l) = stateNum % this->dimLoc();
			}
			return;
		}

		template<class Array>
		__host__ __device__ Index config_to_ordinal_impl(Array const& config) const {
			assert(static_cast<Index>(config.size()) >= this->sysSize());
			Index res  = 0;
			Index base = 1;
			for(int l = 0; l < this->sysSize(); ++l, base *= this->dimLoc()) {
				res += config(l) * base;
			}
			return res;
		}

		template<typename... Args>
		__host__ __device__ Index translate_impl(Index stateNum, int trans, Args...) const {
			assert(stateNum < this->dim());
			assert(0 <= trans && static_cast<Index>(trans) < this->sysSize());
			Index base = 1;
			for(auto l = 0; l != trans; ++l) base *= this->dimLoc();
			Index const baseCompl = this->dim() / base;
			return stateNum / baseCompl + (stateNum % baseCompl) * base;
		}

		__host__ __device__ Index reverse_impl(Index stateNum) const {
			assert(stateNum < this->dim());
			Index res = 0, base = 1;
			for(int l = 0; l < this->sysSize(); ++l, base *= this->dimLoc()) {
				res += (this->dim() / base / this->dimLoc()) * ((stateNum / base) % this->dimLoc());
			}
			return res;
		}
		template<class Array>
		__host__ __device__ Index reverse_impl(Index stateNum, Array&) const {
			return this->reverse_impl(stateNum);
		}
		/* @} */
};