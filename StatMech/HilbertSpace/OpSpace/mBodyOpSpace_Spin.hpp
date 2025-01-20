#pragma once

#include "typedefs.hpp"
#include "mBodyOpSpace_Base.hpp"
#include "../ManyBodyHilbertSpace/ManyBodySpinSpace.hpp"
#include "../Algorithm/BaseConverter.hpp"
#include "../Algorithm/Combination.hpp"
#include <Eigen/Dense>

// mBodyOpSpace for Spin systems
template<typename Scalar_>
class mBodyOpSpace<ManyBodySpinSpace, Scalar_>
    : public ManyBodyOpSpaceBase< mBodyOpSpace<ManyBodySpinSpace, Scalar_> > {
	private:
		using Self = mBodyOpSpace<ManyBodySpinSpace, Scalar_>;
		using Base = ManyBodyOpSpaceBase<Self>;
		using BIT  = typename Combination::BIT;

	public:
		using BaseSpace  = typename Base::BaseSpace;  // ManyBodySpinSpace
		using Scalar     = typename Base::Scalar;     // Scalar_
		using RealScalar = typename Base::RealScalar;
		using LocalSpace = typename Base::LocalSpace;  //

	private:
		Index                                      m_mBody   = 0;
		Index                                      m_spinDim = 0;
		Combination                                m_actingSites;
		BaseConverter<Index>                       m_opConfig;
		static inline __host__ __device__ uint64_t flip(int const x) { return (~x >> 1) & 1; }
		static inline __host__ __device__ uint64_t phase(int const x) { return (x > 0); }
		static inline __host__ __device__ int      popcountll(uint64_t const x) {
#ifdef __CUDA_ARCH__
			return __popcll(x);
#else
			return __builtin_popcountll(x);
#endif
		}
		RealScalar m_normalization = 0.0;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param m
		 * @param sysSize
		 * @param spinDim
		 */
		__host__ __device__ mBodyOpSpace(Index m = 0, Index sysSize = 0, Index spinDim = 2)
		    : Base(BaseSpace(sysSize, spinDim), sysSize,
		           OpSpace<Scalar>(HilbertSpace<int>(spinDim))),
		      m_mBody{m},
		      m_spinDim{spinDim},
		      m_actingSites(sysSize, m),
		      m_opConfig{spinDim * spinDim - 1, m} {
			cuASSERT(spinDim == 2, "Error: Spin mBodyOpSpace currently supports only spinDim = 2.");
			cuASSERT(sysSize <= 32, "Error: sysSize > 32 is not supported.");
			printf("#\tdim = %d, sysSize = %d = %d, m = %d = %d, spinDim = %d = %d\n",
			       int(this->dim()), int(sysSize), int(this->sysSize()), int(m), int(this->m()),
			       int(spinDim), int(this->baseSpace().dimLoc()));

			m_normalization = double(1 << this->sysSize());
			m_normalization = 1.0 / sqrt(m_normalization);
		}

		/**
		 * @brief Custom constructor
		 *
		 * @param m
		 * @param baseSpace
		 */
		__host__ __device__ mBodyOpSpace(Index m, BaseSpace const& baseSpace)
		    : mBodyOpSpace(m, baseSpace.sysSize(), baseSpace.dimLoc()) {}

		mBodyOpSpace(mBodyOpSpace const&)            = default;
		mBodyOpSpace& operator=(mBodyOpSpace const&) = default;
		mBodyOpSpace(mBodyOpSpace&&)                 = default;
		mBodyOpSpace& operator=(mBodyOpSpace&&)      = default;
		~mBodyOpSpace()                              = default;

		__host__ __device__ Index m() const { return m_mBody; }
		__host__ __device__ Index spinDim() const { return m_spinDim; }

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace< mBodyOpSpace >;
		__host__ __device__ Index dim_impl() const {
			return m_actingSites.dim() * m_opConfig.max();
		}
		/* @} */

		/*! @name Implementation for methods of ancestor class OpSpaceBase */
		/* @{ */
		friend OpSpaceBase< mBodyOpSpace >;
		__host__ __device__ Index actionWorkSize_impl() const;
		template<class Array>
		__host__ __device__ void action_impl(Index& resStateNum, Scalar& coeff, Index opNum,
		                                     Index stateNum, Array& work) const;
		__host__ __device__ void action_impl(Index& resStateNum, Scalar& coeff, Index opNum,
		                                     Index stateNum) const {
#ifdef __CUDA_ARCH__
			static_assert([]() { return false; });
#endif
			Eigen::ArrayXi work(this->actionWorkSize());
			return this->action_impl(resStateNum, coeff, opNum, stateNum, work);
		}
		/* @} */

		/*! @name Implementation for methods of ancestor class ManyBodySpaceBase */
		/* @{ */
		friend ManyBodySpaceBase< mBodyOpSpace >;
		__host__ __device__ Index locState_impl(Index opNum, int pos) const {
			assert(opNum < this->dim());
			assert(0 <= pos && static_cast<Index>(pos) < this->sysSize());
			auto const posConfNum = opNum / m_opConfig.max();
			if(this->m_actingSites.locNumber(posConfNum, pos) == 0)
				return 0;
			else {
				BIT const      posConf   = m_actingSites.ordinal_to_config(posConfNum);
				uint64_t const mask      = (1ULL << pos) - 1;
				int const      index     = popcountll(posConf & mask);
				auto const     opConfNum = opNum % m_opConfig.max();
				return this->m_opConfig.digit(opConfNum, index) + 1;
			}
		}

		template<class Array>
		__host__ __device__ void ordinal_to_config_impl(Array&& config, Index opNum) const {
			assert(opNum < this->dim());
			auto const posConfNum = opNum / m_opConfig.max();
			auto       opConfNum  = opNum % m_opConfig.max();
			BIT const  posConf    = m_actingSites.ordinal_to_config(posConfNum);

			if constexpr(std::is_same_v<std::decay_t<Array>, BIT>) {
				static_assert([]() { return false; });
				config = 0;
				for(auto pos = 0; pos < this->sysSize(); ++pos) {
					if((~posConf >> pos) & 1) continue;
					auto const locOp = opConfNum % 3;
					opConfNum /= 3;
					config |= (phase(locOp) << (pos + 32)) | (flip(locOp) << pos);
				}
			}
			else {
				assert(config.size() >= this->sysSize());
				int index = 0;
				for(auto pos = 0; pos < this->sysSize(); ++pos) {
					config[pos] = ((~posConf >> pos) & 1)
					                  ? 0
					                  : 1 + this->m_opConfig.digit(opConfNum, index++);
				}
			}
			return;
		}
		template<class Array>
		__host__ __device__ Index config_to_ordinal_impl(Array const& config) const {
			Index opConfNum = 0, base = 1;
			BIT   posConf = 0ULL;

			if constexpr(std::is_same_v<std::decay_t<Array>, BIT>) {
				// static_assert([]() { return false; });
				BIT const mask      = (1ULL << this->sysSize()) - 1;
				BIT const flipConf  = config & mask;
				BIT const phaseConf = (config >> 32) & mask;

				posConf = flipConf | phaseConf;
				for(auto pos = 0; pos < this->sysSize(); ++pos) {
					if((~posConf >> pos) & 1) continue;
					BIT const phase   = (phaseConf >> pos) & 1;
					BIT const negflip = (~flipConf >> pos) & 1;
					opConfNum += ((negflip << 1) | (negflip ^ phase)) * base;
					base *= 3;
				}
			}
			else {
				assert(static_cast<Index>(config.size()) >= this->sysSize());
				for(auto pos = 0; pos < this->sysSize(); ++pos) {
					if(config(pos) == 0) continue;
					posConf |= (1ULL << pos);
					opConfNum += (config(pos) - 1) * base;
					base *= this->m_opConfig.radix();
				}
			}
			return opConfNum + this->m_actingSites.config_to_ordinal(posConf) * m_opConfig.max();
		}

		__host__ __device__ Index translate_impl(Index const opNum, int shift) const {
			assert(opNum < this->dim());
			assert(0 <= shift && static_cast<Index>(shift) <= this->sysSize());
			BIT const mask = (1ULL << this->sysSize()) - 1;
			BIT config;
			this->ordinal_to_config(config, opNum);
			BIT flipConf  = config & mask;
			BIT phaseConf = (config >> 32) & mask;
			flipConf      = (flipConf << shift) | (flipConf >> (this->sysSize() - shift));
			flipConf &= mask;
			phaseConf = (phaseConf << shift) | (phaseConf >> (this->sysSize() - shift));
			phaseConf &= mask;
			config = (phaseConf << 32) | flipConf;
			return this->config_to_ordinal(config);
		}
		template<class Array>
		__host__ __device__ Index translate_impl(Index const opNum, int shift, Array& work) const {
			(void)work;
			return this->translate_impl(opNum, shift);
		}

		__host__ __device__ Index reverse_impl(Index opNum) const {
			assert(opNum < this->dim());
			BIT reversed = 0;
			this->ordinal_to_config(reversed, opNum);
			reversed = this->m_actingSites.reverseBits(reversed);
			reversed = (reversed << 32) | (reversed >> 32);
			reversed >>= 32 - this->sysSize();

			return this->config_to_ordinal(reversed);
		}
		template<class Array>
		__host__ __device__ Index reverse_impl(Index const opNum, Array& work) const {
			(void)work;
			return this->reverse_impl(opNum);
		}
		/* @} */
};

template<typename Scalar_>
__host__ __device__ inline Index mBodyOpSpace<ManyBodySpinSpace, Scalar_>::actionWorkSize_impl()
    const {
	return 0;
}

template<typename Scalar_>
template<class Array>
__host__ __device__ inline void mBodyOpSpace<ManyBodySpinSpace, Scalar_>::action_impl(
    Index& resStateNum, Scalar& coeff, Index opNum, Index stateNum, Array& work) const {
	(void)work;
	assert(opNum < this->dim());
	assert(stateNum < this->baseDim());
	resStateNum = stateNum;
	coeff       = 1.0;

	BIT config;
	this->ordinal_to_config(config, opNum);
	BIT const flipConf  = config & ((1ULL << 32) - 1);
	BIT const phaseConf = config >> 32;

	resStateNum           = stateNum ^ flipConf;
	uint32_t const parity = popcountll(phaseConf & stateNum) & 1;
	uint32_t const ynum   = popcountll(flipConf & phaseConf) & 3;
	coeff = m_normalization * (parity ^ (ynum >> 1) ? -1 : 1) * (ynum & 1 ? Scalar(0, 1) : 1);
}

#ifdef __NVCC__
	#include "mBodyOpSpace_Spin.cuh"
#endif

// Trial 1. Total Test time (real) = 144.48 sec
// Trial 2. Total Test time (real) =  30.17 sec : Changing to Combination class
// Trial 3. Total Test time (real) =  12.31 sec : Totally migrating to Bit representations