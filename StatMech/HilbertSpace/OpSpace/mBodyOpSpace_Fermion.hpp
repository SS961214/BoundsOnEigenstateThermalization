#pragma once

#include "mBodyOpSpace_Base.hpp"
#include "../ManyBodyHilbertSpace/ManyBodyFermionSpace.hpp"
#include "../Algorithm/Combination.hpp"
#include <Eigen/Dense>

// mBodyOpSpace for Fermion systems
template<typename Scalar_>
class mBodyOpSpace<ManyBodyFermionSpace, Scalar_>
    : public ManyBodyOpSpaceBase< mBodyOpSpace<ManyBodyFermionSpace, Scalar_> > {
	private:
		using Self = mBodyOpSpace<ManyBodyFermionSpace, Scalar_>;
		using Base = ManyBodyOpSpaceBase<Self>;
		using BIT  = typename ManyBodyFermionSpace::BIT;

	public:
		using BaseSpace  = typename Base::BaseSpace;  // ManyBodyFermionSpace
		using Scalar     = typename Base::Scalar;     // Scalar_
		using RealScalar = typename Base::RealScalar;
		using LocalSpace = typename Base::LocalSpace;

	private:
		Index       m_mBody = 0;
		Index       m_N     = 0;
		Combination m_opConfig;  // For creation operators
		// Operator is numbered as (CreationOp) * m_opConfig.dim() + (AnnihilationOp)

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param m
		 * @param sysSize
		 * @param N
		 */
		__host__ __device__ mBodyOpSpace(Index m = 0, Index sysSize = 0, Index N = 0)
		    : Base(BaseSpace(sysSize, N), sysSize, OpSpace<Scalar>(HilbertSpace<int>(N + 1))),
		      m_mBody{m},
		      m_N{N},
		      m_opConfig(sysSize, std::min(m, sysSize - N)) {
			cuASSERT(sysSize <= 32, "Error: sysSize > 32 is not supported.");
			printf("# mBodyOpSpace:\tdim = %d, sysSize = %d = %d, m = %d = %d, N = %d = %d\n",
			       int(this->dim()), int(sysSize), int(this->sysSize()), int(m), int(this->m()),
			       int(N), int(N));
		}

		/**
		 * @brief Custom constructor
		 *
		 * @param m
		 * @param baseSpace
		 */
		__host__ __device__ mBodyOpSpace(Index m, BaseSpace const& baseSpace)
		    : Base(baseSpace, baseSpace.sysSize(), OpSpace<Scalar>(baseSpace.locSpace())),
		      m_mBody{m},
		      m_opConfig(baseSpace.sysSize(), std::min(m, baseSpace.sysSize() - baseSpace.N())) {
			cuASSERT(baseSpace.sysSize() <= 32, "Error: sysSize > 32 is not supported");
		}

		mBodyOpSpace(mBodyOpSpace const&)            = default;
		mBodyOpSpace& operator=(mBodyOpSpace const&) = default;
		mBodyOpSpace(mBodyOpSpace&&)                 = default;
		mBodyOpSpace& operator=(mBodyOpSpace&&)      = default;
		~mBodyOpSpace()                              = default;

		__host__ __device__ Index m() const { return m_mBody; }
		__host__ __device__ Index N() const { return m_N; }
		__host__ __device__ Index maxOnSite() const { return 1; }

	private:
		/*! @name Implementation for methods of ancestor class HilbertSpace */
		/* @{ */
		friend HilbertSpace< mBodyOpSpace >;
		__host__ __device__ Index dim_impl() const { return m_opConfig.dim() * m_opConfig.dim(); }
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
			Index const cOpNum    = opNum / this->m_opConfig.dim();
			Index const aOpNum    = opNum % this->m_opConfig.dim();
			Index const cLocOpNum = this->m_opConfig.locNumber(cOpNum, pos);
			Index const aLocOpNum = this->m_opConfig.locNumber(aOpNum, pos);
			return cLocOpNum * (this->maxOnSite() + 1) + aLocOpNum;
		}

		template<class Array>
		__host__ __device__ void ordinal_to_config_impl(Array&& config, Index opNum) const {
			assert(opNum < this->dim());
			Index const cOpNum  = opNum / this->m_opConfig.dim();
			Index const aOpNum  = opNum % this->m_opConfig.dim();
			BIT const   cConfig = this->m_opConfig.ordinal_to_config(cOpNum);
			BIT const   aConfig = this->m_opConfig.ordinal_to_config(aOpNum);
			if constexpr(std::is_same_v<std::decay_t<Array>, BIT>) {
				config = (cConfig << 32) | aConfig;
			}
			else {
				assert(config.size() >= this->sysSize());
				for(auto pos = 0; pos < this->sysSize(); ++pos) {
					int const locC = (cConfig >> pos) & 1;
					int const locA = (aConfig >> pos) & 1;
					config(pos)    = locC * (this->maxOnSite() + 1) + locA;
				}
			}
			return;
		}
		template<class Array>
		__host__ __device__ Index config_to_ordinal_impl(Array const& config) const {
			BIT cConfig = 0;
			BIT aConfig = 0;
			if constexpr(std::is_same_v<std::decay_t<Array>, BIT>) {
				cConfig = config >> 32;
				aConfig = config & 0xFFFFFFFF;
			}
			else {
				assert(static_cast<Index>(config.size()) >= this->sysSize());
				for(int pos = 0; pos < this->sysSize(); ++pos) {
					int const locC = config(pos) / (this->maxOnSite() + 1);
					int const locA = config(pos) % (this->maxOnSite() + 1);
					cConfig |= (locC << pos);
					aConfig |= (locA << pos);
				}
			}
			Index const cOpNum = this->m_opConfig.config_to_ordinal(cConfig);
			Index const aOpNum = this->m_opConfig.config_to_ordinal(aConfig);
			return cOpNum * this->m_opConfig.dim() + aOpNum;
		}

		template<class Array>
		__host__ __device__ Index translate_impl(Index opNum, int trans, Array& work) const {
			return this->translate_impl(opNum, trans);
		}
		__host__ __device__ Index translate_impl(Index opNum, int trans) const {
			assert(opNum < this->dim());
			assert(0 <= trans && static_cast<Index>(trans) < this->sysSize());
			Index const cOpNum      = opNum / this->m_opConfig.dim();
			Index const aOpNum      = opNum % this->m_opConfig.dim();
			Index const translatedC = this->m_opConfig.translate(cOpNum, trans);
			Index const translatedA = this->m_opConfig.translate(aOpNum, trans);
			return translatedC * this->m_opConfig.dim() + translatedA;
		}

		template<class Array_>
		__host__ __device__ Index reverse_impl(Index opNum, Array_& work) const {
			return this->reverse_impl(opNum);
		}
		__host__ __device__ Index reverse_impl(Index opNum) const {
			assert(opNum < this->dim());
			Index const cOpNum    = opNum / this->m_opConfig.dim();
			Index const aOpNum    = opNum % this->m_opConfig.dim();
			Index const reversedC = this->m_opConfig.reverse(cOpNum);
			Index const reversedA = this->m_opConfig.reverse(aOpNum);
			return reversedC * this->m_opConfig.dim() + reversedA;
		}
		/* @} */
};

template<typename Scalar_>
__host__ __device__ inline Index mBodyOpSpace<ManyBodyFermionSpace, Scalar_>::actionWorkSize_impl()
    const {
	return 0;
}

template<typename Scalar_>
template<class Array>
__host__ __device__ inline void mBodyOpSpace<ManyBodyFermionSpace, Scalar_>::action_impl(
    Index& resStateNum, Scalar& coeff, Index opNum, Index stateNum, Array& work) const {
	(void)work;
	assert(opNum < this->dim());
	assert(stateNum < this->baseDim());
	resStateNum = stateNum;
	coeff       = 1.0;

	Index const cOpNum  = opNum / this->m_opConfig.dim();
	Index const aOpNum  = opNum % this->m_opConfig.dim();
	BIT const   cConfig = this->m_opConfig.ordinal_to_config(cOpNum);
	BIT const   aConfig = this->m_opConfig.ordinal_to_config(aOpNum);
	BIT         sConfig;
	this->baseSpace().ordinal_to_config(sConfig, stateNum);
	BIT const asConfig = sConfig & ~aConfig;  // State after the action of annihilation operators
	// Check if the annihilation operators are acting on filled sites
	if(aConfig != (aConfig & sConfig)) {
		coeff = 0;
		return;
	}
	// Check if the creation operators are acting on empty sites
	if(cConfig != (cConfig & ~asConfig)) {
		coeff = 0;
		return;
	}
	BIT const casConfig = cConfig | asConfig;
	resStateNum         = this->baseSpace().config_to_ordinal(casConfig);

	// std::cout << "inState: " << Eigen::RowVectorXi::NullaryExpr(this->sysSize(), [&sConfig](int i) {
	// 	return (sConfig >> i) & 1;
	// }) << std::endl;
	// std::cout << "aConfig: " << Eigen::RowVectorXi::NullaryExpr(this->sysSize(), [&aConfig](int i) {
	// 	return (aConfig >> i) & 1;
	// }) << std::endl;
	// std::cout << "cConfig: " << Eigen::RowVectorXi::NullaryExpr(this->sysSize(), [&cConfig](int i) {
	// 	return (cConfig >> i) & 1;
	// }) << std::endl;

	auto const countBitsOver = [](BIT const x, int const pos) {
		BIT const mask      = ~((1ULL << (pos + 1)) - 1);
		BIT const maskedVal = x & mask;
#ifdef __CUDA_ARCH__
		return __popcll(maskedVal);
#else
		return __builtin_popcountll(maskedVal);
#endif
	};

	int sign = 0;
	for(auto pos = 0; pos < this->sysSize(); ++pos) {
		int const locC = (cConfig >> pos) & 1;
		int const locA = (aConfig >> pos) & 1;
		if(locA == 1) {
			// sConfig &= ~(1ull << pos);
			sign ^= (countBitsOver(sConfig, pos) & 1);
		}
		if(locC == 1) {
			// sConfig |= (1ull << pos);
			sign ^= (countBitsOver(asConfig, pos) & 1);
		}
	}
	coeff = (sign & 1 ? -1 : +1);
	// resStateNum = this->baseSpace().config_to_ordinal(sConfig);
	// std::cout << "resState: "
	//           << Eigen::RowVectorXi::NullaryExpr(this->sysSize(),
	//                                              [&sConfig](int i) { return (sConfig >> i) & 1; })
	//           << std::endl;
	assert(resStateNum < this->baseDim());
	return;
}

#ifdef __NVCC__
	#include "mBodyOpSpace_Fermion.cuh"
#endif