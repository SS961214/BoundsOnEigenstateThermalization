#pragma once

#include "mBodyOpSpace_Base.hpp"
#include "../ManyBodyHilbertSpace/ManyBodyBosonSpace.hpp"
#include "../Algorithm/IntegerComposition.hpp"
#include <Eigen/Dense>

// mBodyOpSpace for Boson systems
template<typename Scalar_>
class mBodyOpSpace<ManyBodyBosonSpace, Scalar_>
    : public ManyBodyOpSpaceBase< mBodyOpSpace<ManyBodyBosonSpace, Scalar_> > {
	private:
		using Self = mBodyOpSpace<ManyBodyBosonSpace, Scalar_>;
		using Base = ManyBodyOpSpaceBase<Self>;

	public:
		using BaseSpace  = typename Base::BaseSpace;  // ManyBodyBosonSpace
		using Scalar     = typename Base::Scalar;     // Scalar_
		using RealScalar = typename Base::RealScalar;
		using LocalSpace = typename Base::LocalSpace;

	private:
		Index              m_mBody = 0;
		Index              m_N     = 0;
		IntegerComposition m_opConfig;  // For creation operators
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
		      m_opConfig{m, sysSize, m} {
			printf("#\tdim = %d, sysSize = %d = %d, m = %d = %d, N = %d = %d\n", int(this->dim()),
			       int(sysSize), int(this->sysSize()), int(m), int(this->m()), int(N), int(N));
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
		      m_opConfig(m, baseSpace.sysSize(), baseSpace.max()) {}

		mBodyOpSpace(mBodyOpSpace const&)            = default;
		mBodyOpSpace& operator=(mBodyOpSpace const&) = default;
		mBodyOpSpace(mBodyOpSpace&&)                 = default;
		mBodyOpSpace& operator=(mBodyOpSpace&&)      = default;
		~mBodyOpSpace()                              = default;

		__host__ __device__ Index m() const { return m_mBody; }
		__host__ __device__ Index N() const { return m_N; }
		__host__ __device__ Index maxOnSite() const { return m_opConfig.max(); }

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
			return cLocOpNum * (this->m_opConfig.max() + 1) + aLocOpNum;
		}

		template<class Array>
		__host__ __device__ void ordinal_to_config_impl(Array&& config, Index opNum) const {
			assert(opNum < this->dim());
			Index const cOpNum = opNum / this->m_opConfig.dim();
			Index const aOpNum = opNum % this->m_opConfig.dim();
			assert(config.size() >= this->sysSize());
			this->m_opConfig.ordinal_to_config(config, aOpNum);
			{
				Index ordinal_copy = cOpNum;
				Index z = 0, zPrev = 0;
				for(auto l = 1; l != this->sysSize(); ++l) {
					while(this->m_opConfig.workA(z, l) <= ordinal_copy) {
						assert(0 <= z && z < this->m_opConfig.workA().rows());
						assert(0 <= l && l < this->m_opConfig.workA().cols());
						z += 1;
					}
					config(this->sysSize() - l) += (z - zPrev) * (this->m_opConfig.max() + 1);
					zPrev = z;
					ordinal_copy -= this->m_opConfig.workB(z, l);
				}
				config(0) += (this->m_opConfig.value() - z) * (this->m_opConfig.max() + 1);
			}
			return;
		}
		template<class Array>
		__host__ __device__ Index config_to_ordinal_impl(Array const& config) const {
			assert(static_cast<Index>(config.size()) >= this->sysSize());
			class cConfig_wapper {
				private:
					Array const& m_config;
					Index const  m_max;

				public:
					__host__ __device__ cConfig_wapper(Array const& config, Index m)
					    : m_config{config}, m_max{m} {}
					__host__ __device__ Index size() const { return m_config.size(); }
					__host__ __device__ Index operator()(int pos) const {
						return m_config(pos) / (m_max + 1);
					}
			};
			cConfig_wapper cConfig(config, this->m_opConfig.max());

			class aConfig_wapper {
				private:
					Array const& m_config;
					Index const  m_max;

				public:
					__host__ __device__ aConfig_wapper(Array const& config, Index m)
					    : m_config{config}, m_max{m} {}
					__host__ __device__ Index size() const { return m_config.size(); }
					__host__ __device__ Index operator()(int pos) const {
						return m_config(pos) % (m_max + 1);
					}
			};
			aConfig_wapper aConfig(config, this->m_opConfig.max());

			Index const cOpNum = this->m_opConfig.config_to_ordinal(cConfig);
			Index const aOpNum = this->m_opConfig.config_to_ordinal(aConfig);

			return cOpNum * this->m_opConfig.dim() + aOpNum;
		}

		template<class Array>
		__host__ __device__ Index translate_impl(Index opNum, int trans, Array& work) const {
			assert(opNum < this->dim());
			assert(0 <= trans && static_cast<Index>(trans) < this->sysSize());
			assert(work.size() >= this->sysSize());
			this->ordinal_to_config(work, opNum);
			class translatedConfig_wapper {
				private:
					Array const& m_config;
					Index const  m_trans;
					Index const  m_length;

				public:
					__host__ __device__ translatedConfig_wapper(Array const& config, Index trans,
					                                            Index L)
					    : m_config{config}, m_trans{trans}, m_length{L} {
						assert(m_config.size() >= m_length);
					}
					__host__ __device__ Index size() const { return m_config.size(); }
					__host__ __device__ Index operator()(int pos) const {
						return m_config((pos + m_trans) % m_length);
					}
			};
			translatedConfig_wapper const translated(work, trans, this->sysSize());
			return this->config_to_ordinal(translated);
		}
		__host__ Index translate_impl(Index opNum, int trans) const {
			assert(opNum < this->dim());
			assert(0 <= trans && static_cast<Index>(trans) < this->sysSize());
			Eigen::ArrayX<Index> work(this->sysSize());
			return this->translate(opNum, trans, work);
		}

		template<class Array_>
		__host__ __device__ Index reverse_impl(Index opNum, Array_& work) const {
			assert(opNum < this->dim());
			assert(work.size() >= this->sysSize());
			this->ordinal_to_config(work, opNum);
			class reversedConfig_wapper {
				private:
					Array_ const& m_config;
					Index const   m_length;

				public:
					__host__ __device__ reversedConfig_wapper(Array_ const& config, Index L)
					    : m_config{config}, m_length{L} {
						assert(m_config.size() >= m_length);
					}
					__host__ __device__ Index size() const { return m_config.size(); }
					__host__ __device__ Index operator()(int pos) const {
						return m_config(m_length - 1 - pos);
					}
			};
			reversedConfig_wapper const reversed(work, this->sysSize());
			return this->config_to_ordinal(reversed);
		}
		__host__ Index reverse_impl(Index opNum) const {
			Eigen::ArrayXi work(this->sysSize());
			return this->reverse_impl(opNum, work);
		}
		/* @} */
};

template<typename Scalar_>
__host__ __device__ inline Index mBodyOpSpace<ManyBodyBosonSpace, Scalar_>::actionWorkSize_impl()
    const {
	return 2 * this->sysSize();
}

template<typename Scalar_>
template<class Array>
__host__ __device__ inline void mBodyOpSpace<ManyBodyBosonSpace, Scalar_>::action_impl(
    Index& resStateNum, Scalar& coeff, Index opNum, Index stateNum, Array& work) const {
	assert(opNum < this->dim());
	assert(stateNum < this->baseDim());
	assert(static_cast<Index>(work.size()) >= this->actionWorkSize());

	this->ordinal_to_config(work, opNum);
	class OpConfig {
		private:
			Array const& m_config;
			Index const  m_offset1;
			Index const  m_offset2;

		public:
			__host__ __device__ OpConfig(Array const& config, Index offset1, Index offset2)
			    : m_config{config}, m_offset1{offset1}, m_offset2{offset2} {};
			__host__ __device__ Index operator()(int pos) const {
				return (m_config(pos) / m_offset1) % m_offset2;
			}
	};
	OpConfig const aConfig(work, 1, this->m_opConfig.max() + 1);
	OpConfig const cConfig(work, this->m_opConfig.max() + 1, this->m_opConfig.max() + 1);

	class StateConfig {
		private:
			Array&      m_config;
			Index const m_shift;
			Index const m_size;

		public:
			__host__ __device__ StateConfig(Array& config, Index shift, Index size)
			    : m_config{config}, m_shift{shift}, m_size{size} {}
			__host__ __device__ auto&       operator()(int pos) { return m_config(pos + m_shift); }
			__host__ __device__ auto const& operator()(int pos) const {
				return m_config(pos + m_shift);
			}
			__host__ __device__ Index size() const { return m_size; }
	};
	StateConfig stateConfig(work, this->sysSize(), this->sysSize());
	this->baseSpace().ordinal_to_config(stateConfig, stateNum);
	// std::cout << "#\t" << stateConfig.transpose() << std::endl;
	// std::cout << "#\t";
	// for(auto pos = 0; pos < this->sysSize(); ++pos) {
	// 	std::cout <<"(" << cConfig(pos) << "," << aConfig(pos) << ") ";
	// }
	// std::cout << std::endl;
	// std::cout << "#\t this->m_opConfig.max() = " << this->m_opConfig.max() << std::endl;

	resStateNum = stateNum;
	coeff       = 1.0;

	for(auto pos = 0; pos < this->sysSize(); ++pos) {
		if(aConfig(pos) > stateConfig(pos)) {
			coeff = 0;
			return;
		}
		if(stateConfig(pos) + cConfig(pos) - aConfig(pos) > this->baseSpace().max()) {
			coeff = 0;
			return;
		}
		std::decay_t<decltype(coeff)> mult = 1.0;
		for(int j = 0; j < aConfig(pos); ++j) {
			mult *= std::sqrt(stateConfig(pos));
			stateConfig(pos) -= 1;
		}
		for(int j = 0; j < cConfig(pos); ++j) {
			stateConfig(pos) += 1;
			mult *= std::sqrt(stateConfig(pos));
		}
		coeff *= mult;
	}
	resStateNum = this->baseSpace().config_to_ordinal(stateConfig);
	assert(resStateNum < this->baseDim());
	return;
}

#ifdef __NVCC__
	#include "mBodyOpSpace_Boson.cuh"
#endif