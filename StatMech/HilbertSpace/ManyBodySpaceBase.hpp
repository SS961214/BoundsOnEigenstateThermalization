#pragma once

#include "typedefs.hpp"
#include "HilbertSpace.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <execution>

template<class Derived>
struct ManyBodySpaceTraits;
// OpSpaceTraits should define the following properties:
// - LocalSpace

template<class Derived>
class ManyBodySpaceBase : public HilbertSpace<Derived> {
	public:
		using LocalSpace = typename ManyBodySpaceTraits<Derived>::LocalSpace;

	private:
		Index      m_sysSize = 0;
		LocalSpace m_locSpace;

	public:
		/**
		 * @brief Custom constructor
		 *
		 * @param systemSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodySpaceBase(Index sysSize, LocalSpace const& locSpace)
		    : m_sysSize{sysSize}, m_locSpace{locSpace} {}

		ManyBodySpaceBase()                                          = default;
		ManyBodySpaceBase(ManyBodySpaceBase const& other)            = default;
		ManyBodySpaceBase& operator=(ManyBodySpaceBase const& other) = default;
		ManyBodySpaceBase(ManyBodySpaceBase&& other)                 = default;
		ManyBodySpaceBase& operator=(ManyBodySpaceBase&& other)      = default;
		~ManyBodySpaceBase()                                         = default;

		__host__ __device__ Index             sysSize() const { return m_sysSize; }
		__host__ __device__ LocalSpace const& locSpace() const { return m_locSpace; }
		__host__ __device__ Index             dimLoc() const { return m_locSpace.dim(); }

		// Statically polymorphic functions
		__host__ __device__ Index locState(Index stateNum, int pos) const {
			return static_cast<Derived const*>(this)->locState_impl(stateNum, pos);
		}
		template<class Array>
		__host__ __device__ void ordinal_to_config(Array&& config, Index stateNum) const {
			static_cast<Derived const*>(this)->ordinal_to_config_impl(config, stateNum);
			return;
		}
		__host__ Eigen::RowVectorX<Index> ordinal_to_config(Index stateNum) const {
			Eigen::RowVectorX<Index> config(this->sysSize());
			this->ordinal_to_config(config, stateNum);
			return config;
		}
		template<class Array>
		__host__ __device__ Index config_to_ordinal(Array const& config) const {
			return static_cast<Derived const*>(this)->config_to_ordinal_impl(config);
		}

		/*! @name Translation operation */
		/* @{ */

	private:
		mutable Eigen::ArrayX<std::pair<Index, int>> m_transEqClass;
		mutable Eigen::ArrayX<std::pair<Index, int>> m_stateToTransEqClass;

		__host__ void   compute_transEqClass_cpu() const;
		__device__ void compute_transEqClass_gpu() const;

	public:
		__host__ __device__ void compute_transEqClass() const {
#ifndef __CUDA_ARCH__
			this->compute_transEqClass_cpu();
#else
			this->compute_transEqClass_gpu();
#endif
		};

		__host__ __device__ Index transEqDim() const { return m_transEqClass.size(); }

		__host__ __device__ Index transEqClassRep(Index eqClassNum) const {
			return m_transEqClass(eqClassNum).first;
		}
		__host__ __device__ int transPeriod(Index eqClassNum) const {
			return m_transEqClass(eqClassNum).second;
		}

		__host__ __device__ Index state_to_transEqClass(Index stateNum) const {
			static_assert([]() { return false; } & "This function is not implemented");
			return m_stateToTransEqClass(stateNum).first;
		}
		__host__ __device__ int state_to_transShift(Index stateNum) const {
			static_assert([]() { return false; } & "This function is not implemented");
			return m_stateToTransEqClass(stateNum).second;
		}
		__host__ __device__ int state_to_transPeriod(Index stateNum) const {
			static_assert([]() { return false; } & "This function is not implemented");
			auto eqClass = this->state_to_transEqClass(stateNum);
			return this->transPeriod(eqClass);
		}

		// Statically polymorphic functions
		__host__ Index translate(Index stateNum, int trans) const {
			return static_cast<Derived const*>(this)->translate_impl(stateNum, trans);
		}
		template<class Array>
		__host__ __device__ Index translate(Index stateNum, int trans, Array& work) const {
			return static_cast<Derived const*>(this)->translate_impl(stateNum, trans, work);
		}
		/* @} */

		/*! @name Parity operation */
		/* @{ */

	public:
		// Statically polymorphic functions
		__host__ Index reverse(Index stateNum) const {
			return static_cast<Derived const*>(this)->reverse_impl(stateNum);
		}
		template<class Array>
		__host__ __device__ Index reverse(Index stateNum, Array& work) const {
			return static_cast<Derived const*>(this)->reverse_impl(stateNum, work);
		}
		/* @} */
};  // class ManyBodySpaceBase

template<class Derived>
__host__ void ManyBodySpaceBase<Derived>::compute_transEqClass_cpu() const {
	if(m_transEqClass.size() >= 1) return;
	if(this->dim() <= 0) return;

	Eigen::ArrayX<bool> calculated = Eigen::ArrayX<bool>::Zero(this->dim());
	m_transEqClass.resize(this->dim());
	Eigen::ArrayXX<Index> translated(this->sysSize(), get_max_threads());
#pragma omp parallel for schedule(dynamic, 10)
	for(Index stateNum = 0; stateNum < this->dim(); ++stateNum) {
		if(calculated(stateNum)) continue;
		calculated(stateNum) = true;

		auto const threadId        = get_thread_num();
		bool       duplicationFlag = false;
		Index      trans, eqClassRep = stateNum;
		translated(0, threadId) = stateNum;
		for(trans = 1; trans != this->sysSize(); ++trans) {
			auto const transed          = this->translate(stateNum, trans);
			translated(trans, threadId) = transed;
			if(transed == stateNum) break;
			eqClassRep = (transed < eqClassRep ? transed : eqClassRep);
			if(transed == eqClassRep && calculated(transed)) {
				duplicationFlag = true;
				break;
			}
			calculated(transed) = true;
		}
		if(duplicationFlag) continue;
		auto const period = trans;
		for(trans = 0; trans != period; ++trans) {
			m_transEqClass(translated(trans, threadId)) = std::make_pair(eqClassRep, period);
		}
	}

#ifndef __APPLE__
	std::sort(std::execution::par_unseq, m_transEqClass.begin(), m_transEqClass.end(),
	          [&](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
#else
	std::sort(m_transEqClass.begin(), m_transEqClass.end(),
	          [&](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
#endif
	size_t const numEqClass
	    = std::unique(m_transEqClass.begin(), m_transEqClass.end()) - m_transEqClass.begin();
	m_transEqClass.conservativeResize(numEqClass);
}

#ifdef __NVCC__
	#include "ManyBodySpaceBase.cuh"
#endif