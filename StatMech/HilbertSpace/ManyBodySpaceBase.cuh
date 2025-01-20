#pragma once

#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"
#include <cub/device/device_radix_sort.cuh>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

// Reference implementation that does not use dynamic parallelism and the dynamic allocation of shared memory
template<class Derived>
__device__ void ManyBodySpaceBase<Derived>::compute_transEqClass_gpu() const {
// 	if(m_transEqClass.size() >= 1) return;
// 	if(this->dim() <= 0) return;

// 	printf("# %s\n", __PRETTY_FUNCTION__);

// 	{
// 		Eigen::ArrayXi appeared = Eigen::ArrayXi::Zero(this->dim());
// 		Index transEqDim = 0;
// 		for(Index j = 0;j < this->dim(); ++j) {
// 			if(appeared(j) != 0) continue;
// 			appeared(j) = 1;
// 			for(auto trans = 1;trans < this->sysSize(); ++trans) {
// 				auto const translated = this->translate(j, trans);
// 				if(translated == j) break;
// 				appeared(translated) = 1;
// 			}
// 			transEqDim += 1;
// 		}
// 		m_transEqClass.resize(transEqDim);
// 		m_stateToTransEqClass.resize(this->dim());
// 	}
// 	{
// 		Eigen::ArrayXi appeared = Eigen::ArrayXi::Zero(this->dim());
// 		Index eqClass = 0;
// 		for(Index j = 0;j < this->dim(); ++j) {
// 			if(appeared(j) != 0) continue;
// 			appeared(j) = 1;
// 			assert(eqClass < m_transEqClass.size());
// 			m_transEqClass(eqClass).first = j;
// 			m_stateToTransEqClass(j).first = eqClass;
// 			m_stateToTransEqClass(j).second = 0;
// 			Index trans = 1;
// 			for(trans = 1;trans < this->sysSize(); ++trans) {
// 				auto const translated = this->translate(j, trans);
// 				if(translated == j) break;
// 				m_stateToTransEqClass(translated).first = eqClass;
// 				m_stateToTransEqClass(translated).second = trans;
// 				appeared(translated) = 1;
// 			}
// 			assert(0 < trans && trans <= this->sysSize());
// 			m_transEqClass(eqClass).second = trans;
// 			eqClass += 1;
// 		}
// 	}

// 	printf("# (END) %s\n", __PRETTY_FUNCTION__);
}