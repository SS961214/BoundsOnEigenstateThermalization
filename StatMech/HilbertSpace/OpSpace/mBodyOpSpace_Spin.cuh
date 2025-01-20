#pragma once

#include "ObjectOnGPU.cuh"
#include "mBodyOpSpace_Spin.hpp"

template<typename Scalar_>
__global__ void construct_mBodyOpSpace_kernel(mBodyOpSpace<ManyBodySpinSpace, Scalar_>* ptr,
                                              Index m, Index sysSize, Index spinDim) {
#ifndef NDEBUG
	int dev;
	cuCHECK(cudaGetDevice(&dev));
	printf("# (Device %d): %s\n", dev, __PRETTY_FUNCTION__);
#endif
	new(ptr) mBodyOpSpace<ManyBodySpinSpace, Scalar_>(m, sysSize, spinDim);
	DEBUG(printf("#\tdim = %d, sysSize = %d, m = %d, spinDim = %d\n", int(ptr->dim()), int(sysSize),
	             int(m), int(spinDim)));
}

template<typename Scalar_>
class ObjectOnGPU< mBodyOpSpace<ManyBodySpinSpace, Scalar_> > {
	private:
		using T   = mBodyOpSpace<ManyBodySpinSpace, Scalar_>;
		T*  m_ptr = nullptr;
		int m_dev = -1;

	public:
		ObjectOnGPU()                              = default;
		ObjectOnGPU(ObjectOnGPU const&)            = delete;
		ObjectOnGPU& operator=(ObjectOnGPU const&) = delete;
		ObjectOnGPU(ObjectOnGPU&& other) : m_ptr(other.m_ptr), m_dev(other.m_dev) {
			other.m_ptr = nullptr;
			other.m_dev = -1;
		}
		ObjectOnGPU& operator=(ObjectOnGPU&& other) {
			if(this != &other) {
				m_ptr       = other.m_ptr;
				m_dev       = other.m_dev;
				other.m_ptr = nullptr;
				other.m_dev = -1;
			}
			return *this;
		}

		ObjectOnGPU(T const& hObj) {
			cuCHECK(cudaGetDevice(&m_dev));
			cuCHECK(cudaMalloc((void**)&m_ptr, sizeof(T)));
			std::cout << "# Device " << m_dev << ":\t hObj.m() = " << hObj.m() << ", "
			          << "hObj.sysSize() = " << hObj.sysSize() << ", "
			          << "hObj.spinDim() = " << hObj.spinDim() << std::endl;
			construct_mBodyOpSpace_kernel<<<1, 1>>>(m_ptr, hObj.m(), hObj.sysSize(),
			                                        hObj.spinDim());
			cuCHECK(cudaGetLastError());
			cuCHECK(cudaDeviceSynchronize());
		}
		~ObjectOnGPU() {
			if(m_ptr != nullptr) {
				cuCHECK(cudaSetDevice(m_dev));
				destruct_object_kernel<<<1, 1>>>(m_ptr);
				cuCHECK(cudaGetLastError());
				cuCHECK(cudaDeviceSynchronize());
				cuCHECK(cudaFree(m_ptr));
			}
		}

		T*       ptr() { return m_ptr; }
		T const* ptr() const { return m_ptr; }
};