#pragma once

#include "ObjectOnGPU.cuh"
#include "mBodyOpSpace_Boson.hpp"

template<typename Scalar_>
__global__ void construct_mBodyOpSpace_kernel(mBodyOpSpace<ManyBodyBosonSpace, Scalar_>* ptr,
                                              Index m, Index sysSize, Index N) {
#ifndef NDEBUG
	int dev;
	cuCHECK(cudaGetDevice(&dev));
	printf("# (Device %d): %s\n", dev, __PRETTY_FUNCTION__);
#endif
	new(ptr) mBodyOpSpace<ManyBodyBosonSpace, Scalar_>(m, sysSize, N);
	DEBUG(printf("#\tdim = %d, sysSize = %d, m = %d, N = %d\n", int(ptr->dim()), int(sysSize),
	             int(m), int(N)));
}

template<typename Scalar_>
class ObjectOnGPU< mBodyOpSpace<ManyBodyBosonSpace, Scalar_> > {
	private:
		using T   = mBodyOpSpace<ManyBodyBosonSpace, Scalar_>;
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
			          << "hObj.N() = " << hObj.N() << std::endl;
			construct_mBodyOpSpace_kernel<<<1, 1>>>(m_ptr, hObj.m(), hObj.sysSize(), hObj.N());
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