#pragma once
#include "typedefs.hpp"

template<class T, class... Args>
__global__ void construct_object_kernel(T* ptr, Args... args) {
	DEBUG(printf("# %s\n", __PRETTY_FUNCTION__));
	new(ptr) T(args...);
}

template<class T, class... Args>
__global__ void destruct_object_kernel(T* ptr) {
#ifndef NDEBUG
	int dev = -1;
	cudaGetDevice(&dev);
	printf("# (Device %d) %s\n", dev, __PRETTY_FUNCTION__);
#endif
	    ptr->~T();
}

template<class T>
class ObjectOnGPU {
	private:
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

		template<class... Args>
		ObjectOnGPU(Args... args) {
			cuCHECK(cudaGetDevice(&m_dev));
			cuCHECK(cudaMalloc((void**)&m_ptr, sizeof(T)));
			construct_object_kernel<<<1, 1>>>(m_ptr, args...);
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