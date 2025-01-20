#pragma once

#include "typedefs.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>

// #include <bitset>

// class BitArray {
// 	private:
// 		uint64_t b;

// 	public:
// 		BitArray(uint64_t value = 0) : b(value) {}

// 		// Overload the operator() to access and modify bits
// 		class BitReference {
// 			private:
// 				uint64_t& b;
// 				int       index;

// 			public:
// 				BitReference(uint64_t& b, int index) : b(b), index(index) {}

// 				// Assignment operator to set the bit
// 				BitReference& operator=(bool value) {
// 					if(value) { b |= (1ULL << index); }
// 					else { b &= ~(1ULL << index); }
// 					return *this;
// 				}

// 				// Conversion operator to get the bit value
// 				operator bool() const { return (b >> index) & 1ULL; }
// 		};

// 		// Overload operator() to return a BitReference
// 		BitReference operator()(int index) { return BitReference(b, index); }

// 		// Const version of operator() to get the bit value
// 		bool operator()(int index) const { return (b >> index) & 1ULL; }

// 		// Print the bit array
// 		void printBits() const {
// 			std::bitset<64> bits(b);
// 			std::cout << bits << std::endl;
// 		}

// 		// Get the underlying uint64_t value
// 		uint64_t getValue() const { return b; }
// };

class Combination {
	private:
		int                   m_L = 0;
		int                   m_N = 0;
		Eigen::ArrayXX<Index> m_work;

	public:
		using BIT = uint64_t;

		// declare all default constructors and assignment operators
		// since this class is used in CUDA kernels
		Combination(Combination const&)            = default;
		Combination& operator=(Combination const&) = default;
		Combination(Combination&&)                 = default;
		Combination& operator=(Combination&&)      = default;
		~Combination()                             = default;

		__host__ __device__ Combination(int L, int N) : m_L(L), m_N(N) {
			cuASSERT(L <= 64, "Error: L > 64 is not supported");
			m_work = Eigen::ArrayXX<Index>::Zero(L + 1, N + 1);
			if(L == 0) return;
			for(auto j = 0; j < m_work.rows(); ++j) {
				m_work(j, 0) = 1;
				for(auto m = 1; m <= std::min(j, m_N); ++m)
					m_work(j, m) = m_work(j - 1, m - 1) + m_work(j - 1, m);
			}
		}

		__host__ __device__ Index dim() const { return m_work(m_L, m_N); }
		__host__ __device__ int   L() const { return m_L; }
		__host__ __device__ int   N() const { return m_N; }

		__host__ __device__ Index config_to_ordinal(BIT config) const {
			Index res        = 0;
			int   ones_count = 0;
			for(int i = m_L - 1; i >= 0; --i) {
				if((config >> i) & 1) {
					assert(0 <= ones_count && ones_count <= m_N);
					res += m_work(i, m_N - ones_count);
					++ones_count;
				}
			}
			return res;
		}

		__host__ __device__ BIT ordinal_to_config(Index ordinal) const {
			BIT config     = 0;
			int ones_count = 0;
			for(int i = m_L - 1; i >= 0; --i) {
				if(ordinal >= m_work(i, m_N - ones_count)) {
					ordinal -= m_work(i, m_N - ones_count);
					config |= (1 << i);
					++ones_count;
				}
			}
			return config;
		}

		__host__ __device__ Index locNumber(Index ordinal, int pos) const {
			BIT const config = ordinal_to_config(ordinal);
			return (config >> pos) & 1;
		}

		// Implement a function that shifts a given configuration by a given amount
		__host__ __device__ Index translate(Index ordinal, int shift) const {
			BIT const config = ordinal_to_config(ordinal);
			return config_to_ordinal(translateBIT(config, shift));
		}

		__host__ __device__ BIT translateBIT(BIT config, int shift) const {
			assert(0 <= shift && shift < m_L);
			BIT const mask = (1ULL << m_L) - 1;
			return ((config << shift) | (config >> (m_L - shift))) & mask;
		}

		// Implement a function that reverses a given configuration
		__host__ __device__ Index reverse(Index ordinal) const {
			BIT const config = ordinal_to_config(ordinal);
			BIT const reversed_config
			    = reverseBits(config) >> (64 - m_L);  // assumes BIT = uint64_t
			return config_to_ordinal(reversed_config);
		}
		// This implementation assumes BIT = uint64_t
		__host__ __device__ uint64_t reverseBits(uint64_t config) const {
			config
			    = ((config & 0x5555555555555555ULL) << 1) | ((config & 0xAAAAAAAAAAAAAAAAULL) >> 1);
			config
			    = ((config & 0x3333333333333333ULL) << 2) | ((config & 0xCCCCCCCCCCCCCCCCULL) >> 2);
			config
			    = ((config & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((config & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
			config
			    = ((config & 0x00FF00FF00FF00FFULL) << 8) | ((config & 0xFF00FF00FF00FF00ULL) >> 8);
			config = ((config & 0x0000FFFF0000FFFFULL) << 16)
			         | ((config & 0xFFFF0000FFFF0000ULL) >> 16);
			config = ((config & 0x00000000FFFFFFFFULL) << 32)
			         | ((config & 0xFFFFFFFF00000000ULL) >> 32);
			return config;
		}
};