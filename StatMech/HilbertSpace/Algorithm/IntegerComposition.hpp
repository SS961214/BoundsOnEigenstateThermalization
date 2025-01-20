#pragma once

#include "typedefs.hpp"
#include <Eigen/Dense>
#include <iostream>

/**
 * @brief Lists a weak composition of an integer N up to length L with a
 * constraint that each summand does not exceed Max.
 *
 */
class IntegerComposition {
	private:
		Index                 m_N      = 0;
		Index                 m_Length = 0;
		Index                 m_Max    = 0;
		Index                 m_dim    = 0;
		Eigen::ArrayXX<Index> m_workA;
		Eigen::ArrayXX<Index> m_workB;

	public:
		__host__ __device__ IntegerComposition(Index N = 0, Index Length = 0, Index Max = 0);
		IntegerComposition(IntegerComposition const&)            = default;
		IntegerComposition& operator=(IntegerComposition const&) = default;
		IntegerComposition(IntegerComposition&&)                 = default;
		IntegerComposition& operator=(IntegerComposition&&)      = default;
		~IntegerComposition()                                    = default;

		__host__ __device__ Index value() const { return m_N; };
		__host__ __device__ Index length() const { return m_Length; };
		__host__ __device__ Index max() const { return m_Max; };
		__host__ __device__ Index dim() const { return m_dim; };
		__host__ __device__ Eigen::ArrayXX<Index> const& workA() const { return m_workA; };
		__host__ __device__ Index workA(Index j, Index k) const { return m_workA(j, k); };
		__host__ __device__ Eigen::ArrayXX<Index> const& workB() const { return m_workB; };
		__host__ __device__ Index workB(Index j, Index k) const { return m_workB(j, k); };

		template<class Array>
		__host__ __device__ Index config_to_ordinal(Array const& vec) const;

		template<class Array>
		__host__ __device__ void ordinal_to_config(Array&& vec, Index const ordinal) const;

		__host__ Eigen::RowVectorX<Index> ordinal_to_config(Index const ordinal) const {
			Eigen::RowVectorX<Index> res(this->length());
			this->ordinal_to_config(res, ordinal);
			return res;
		}

		__host__ __device__ Index locNumber(Index ordinal, int const pos) const;

		template<class Array>
		__host__ __device__ Index translate(Index const ordinal, int trans, Array& work) const {
			assert(ordinal < this->dim());
			assert(0 <= trans && static_cast<Index>(trans) <= this->length());
			assert(static_cast<Index>(work.size()) >= this->length());
			this->ordinal_to_config(work, ordinal);
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
			translatedConfig_wapper const translated(work, trans, this->length());
			return this->config_to_ordinal(translated);
		}

		__host__ Index translate(Index const ordinal, int trans) const {
			Eigen::ArrayX<Index> config(m_Length);
			return this->translate(ordinal, trans, config);
		}

		template<class Array>
		__host__ __device__ Index reverse(Index const ordinal, Array& work) const {
			assert(ordinal < this->dim());
			assert(static_cast<Index>(work.size()) >= this->length());
			this->ordinal_to_config(work, ordinal);
			class reversedConfig_wapper {
				private:
					Array const& m_config;
					Index const  m_length;

				public:
					__host__ __device__ reversedConfig_wapper(Array const& config, Index L)
					    : m_config{config}, m_length{L} {
						assert(m_config.size() >= m_length);
					}
					__host__ __device__ Index size() const { return m_config.size(); }
					__host__ __device__ Index operator()(int pos) const {
						return m_config(m_length - 1 - pos);
					}
			};
			reversedConfig_wapper const reversed(work, this->length());
			return this->config_to_ordinal(reversed);
		}

		__host__ Index reverse(Index const ordinal) const {
			Eigen::ArrayX<Index> config(m_Length);
			return this->reverse(ordinal, config);
		}
};

inline IntegerComposition::IntegerComposition(Index N, Index Length, Index Max)
    : m_N{N},
      m_Length{Length},
      m_Max{Max < N ? Max : N},
      m_workA(N + 1, (Length > 0 ? Length : 1)),
      m_workB(N + 1, (Length > 0 ? Length : 1)) {
	// std::cout << __PRETTY_FUNCTION__ << std::endl;
	if(m_Max * m_Length < m_N) {
		printf("Error at [%s:%d]\n\t%s", __FILE__, __LINE__, __PRETTY_FUNCTION__);
		printf("\n\tMessage:\t m_Max(%d) * m_Length(%d) = %d < m_N(%d)\n", int(m_Max),
		       int(m_Length), int(m_Max * m_Length), int(m_N));
#ifndef __CUDA_ARCH__
		std::exit(EXIT_FAILURE);
#else
		return;
#endif
	}
	if(m_Length <= 1) {
		m_dim = m_Length;
		return;
	}

	auto& Dims = m_workB;
	for(Index l = 0; l != m_Length; ++l) Dims(0, l) = 1;
	for(Index n = m_N; n != m_Max; --n) {
		Dims(n, 1)    = 0;
		Dims(n, 0)    = 0;
		m_workA(n, 0) = 0;
	}
	for(Index n = 0; n != m_Max + 1; ++n) {
		Dims(n, 1)    = 1;
		Dims(n, 0)    = 0;
		m_workA(n, 0) = 0;
	}
	// Recursively calculate Dims(l,n) for remaining (l,n)
	for(Index l = 2; l != m_Length; ++l)
		for(Index n = 1; n <= m_N; ++n) {
			Dims(n, l) = 0;
			for(Index k = 0; k <= (m_Max < n ? m_Max : n); ++k) {
				Dims(n, l) += Dims(n - k, l - 1);
			}
		}
	for(Index l = 1; l != m_Length; ++l) {
		m_workA(0, l) = Dims(m_N, m_Length - l);
		for(Index n = 1; n <= m_N; ++n)
			m_workA(n, l) = m_workA(n - 1, l) + Dims(m_N - n, m_Length - l);
	}

	for(Index l = 0; l != m_Length; ++l) m_workB(0, l) = 0;
	for(Index n = 1; n <= N; ++n) {
		for(Index l = 1; l != m_Length - 1; ++l)
			m_workB(n, l) = m_workA(n - 1, l) - m_workA(n - 1, l + 1);
		m_workB(n, m_Length - 1) = m_workA(n - 1, m_Length - 1);
	}

	m_dim = m_workA(m_Max, 1);
}

template<class Array>
__host__ __device__ inline Index IntegerComposition::config_to_ordinal(Array const& config) const {
	assert(static_cast<Index>(config.size()) >= m_Length);
	Index z = 0, res = 0;
	for(Index l = 1; l < m_Length; ++l) {
		assert(0 <= m_Length - l);
		assert(m_Length - l < config.size());
		z += config(m_Length - l);
		if(z >= m_workB.rows()) {
			printf("# z = %d, m_workB.rows() = %d, m_Length = %d\n", int(z), int(m_workB.rows()),
			       int(m_Length));
			printf("# ");
			for(auto pos = 0; pos < config.size(); ++pos) { printf("%d ", int(config(pos))); }
			printf("\n");
		}
		assert(z < m_workB.rows());
		assert(l < m_workB.cols());
		res += m_workB(z, l);
	}
	return res;
}

template<class Array>
__host__ __device__ inline void IntegerComposition::ordinal_to_config(Array&&     config,
                                                                      Index const ordinal) const {
	assert(static_cast<Index>(config.size()) >= m_Length);
	Index ordinal_copy = ordinal;
	Index z = 0, zPrev = 0;
	for(Index l = 1; l < m_Length; ++l) {
		while(m_workA(z, l) <= ordinal_copy) {
			assert(0 <= z && z < m_workA.rows());
			assert(0 <= l && l < m_workA.cols());
			z += 1;
		}
		config(m_Length - l) = z - zPrev;
		zPrev                = z;
		ordinal_copy -= m_workB(z, l);
	}
	config(0) = m_N - z;
}

__host__ __device__ inline Index IntegerComposition::locNumber(Index ordinal, int const pos) const {
	assert(0 <= pos && static_cast<Index>(pos) < this->length());
	Index z = 0, zPrev = 0;
	for(Index l = 1; l != m_Length - pos; ++l) {
		while(m_workA(z, l) <= ordinal) z += 1;
		zPrev = z;
		ordinal -= m_workB(z, l);
	}
	if(pos == 0)
		return m_N - z;
	else {
		while(m_workA(z, m_Length - pos) <= ordinal) z += 1;
		return z - zPrev;
	}
}