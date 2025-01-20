#pragma once

#include "typedefs.hpp"
#include <Eigen/Dense>
#include <limits>

template<typename Integer_,
         typename std::enable_if_t< std::numeric_limits<Integer_>::is_integer >* = nullptr >
class BaseConverter {
	private:
		using Integer = Integer_;
		Integer m_radix;
		Integer m_length;
		Integer m_max;

		__host__ __device__ Integer powi(Integer radix, Integer expo) const {
			Integer res = 1;
			for(Integer j = 0; j != expo; ++j) res *= radix;
			return res;
		};

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param radix
		 * @param length
		 */
		__host__ __device__ BaseConverter(Integer radix = 0, Integer length = 0)
		    : m_radix(radix), m_length(length), m_max(powi(radix, length)) {}

		BaseConverter(BaseConverter const&)            = default;
		BaseConverter& operator=(BaseConverter const&) = default;
		BaseConverter(BaseConverter&&)                 = default;
		BaseConverter& operator=(BaseConverter&&)      = default;
		~BaseConverter()                               = default;

		__host__ __device__ Integer radix() const { return this->m_radix; }
		__host__ __device__ Integer length() const { return this->m_length; }
		__host__ __device__ Integer max() const { return this->m_max; }

		__host__ __device__ Integer digit(Integer const num, Integer pos) const {
			pos = (pos % m_length + m_length) % m_length;
			return (num / powi(m_radix, pos)) % m_radix;
		}

		template<class Array>
		__host__ __device__ Integer config_to_number(Array const& config) const {
			Integer res = 0, base = 1;
			for(Integer l = 0; l != m_length; ++l, base *= m_radix) res += config[l] * base;
			return res;
		}

		template<class Array>
		__host__ __device__ void number_to_config(Array& res, Integer num) const {
			res.resize(m_length);
			for(Integer l = 0; l != m_length; ++l, num /= m_radix) res(l) = num % m_radix;
			return;
		}
		__host__ __device__ Eigen::RowVectorX<Integer> number_to_config(Integer num) const {
			Eigen::RowVectorX<Integer> res;
			this->number_to_config(res, num);
			return res;
		}

		__host__ __device__ Integer translate(Integer const num, Integer trans) const {
			trans                  = (trans % m_length + m_length) % m_length;
			Integer const compBase = powi(m_radix, m_length - 1 - trans);
			return (num / compBase) + (num % compBase) * powi(m_radix, trans);
		}
};