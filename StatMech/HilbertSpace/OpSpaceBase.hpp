#pragma once

#include "typedefs.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>

template<class Derived>
struct OpSpaceTraits;
// OpSpaceTraits should define the following properties:
// - BaseSpace
// - Scalar

template<class Derived>
class OpSpaceBase {
	public:
		using BaseSpace  = typename OpSpaceTraits<Derived>::BaseSpace;
		using Scalar     = typename OpSpaceTraits<Derived>::Scalar;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

	private:
		BaseSpace m_baseSpace;

	public:
		__host__ __device__ OpSpaceBase(BaseSpace const& baseSpace) : m_baseSpace{baseSpace} {};

		OpSpaceBase()                              = default;
		OpSpaceBase(OpSpaceBase const&)            = default;
		OpSpaceBase& operator=(OpSpaceBase const&) = default;
		OpSpaceBase(OpSpaceBase&&)                 = default;
		OpSpaceBase& operator=(OpSpaceBase&&)      = default;
		~OpSpaceBase()                             = default;

		__host__ __device__ bool operator==(OpSpaceBase const& other) const {
			return m_baseSpace == other.m_baseSpace;
		}

		__host__ __device__ BaseSpace const& baseSpace() const { return m_baseSpace; }
		__host__ __device__ Index            baseDim() const { return m_baseSpace.dim(); }

		__host__ std::pair<Index, Scalar> action(Index opNum, Index basisNum) const {
			Index  resStateNum;
			Scalar coeff;
			static_cast<Derived const*>(this)->action(resStateNum, coeff, opNum, basisNum);
			return std::make_pair(resStateNum, coeff);
		}

		__host__ void basisOp(Eigen::SparseMatrix<Scalar>& res, Index opNum) const {
			res.resize(this->baseDim(), this->baseDim());
			res.reserve(Eigen::VectorXi::Constant(this->baseDim(), 1));
#pragma omp parallel for
			for(Index basisNum = 0; basisNum < this->baseDim(); ++basisNum) {
				auto [resStateNum, coeff]           = this->action(opNum, basisNum);
				res.coeffRef(resStateNum, basisNum) = coeff;
			}
			res.makeCompressed();
		}

		__host__ Eigen::SparseMatrix<Scalar> basisOp(Index opNum) const {
			Eigen::SparseMatrix<Scalar> res;
			this->basisOp(res, opNum);
			return res;
		}

		// statically polymorphic functions
		__host__ __device__ Index dim() const {
			return static_cast<Derived const*>(this)->dim_impl();
		}

		__host__ __device__ Index actionWorkSize() const {
			return static_cast<Derived const*>(this)->actionWorkSize_impl();
		}
		template<class Array>
		__host__ __device__ void action(Index& resStateNum, Scalar& coeff, Index opNum,
		                                Index stateNum, Array& work) const {
			static_cast<Derived const*>(this)->action_impl(resStateNum, coeff, opNum, stateNum,
			                                               work);
		}
		__host__ __device__ void action(Index& resStateNum, Scalar& coeff, Index opNum,
		                                Index stateNum) const {
			static_cast<Derived const*>(this)->action_impl(resStateNum, coeff, opNum, stateNum);
		}
};