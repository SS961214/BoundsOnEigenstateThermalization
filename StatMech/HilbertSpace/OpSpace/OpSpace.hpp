#pragma once

#include "typedefs.hpp"
#include "OpSpaceBase.hpp"
#include "HilbertSpace.hpp"
#include <Eigen/Core>

template<typename Scalar_>
class OpSpace;
template<typename Scalar_>
struct OpSpaceTraits< OpSpace<Scalar_> > {
		using BaseSpace = HilbertSpace<int>;
		using Scalar    = Scalar_;
};

template<typename Scalar_>
class OpSpace : public OpSpaceBase< OpSpace<Scalar_> > {
	private:
		using Base = OpSpaceBase< OpSpace<Scalar_> >;

	public:
		using BaseSpace  = typename Base::BaseSpace;
		using Scalar     = typename Base::Scalar;
		using RealScalar = typename Base::RealScalar;

		using OpSpaceBase< OpSpace >::OpSpaceBase;

		__host__ __device__ RealScalar opHSNormSq(Index opNum) const {
			assert(opNum < this->dim());
			if(opNum < this->baseDim())
				return this->baseDim();
			else
				return 2;
		}

	private:
		friend OpSpaceBase< OpSpace >;
		__host__ __device__ Index dim_impl() const {
			if constexpr(std::is_same_v<Scalar, RealScalar>) {
				return (this->baseDim() * (this->baseDim() + 1)) / 2;
			}
			else { return this->baseDim() * this->baseDim(); }
			return 0;
		}

		__host__ __device__ Index actionWorkSize_impl() const { return 0; };
		template<class Array>
		__host__ __device__ void action_impl(Index& resStateNum, Scalar& coeff, Index opNum,
		                                     Index stateNum, Array& work) const {
			(void)work;
			return this->action_impl(resStateNum, coeff, opNum, stateNum);
		}
		__host__ __device__ void action_impl(Index& resStateNum, Scalar& coeff, Index opNum,
		                                     Index stateNum) const;
};

template<typename Scalar_>
__host__ __device__ void OpSpace<Scalar_>::action_impl(Index& resStateNum, Scalar& coeff,
                                                       Index opNum, Index stateNum) const {
	assert(opNum < this->dim());
	assert(stateNum < this->baseDim());

	resStateNum = stateNum;
	coeff       = 0.0;
	Index Digit1, Digit2;

	if(opNum < this->baseDim()) {
		// Variant of sigma Z
		if(opNum == 0) { coeff = 1.0; }
		else if(opNum == this->baseDim() - 1 && this->baseDim() % 2 == 0) {
			coeff = (stateNum % 2 == 0 ? 1.0 : -1.0);
		}
		else {
			coeff = RealScalar(sqrt(2.0))
			        * sin(M_PI
			              * (2 * ((opNum + 1) / 2) * stateNum / RealScalar(this->baseDim())
			                 + (opNum % 2) / 2.0));
		}
	}
	else if(opNum < this->baseDim() * (this->baseDim() + 1) / 2) {
		// Variant of sigma X
		opNum -= this->baseDim();
		Digit1 = static_cast<int>(sqrt(RealScalar(2 * opNum) + 0.25) + 0.5);
		Digit2 = opNum - Digit1 * (Digit1 - 1) / 2;
		if(Digit1 == stateNum) {
			resStateNum = Digit2;
			coeff       = 1.0;
		}
		else if(Digit2 == stateNum) {
			resStateNum = Digit1;
			coeff       = 1.0;
		}
	}
	else {
		if constexpr(std::is_same_v<Scalar, RealScalar>) {
			static_assert([]() { return false; },
			              "typename Scalar is real, thus opNum must be less than "
			              "this->baseDim() * (this->baseDim() + 1) / 2.");
		}
		else {
			//! Variant of sigma Y
			opNum -= this->baseDim() * (this->baseDim() + 1) / 2;
			Digit1 = static_cast<int>(sqrt(RealScalar(2 * opNum) + 0.25) + 0.5);
			Digit2 = opNum - Digit1 * (Digit1 - 1) / 2;
			if(Digit1 == stateNum) {
				resStateNum = Digit2;
				coeff       = Scalar(0.0, -1.0);
			}
			else if(Digit2 == stateNum) {
				resStateNum = Digit1;
				coeff       = Scalar(0.0, +1.0);
			}
		}
	}
	return;
}