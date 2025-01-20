#pragma once

#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"
#include "OpSpaceBase.hpp"

template<class Derived>
class ManyBodyOpSpaceBase : public OpSpaceBase<Derived>,
                            public ManyBodySpaceBase<Derived> {
	public:
		using BaseSpace  = typename OpSpaceBase<Derived>::BaseSpace;
		using Scalar     = typename OpSpaceBase<Derived>::Scalar;
		using RealScalar = typename OpSpaceBase<Derived>::RealScalar;
		using LocalSpace = typename ManyBodySpaceBase<Derived>::LocalSpace;

	public:
		using ManyBodySpaceBase<Derived>::dim;

		/**
		 * @brief Custom constructor 1
		 *
		 * @param baseSpace
		 * @param sysSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodyOpSpaceBase(BaseSpace const& baseSpace, Index sysSize,
		                                        LocalSpace const& locSpace)
		    : OpSpaceBase<Derived>(baseSpace), ManyBodySpaceBase<Derived>(sysSize, locSpace) {}
		__host__ __device__ ManyBodyOpSpaceBase(BaseSpace&& baseSpace, Index sysSize,
		                                        LocalSpace&& locSpace)
		    : OpSpaceBase<Derived>(std::move(baseSpace)),
		      ManyBodySpaceBase<Derived>(sysSize, std::move(locSpace)) {}

		ManyBodyOpSpaceBase()                                      = default;
		ManyBodyOpSpaceBase(ManyBodyOpSpaceBase const&)            = default;
		ManyBodyOpSpaceBase& operator=(ManyBodyOpSpaceBase const&) = default;
		ManyBodyOpSpaceBase(ManyBodyOpSpaceBase&&)                 = default;
		ManyBodyOpSpaceBase& operator=(ManyBodyOpSpaceBase&&)      = default;
		~ManyBodyOpSpaceBase()                                     = default;
};