#pragma once

#include "OpSpace.hpp"
#include "../ManyBodyOpSpaceBase.hpp"

template<class BaseSpace_, typename Scalar_>
class mBodyOpSpace;
template<class BaseSpace_, typename Scalar_>
struct OpSpaceTraits< mBodyOpSpace<BaseSpace_, Scalar_> > {
		using BaseSpace = BaseSpace_;
		using Scalar    = Scalar_;
};
template<class BaseSpace_, typename Scalar_>
struct ManyBodySpaceTraits< mBodyOpSpace<BaseSpace_, Scalar_> > {
		using LocalSpace = OpSpace<Scalar_>;
};
