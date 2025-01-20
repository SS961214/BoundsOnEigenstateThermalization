#include "tests.hpp"
#include <Eigen/Core>
#include <cuda/std/complex>

namespace Eigen {
	template<typename Real_>
	struct NumTraits<cuda::std::complex<Real_> > : GenericNumTraits<cuda::std::complex<Real_> > {
			typedef Real_                              Real;
			typedef typename NumTraits<Real_>::Literal Literal;
			enum {
				IsComplex             = 1,
				RequireInitialization = NumTraits<Real_>::RequireInitialization,
				ReadCost              = 2 * NumTraits<Real_>::ReadCost,
				AddCost               = 2 * NumTraits<Real>::AddCost,
				MulCost               = 4 * NumTraits<Real>::MulCost + 2 * NumTraits<Real>::AddCost
			};

			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline Real epsilon() {
				return NumTraits<Real>::epsilon();
			}
			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline Real dummy_precision() {
				return NumTraits<Real>::dummy_precision();
			}
			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline int digits10() {
				return NumTraits<Real>::digits10();
			}
	};
}  // namespace Eigen

#include "mBodyOpSpace_Spin.hpp"
#include "../ManyBodySpaceBase_test.hpp"
#include "../OpSpaceBase_test.hpp"
#include <iostream>

using Scalar = cuda::std::complex<double>;

TEST_CASE("mBodyOpSpace_Spin_onGPU", "test") {
	constexpr int LMax = 18;
	constexpr int LMin = 6;
	constexpr int dLoc = 2;

	// test for class ManyBodySpinSpace
	{
		// Default constructor
		ManyBodySpinSpace                       mbSpace;
		mBodyOpSpace<decltype(mbSpace), Scalar> hOpSpace;
		ObjectOnGPU<decltype(hOpSpace)>         dOpSpace(hOpSpace);
		test_ManyBodySpaceBase(dOpSpace, hOpSpace);
		test_OpSpace(dOpSpace, hOpSpace);
	}
	{
		// test Constructor1
		ManyBodySpinSpace                       mbSpace(0, dLoc);
		mBodyOpSpace<decltype(mbSpace), Scalar> hOpSpace(0, mbSpace);
		ObjectOnGPU<decltype(hOpSpace)>         dOpSpace(hOpSpace);
		test_ManyBodySpaceBase(dOpSpace, hOpSpace);
		test_OpSpace(dOpSpace, hOpSpace);

		for(auto sysSize = LMin; sysSize <= LMax; ++sysSize) {
			ManyBodySpinSpace mbSpace(sysSize, dLoc);
			for(auto m = 1; m <= sysSize; ++m) {
				std::cout << "\nsysSize = " << sysSize << ", m = " << m << std::endl;
				mBodyOpSpace<decltype(mbSpace), Scalar> hOpSpace(m, mbSpace);
				ObjectOnGPU<decltype(hOpSpace)>         dOpSpace(hOpSpace);
				// if(m > 4 && hOpSpace.dim() > 100000000) {
				// 	std::cout << "hOpSpace.dim() = " << hOpSpace.dim()
				// 	          << " is so large. Skipping test..." << std::endl;
				// 	continue;
				// }
				test_ManyBodySpaceBase(dOpSpace, hOpSpace);
				test_OpSpace(dOpSpace, hOpSpace);
			}
		}
	}
}