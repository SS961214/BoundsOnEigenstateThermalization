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

#include "mBodyOpSpace_Boson.hpp"
#include "../ManyBodySpaceBase_test.hpp"
#include "../OpSpaceBase_test.hpp"
#include <iostream>

using Scalar = cuda::std::complex<double>;

TEST_CASE("mBodyOpSpace_Boson_onGPU", "test") {
	size_t pValue;
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;
	pValue *= 16;
	cuCHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, pValue));
	cuCHECK(cudaDeviceGetLimit(&pValue, cudaLimitMallocHeapSize));
	std::cout << "cudaLimitMallocHeapSize = " << pValue << std::endl;

	constexpr int LMax = 18;
	constexpr int NMax = 9;

	// test for class ManyBodyBosonSpace
	{
		// Default constructor
		ManyBodyBosonSpace                      mbSpace;
		mBodyOpSpace<decltype(mbSpace), Scalar> hOpSpace;
		ObjectOnGPU<decltype(hOpSpace)>         dOpSpace(hOpSpace);
		test_ManyBodySpaceBase(dOpSpace, hOpSpace);
		test_OpSpace(dOpSpace, hOpSpace);
	}
	{
		// // test Constructor1
		// ManyBodyBosonSpace                      mbSpace(0, dLoc);
		// mBodyOpSpace<decltype(mbSpace), Scalar> hOpSpace(0, mbSpace);
		// ObjectOnGPU<decltype(hOpSpace)>         dOpSpace(hOpSpace);
		// test_ManyBodySpaceBase(dOpSpace, hOpSpace);
		// test_OpSpace(dOpSpace, hOpSpace);

		for(auto L = 1; L <= LMax; ++L)
			for(auto N = 1; N <= NMax; ++N) {
				ManyBodyBosonSpace mbSpace(L, N);
				for(auto m = 1; m <= N; ++m) {
					std::cout << "# L = " << L << ", N = " << N << ", m = " << m << std::endl;
					mBodyOpSpace<decltype(mbSpace), Scalar> hOpSpace(m, L, N);
					ObjectOnGPU<decltype(hOpSpace)>         dOpSpace(hOpSpace);
					test_ManyBodySpaceBase(dOpSpace, hOpSpace);
					test_OpSpace(dOpSpace, hOpSpace);
				}
			}
	}
}