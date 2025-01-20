#pragma once

#include "SparseCompressed.cuh"
#include "globalWorkSpaceManagement.cuh"
#include <HilbertSpace>
#include <MatrixGPU>
#include <Eigen/Dense>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace StatMech {
	template<typename Scalar, typename RealScalar>
	__global__ void ETHMeasure2_kernel(
	    double* __restrict__ res, RealScalar const* __restrict__ dEigValPtr, RealScalar const dE,
	    Scalar const* __restrict__ dEigVecPtr, Index const LD,
	    SparseMatrix<Scalar> const* __restrict__ adBasisPtr,
	    mBodyOpSpace<ManyBodySpinSpace, Scalar> const* __restrict__ opSpacePtr,
	    Index const opTransEqDim, Index const* __restrict__ transEqClassRepPtr,
	    int const* __restrict__ transPeriodPtr) {
		mBodyOpSpace<ManyBodySpinSpace, Scalar> const& opSpace = *opSpacePtr;
		Index const                                    dimHtot = opSpace.baseSpace().dim();
		SparseMatrix<Scalar> const&                    adBasis = *adBasisPtr;

		Index const                                        sectorDim = adBasis.rows();
		Eigen::Map<Eigen::VectorX<RealScalar> const> const eigVals(dEigValPtr, sectorDim);
		Eigen::Map<Eigen::MatrixX<Scalar> const, 0, Eigen::OuterStride<>> const eigVecs(
		    dEigVecPtr, sectorDim, sectorDim, Eigen::OuterStride<>(LD));
		// Orbits of translation operation in the many-body operator space.
		Eigen::Map<Eigen::VectorX<Index> const> const transEqClassRep(transEqClassRepPtr,
		                                                              opTransEqDim);
		Eigen::Map<Eigen::VectorXi const> const       transPeriod(transPeriodPtr, opTransEqDim);

		Index const opEqClass = blockIdx.x + gridDim.x * blockIdx.y;
		// #ifndef NDEBUG
		// 		if(opEqClass == 0 && threadIdx.x == 0) {
		// 			printf("# %s\n#\t sectorDim=%d, dimHtot=%d, opTransEqDim = %d\n", __PRETTY_FUNCTION__,
		// 			       int(sectorDim), int(dimHtot), int(opTransEqDim));
		// 			printf("#\t sectorDim = %d, adBasis.rows() = %d, adBasis.cols() = %d\n", int(sectorDim),
		// 			       int(adBasis.rows()), int(adBasis.cols()));
		// 		}
		// #endif
		if(opEqClass >= opTransEqDim) return;
		Index const opOrdinal = transEqClassRep(opEqClass);
		assert(opOrdinal < opSpace.dim());

		// Step 1: Compute eigenstate expectation values
		extern __shared__ double shMem[];
		double* __restrict__ dataPtr = shMem;
		Eigen::Map<Eigen::VectorXd> expval(dataPtr, eigVals.size());
		for(Index alpha = threadIdx.x; alpha < expval.size(); alpha += blockDim.x)
			expval(alpha) = 0.0;
		__syncthreads();
		// if(threadIdx.x == 0)
		// 	for(Index beta = 0; beta < expval.size(); ++beta) assert(abs(expval(beta)) < 1.0E-12);
		// __syncthreads();
		// define: dataPtr: (size in bytes) = opSpace.sysSize() * blockDim.x * sizeof(int);
		int* __restrict__ workPtr = reinterpret_cast<int*>(dataPtr + expval.size())
		                            + opSpace.actionWorkSize() * threadIdx.x;
		Eigen::Map<Eigen::ArrayXi> work(workPtr, opSpace.actionWorkSize());

		// if(threadIdx.x == 0) printf("#\t dimHtot = %d, adBasis.cols() = %d\n", int(dimHtot), int(adBasis.cols()));
		assert(dimHtot == adBasis.cols());
		assert(sectorDim == adBasis.rows());
		assert(sectorDim == expval.size());
		assert(sectorDim == eigVecs.rows());
		assert(sectorDim == eigVecs.cols());

		for(Index inBasis = threadIdx.x; inBasis < dimHtot; inBasis += blockDim.x) {
			Index  outBasis = inBasis;
			Scalar coeff    = 1.0;
			opSpace.action(outBasis, coeff, opOrdinal, inBasis, work);

			for(Index pos2 = adBasis.outerIndexPtr()[inBasis];
			    pos2 < adBasis.outerIndexPtr()[inBasis + 1]; ++pos2) {
				assert(adBasis.outerIndexPtr()[inBasis] == inBasis);
				Index const  idx2     = adBasis.innerIndexPtr()[pos2];
				Scalar const adBElem2 = adBasis.valuePtr()[pos2];
				assert(0 <= idx2 && idx2 < sectorDim);

				for(Index pos1 = adBasis.outerIndexPtr()[outBasis];
				    pos1 < adBasis.outerIndexPtr()[outBasis + 1]; ++pos1) {
					assert(adBasis.outerIndexPtr()[outBasis] == outBasis);
					Index const  idx1     = adBasis.innerIndexPtr()[pos1];
					Scalar const adBElem1 = adBasis.valuePtr()[pos1];
					assert(0 <= idx1 && idx1 < sectorDim);

					for(Index beta = 0; beta < eigVecs.cols(); ++beta) {
						// To avoid bank conflicts in accessing "expval".
						Index const  alpha = (beta + threadIdx.x) % expval.size();
						double const elem  = real(conj(eigVecs(idx1, alpha)) * adBElem1 * coeff
						                          * conj(adBElem2) * eigVecs(idx2, alpha));
						// This instruction takes much time. (without) 3s → (with) 11s
						atomicAdd(&expval(alpha), elem);
					}
				}
			}
		}
		__syncthreads();
		// if(threadIdx.x == 0) {
		// 	double const trace = expval.sum();
		// 	// for(Index alpha = 0; alpha < expval.size(); ++alpha) {
		// 	// 	trace += expval(alpha);
		// 	// if(!(abs(expval(alpha) - 1.0) < 1.0E-6)) {
		// 	// 	printf("#\t opEqClass = %d,\t opOrdinal = %d,\t expval(%d) = %lf\n",
		// 	// 	       int(opEqClass), int(opOrdinal), int(alpha), expval(alpha));
		// 	// }
		// 	// assert(abs(expval(alpha) - 1.0) < 1.0E-6);
		// 	// }
		// 	// printf("#\t opEqClass = %d,\t opOrdinal = %d,\t trace = %lf\n", int(opEqClass),
		// 	//    int(opOrdinal), trace);
		// 	// assert(abs(trace - sectorDim) < 1.0E-6);
		// 	// if(!(abs(trace - sectorDim) < 1.0E-4)) {
		// 	// 	printf("#\t opEqClass = %d,\t opOrdinal = %d,\t trace = %lf\n", int(opEqClass),
		// 	// 	       int(opOrdinal), trace);
		// 	// }
		// 	// assert(abs(trace - sectorDim) < 1.0E-4);
		// 	// // This assertion does not always hold mathematically.
		// 	// if(!(abs(trace) < 1.0E-4)) {
		// 	// 	printf("#\t opEqClass = %d,\t opOrdinal = %d,\t trace = %lf\n", int(opEqClass),
		// 	// 	       int(opOrdinal), trace);
		// 	// }
		// 	// assert(abs(trace) < 1.0E-4);
		// }
		// __syncthreads();

		// Step 2: Compute the difference between expectation values and the microcanonical average
		auto const get_smid = []() {
			unsigned ret;
			asm("mov.u32 %0, %smid;" : "=r"(ret));
			return ret;
		};
		Index const  smid          = get_smid();
		double const factor        = transPeriod(opEqClass);
		auto const   upperQuotient = [&](Index x, Index y) { return x / y + (x % y == 0 ? 0 : 1); };
		for(Index j = 0; j != upperQuotient(expval.size(), blockDim.x); ++j) {
			Index const alpha = threadIdx.x + j * blockDim.x;
			if(alpha >= expval.size()) continue;

			Index idMin = alpha, idMax = alpha;
			for(idMin = alpha; idMin >= 0; --idMin) {
				if(eigVals(alpha) - eigVals(idMin) > dE) break;
			}
			++idMin;
			for(idMax = alpha; idMax < eigVals.size(); ++idMax) {
				if(eigVals(idMax) - eigVals(alpha) > dE) break;
			}
			--idMax;

			double mcAve = 0.0;
			for(Index beta = idMin; beta <= idMax; ++beta) mcAve += expval(beta);
			mcAve /= double(idMax - idMin + 1);
			double const difference = expval(alpha) - mcAve;
			atomicAdd(&res[alpha + smid * sectorDim], factor * difference * difference);
		}

		double const p = (opEqClass + 1) / double(opTransEqDim) * 100;
		if(abs(p - int(p / 5) * 5.0) < 100.0 / opTransEqDim && threadIdx.x == 0) {
			printf("#\t Progress: %d%\n", int(p / 5) * 5);
		}
	}

	template<class MatrixCPU, class Scalar_, class TransSector>
	__host__ Eigen::ArrayXd ETHMeasure2(
	    thrust::device_vector<double> const& dEigVals, GPU::MatrixGPU<MatrixCPU> const& dEigVecs,
	    mBodyOpSpace<ManyBodySpinSpace, Scalar_> const&                hmbOpSpace,
	    ObjectOnGPU< mBodyOpSpace<ManyBodySpinSpace, Scalar_> > const& dmbOpSpace,
	    TransSector const& hSector, double shWidth) {
		using Scalar     = typename MatrixCPU::Scalar;
		using RealScalar = typename MatrixCPU::RealScalar;
		using MBOpSpace  = mBodyOpSpace<ManyBodySpinSpace, Scalar_>;
		std::cout << __PRETTY_FUNCTION__ << std::endl;

		std::cout << "#\t TransSector.dim() = " << hSector.dim() << std::endl;
		hmbOpSpace.compute_transEqClass();
		std::cout << "#\t mbOpSpace.transEqDim() = " << hmbOpSpace.transEqDim() << std::endl;

		int blocks = std::sqrt(hmbOpSpace.transEqDim());
		if(blocks * blocks < hmbOpSpace.transEqDim()) blocks += 1;
		dim3 const nBlocks(blocks, blocks);

		// Configure the kernels
		constexpr int dev              = 0;
		int const     maxShMemPerBlock = GPU::MAGMA::prop(dev).sharedMemPerBlockOptin;
		int           nThreads, shMem;
		{
			auto const&        kernel = ETHMeasure2_kernel<Scalar, RealScalar>;
			cudaFuncAttributes attr;
			cuCHECK(cudaFuncGetAttributes(&attr, kernel));
			cuCHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
			                             maxShMemPerBlock - attr.sharedSizeBytes));
			int const shMemPerBlock  = hSector.dim() * sizeof(RealScalar);
			int const shMemPerThread = sizeof(int) * hmbOpSpace.actionWorkSize();
			configureKernel(nThreads, shMem, shMemPerBlock, shMemPerThread, kernel);
			cuCHECK(
			    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shMem));
		}

		// // Configuration of the shared memory
		// int const requestedShMem = hSector.dim() * sizeof(double)
		//                            + hmbOpSpace.actionWorkSize() * sizeof(int) * hSector.dimTot();
		// int const minEeqShMem
		//     = hSector.dim() * sizeof(double) + hmbOpSpace.actionWorkSize() * sizeof(int);
		// std::cout << "#\t requestedShMem = " << requestedShMem << "\n"
		//           << "#\t    minEeqShMem = " << minEeqShMem << std::endl;

		// std::cout << "#\t cudaDevAttrMaxSharedMemoryPerBlock = " << maxShMemPerBlock << std::endl;
		// if(minEeqShMem > maxShMemPerBlock) {
		// 	std::cerr
		// 	    << "Error: " << __FILE__ << ":" << __LINE__
		// 	    << "\tShared memory on the device is insufficient to run \"ETHMeasure2_kernel\"."
		// 	    << std::endl;
		// 	std::exit(EXIT_FAILURE);
		// }
		// if(maxShMemPerBlock > requestedShMem) {
		// 	cuCHECK(cudaFuncSetAttribute(ETHMeasure2_kernel<Scalar, RealScalar>,
		// 	                             cudaFuncAttributeMaxDynamicSharedMemorySize,
		// 	                             requestedShMem));
		// }
		// else {
		// 	cuCHECK(cudaFuncSetAttribute(ETHMeasure2_kernel<Scalar, RealScalar>,
		// 	                             cudaFuncAttributeMaxDynamicSharedMemorySize,
		// 	                             maxShMemPerBlock));
		// }
		// struct cudaFuncAttributes attr;
		// cuCHECK(cudaFuncGetAttributes(&attr, ETHMeasure2_kernel<Scalar, RealScalar>));
		// std::cout << "#\t     attr.maxDynamicSharedSizeBytes = " << attr.maxDynamicSharedSizeBytes
		//           << std::endl;
		// assert(attr.maxDynamicSharedSizeBytes
		//        >= hSector.dim() * sizeof(double) + hmbOpSpace.actionWorkSize() * sizeof(int));
		// int const nConfig = (attr.maxDynamicSharedSizeBytes - hSector.dim() * sizeof(double))
		//                     / (hmbOpSpace.actionWorkSize() * sizeof(int));
		// assert(nConfig >= 1);
		// int const  nThreads = min(nConfig, attr.maxThreadsPerBlock);
		// dim3 const threads(nThreads, 1);
		// int const  shMem
		//     = hSector.dim() * sizeof(double) + hmbOpSpace.actionWorkSize() * sizeof(int) * nThreads;
		// assert(shMem <= attr.maxDynamicSharedSizeBytes);
		// cuCHECK(cudaFuncSetAttribute(ETHMeasure2_kernel<Scalar, RealScalar>,
		//                              cudaFuncAttributeMaxDynamicSharedMemorySize, shMem));

		// std::cout << "#\t  nBlocks = " << nBlocks << std::endl;
		// std::cout << "#\t nThreads = " << nThreads << std::endl;
		// std::cout << "#\t    shMem = " << shMem << std::endl;
		// std::cout << "#\t  nConfig = " << nConfig << std::endl;

		// 		// Prepare adjoint basis on GPU
		assert(hSector.basis().isCompressed());
		Eigen::SparseMatrix<Scalar, 0, Eigen::Index> adBasis = hSector.basis().adjoint();
		adBasis.makeCompressed();
		assert(
		    abs(Eigen::MatrixX<Scalar>(adBasis * adBasis.adjoint()).trace() - double(hSector.dim()))
		    < 1.0E-4);
		ObjectOnGPU<SparseMatrix<Scalar>> dAdBasis(adBasis);

		// Orbits of translation opration on the many-body operator space.
		thrust::host_vector<Index> opTransEqClassRep(hmbOpSpace.transEqDim());
		thrust::host_vector<int>   opTransPeriod(hmbOpSpace.transEqDim());
#pragma omp parallel for
		for(Index j = 0; j < hmbOpSpace.transEqDim(); ++j) {
			opTransEqClassRep[j] = hmbOpSpace.transEqClassRep(j);
			opTransPeriod[j]     = hmbOpSpace.transPeriod(j);
			assert(opTransEqClassRep[j] < hmbOpSpace.dim());
			assert(0 < opTransPeriod[j] && opTransPeriod[j] <= hmbOpSpace.sysSize());
		}
		thrust::device_vector<Index> dopTransEqClassRep(opTransEqClassRep);
		thrust::device_vector<int>   dopTransPeriod(opTransPeriod);

		int const mpNum = GPU::MAGMA::prop(0).multiProcessorCount;
		std::cout << "#\t devProp.multiProcessorCount = " << mpNum << std::endl;
		thrust::device_vector<double> dres(hSector.dim() * mpNum, 0);
		ETHMeasure2_kernel<<<nBlocks, nThreads, shMem>>>(
		    dres.data().get(), dEigVals.data().get(), shWidth, dEigVecs.data(), dEigVecs.LD(),
		    dAdBasis.ptr(), dmbOpSpace.ptr(), hmbOpSpace.transEqDim(),
		    dopTransEqClassRep.data().get(), dopTransPeriod.data().get());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
		thrust::host_vector<double> hres(dres);
		// std::cout << Eigen::Map<Eigen::ArrayXXd>(hres.data(), hSector.dim(), mpNum) << std::endl;
		Eigen::ArrayXd res
		    = Eigen::Map<Eigen::ArrayXXd>(hres.data(), hSector.dim(), mpNum).rowwise().sum();
		// Eigen::ArrayXd res = Eigen::ArrayXd::Zero(hSector.dim());
		return res;
	}
}  // namespace StatMech
