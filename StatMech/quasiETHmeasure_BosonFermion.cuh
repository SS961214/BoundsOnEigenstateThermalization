#pragma once

#include "SparseCompressed.cuh"
#include "quasiETHmeasure_BosonFermion.hpp"
#include "globalWorkSpaceManagement.cuh"
#include "CholeskyFactorization.cuh"
#include "calculateGramMat.cuh"
#include "calculateEEV.cuh"
#include "distributeBlocks.hpp"
#include <HilbertSpace>
#include <MatrixGPU>
#include <Eigen/Dense>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/**
 * TODO: 1. Implement the function ETHmeasure2Sq for a single GPU
 * TODO: (Completed) 1.1 Write a skelton of the function ETHmeasure2Sq
 * TODO: (Completed 2024-07-23) 1.2 Implement and test the function calculateGramMat_kernel
 * TODO: (Completed 2024-07-24) 1.3 Implement and test the function calculateEEV_kernel
 * TODO: (Completed 2024-07-24) 1.4 Implement and test the function calculatePartialNormSq_kernel
 * TODO: (Completed 2024-07-25) 2 Implement quasiETHmeasure_Boson_onGPU for a single GPU
 * TODO: (Completed 2024-09-25) 2. Optimize the function ETHmeasure2Sq for a single GPU
 * 		- Implementing the unified_kernel strategy for the function calculatePartialNormSqForBlock_kernel
 *		- Tested a device function "calculateGramMat"
 *		- Tested a device function "calculateEEV"
 *		- Tested a device function "calculatePartialNormSq"
 *		- Tested a device function "calculatePartialNormSqForBlock"
 *		- Completed the tests
 * TODO: (Completed 2024-09-26) 3. Implement the function ETHmeasure2Sq for multiple GPUs
 * TODO: 4. Optimize the function ETHmeasure2Sq for multiple GPUs
 */

#ifdef FLOAT
using RealScalar               = float;
RealScalar constexpr precision = 1.0e-6;
#else
using RealScalar               = double;
RealScalar constexpr precision = 1.0e-12;
#endif

namespace StatMech {
	template<class StateSpace_, typename Scalar_, typename RealScalar_>
	__global__ void calculatePartialNormForBlockSq_kernel(
	    RealScalar_* __restrict__ res, unsigned long long* sm_slots,
	    Scalar_* const __restrict__* gWorks, RealScalar_ const* __restrict__ dEigValPtr,
	    RealScalar_ const dE, Scalar_ const* __restrict__ dEigVecPtr, Index const LD,
	    SparseMatrix<Scalar_> const* __restrict__ adBasisPtr,
	    mBodyOpSpace<StateSpace_, Scalar_> const* __restrict__ opSpacePtr,
	    Index const* __restrict__ transEqClassRepPtr, int const* __restrict__ offsetPtr,
	    int const* __restrict__ transEqPeriodPtr, int const numBs) {
		using Scalar     = Scalar_;
		using RealScalar = RealScalar_;
		static_assert(std::is_same_v<RealScalar, typename Eigen::NumTraits<Scalar>::Real>);
		static_assert(std::is_same_v<std::decay_t<StateSpace_>, ManyBodyBosonSpace>
		              || std::is_same_v<std::decay_t<StateSpace_>, ManyBodyFermionSpace>);

		if(blockIdx.x >= numBs) return;
		SparseMatrix<Scalar> const& adBasis   = *adBasisPtr;
		int const                   sectorDim = adBasis.rows();
		auto const&                 opSpace   = *opSpacePtr;
		Index const* __restrict__ eqClassRep  = transEqClassRepPtr + offsetPtr[blockIdx.x];
		int const         blockSize           = offsetPtr[blockIdx.x + 1] - offsetPtr[blockIdx.x];
		extern __shared__ __align__(sizeof(Scalar)) int sm_data[];
		// unsigned                                        shMemSize;
		// asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(shMemSize));

		long long int start, stop;

		// Get a work slot for this block
		int const                       my_sm   = get_smid();
		int                             my_slot = -1;
		Scalar* __restrict__ __shared__ workPtr;
		if(threadIdx.x == 0) {
			// DEBUG(printf("# %s: block = %d/%d,\t blockSize = %d\n", __func__, blockIdx.x, gridDim.x,
			//              blockSize));
			my_slot = get_slot(sm_slots + my_sm);
			workPtr = gWorks[my_sm * slots_per_sm + my_slot];
			assert(my_sm >= 0);
			assert(0 <= my_slot && my_slot < slots_per_sm);
			asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));
		}
		__syncthreads();
		assert(workPtr != nullptr);

		// Calculate and factorize the Gram matrix
		RealScalar* __restrict__ const gramMatPtr = reinterpret_cast<RealScalar*>(workPtr);
		Eigen::Map<Eigen::MatrixX<RealScalar>> gramMat(gramMatPtr, blockSize, blockSize);
		calculateGramMat(gramMat, opSpace, blockSize, eqClassRep, sm_data);

		Scalar* __restrict__ const eevPtr
		    = reinterpret_cast<Scalar*>(gramMat.data() + gramMat.size() + (gramMat.size() & 1));
		// The last "(gramMat.size() & 1)" is to ensure the alignment of the pointer.
		Eigen::Map<Eigen::MatrixX<Scalar>> eev(eevPtr, sectorDim, blockSize);
		Scalar* __restrict__ expvalPtr = eev.data() + eev.size();
		calculateEEV(eev, dEigValPtr, dE, dEigVecPtr, LD, adBasis, opSpace, blockSize, eqClassRep,
		             expvalPtr, sm_data);
		__syncthreads();

		if(blockSize > 1) {
			Scalar* __restrict__ const invEEVPtr
			    = reinterpret_cast<Scalar*>(eev.data() + eev.size());
			Eigen::Map<Eigen::MatrixX<Scalar>> invEEV(invEEVPtr, sectorDim, blockSize);
			for(int alpha = threadIdx.x; alpha < sectorDim; alpha += blockDim.x) {
				for(int opIdx = 0; opIdx < blockSize; ++opIdx) {
					invEEV(alpha, opIdx) = eev(alpha, opIdx);
				}
			}
			__syncthreads();

			utuFactorization_basic(gramMat);
			forwardSubstitution(gramMat.transpose(), invEEV.transpose());
			backSubstitution(gramMat, invEEV.transpose());

			for(int alpha = threadIdx.x; alpha < sectorDim; alpha += blockDim.x) {
				double sum = 0.0;
				for(int opIdx = 0; opIdx < blockSize; ++opIdx) {
					sum += real(conj(eev(alpha, opIdx)) * invEEV(alpha, opIdx));
				}
				sum *= transEqPeriodPtr[blockIdx.x];
				atomicAdd(&res[alpha + sectorDim * my_sm], sum);
			}
			__syncthreads();
		}
		else {
			for(int alpha = threadIdx.x; alpha < sectorDim; alpha += blockDim.x) {
				double sum = 0.0;
				for(int opIdx = 0; opIdx < blockSize; ++opIdx) {
					sum += real(conj(eev(alpha, opIdx)) * eev(alpha, opIdx)) / gramMat(0, 0);
				}
				sum *= transEqPeriodPtr[blockIdx.x];
				atomicAdd(&res[alpha + sectorDim * my_sm], sum);
			}
			__syncthreads();
		}

		if(threadIdx.x == 0) release_slot(sm_slots + my_sm, my_slot);

#ifndef NDEBUG
		if(threadIdx.x == 0) {
			asm volatile("mov.u64  %0, %globaltimer;" : "=l"(stop));
			int dev;
			cudaGetDevice(&dev);
			printf("# (Device %d) %s: Block = %d/%d,\t blockSize = %d,\t elapsed = %lf (sec)\n",
			       dev, __func__, blockIdx.x, gridDim.x, blockSize, double(stop - start) / 1.0e9);
		}
#else
		double const p = (blockIdx.x + 1) / double(gridDim.x) * 100;
		if(threadIdx.x == 0 && abs(p - int(p / 5) * 5.0) < 100.0 / gridDim.x) {
			asm volatile("mov.u64  %0, %globaltimer;" : "=l"(stop));
			int dev;
			cudaGetDevice(&dev);
			printf("# (Device %d) %s: Progress: %d%%,\t Block = %d/%d,\t blockSize = %d,\t elapsed "
			       "= %lf (sec)\n",
			       int(dev), __func__, int(p / 5) * 5, int(blockIdx.x), int(gridDim.x),
			       int(blockSize), double(stop - start) / 1.0e9);
		}
#endif
	}

	template<class Solver_, class OpSpace_, class SubSpace_,
	         std::enable_if_t<IsParticleOpSpace<OpSpace_>::value>* = nullptr >
	__host__ Eigen::ArrayXd ETHmeasure2Sq(Solver_ const& dsolver, OpSpace_ const& hmbOpSpace,
	                                      SubSpace_ const& hSector, double const shWidth) {
		using Scalar     = std::decay_t<decltype(dsolver.eigenvectors()(0, 0))>;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		using StateSpace = typename OpSpace_::BaseSpace;
		static_assert(
		    std::is_same_v< RealScalar, std::decay_t<decltype(dsolver.eigenvalues()(0))> >);
		static_assert(std::is_same_v<StateSpace, ManyBodyBosonSpace>
		              || std::is_same_v<StateSpace, ManyBodyFermionSpace>);
		static_assert(std::is_same_v< SubSpace_, TransSector<StateSpace, Scalar> >
		              || std::is_same_v< SubSpace_, TransParitySector<StateSpace, Scalar> >);

		std::cout << "# " << __PRETTY_FUNCTION__ << std::endl;
		assert(dsolver.eigenvectors().rows() == dsolver.eigenvalues().size());
		assert(dsolver.eigenvectors().rows() == hSector.dim());
		assert(hmbOpSpace.baseDim() == hSector.dimTot());

		Eigen::ArrayXd resCPU = Eigen::ArrayXd::Zero(hSector.dim());
		Eigen::ArrayXd resGPU = Eigen::ArrayXd::Zero(hSector.dim());

		auto const  adBasis = hSector.basis().adjoint().eval();
		auto const& eigVecs = dsolver.eigenvectors();

		//----- 1. Calculate the workloads for each block -----//
		auto blocks = blocksInGramMat(hmbOpSpace);
		static_assert(std::is_same_v<Index, std::decay_t<decltype(blocks[0][0][0])>> == true);
		auto const [workloads, blockSizes] = distributeBlocks(blocks, GPU::MAGMA::ngpus());

		Index const costTot         = blockSizes.col(2).sum();
		Index       cost            = 0;
		Index       minBlockSizeCPU = blockSizes(0, 0);
		for(auto j = 0; j < blockSizes.rows() - 1; ++j) {
			cost += blockSizes(j, 2);
			if(blockSizes(j, 0) >= 1000) continue;
			minBlockSizeCPU = blockSizes(j, 0);
			if(cost > costTot * 0.3) break;
		}
		// minBlockSizeCPU = 2;
		std::cout << "#\t minBlockSizeCPU = " << minBlockSizeCPU << std::endl;
		//----- (END) 1. Calculate the workloads for each block -----//

		// Prepare the data on GPU
		std::vector< thrust::device_vector<RealScalar> > dEigVals(GPU::MAGMA::ngpus());
		std::vector< GPU::MatrixGPU<std::decay_t<decltype(dsolver.eigenvectors())>> > dEigVecs(
		    GPU::MAGMA::ngpus());
		std::vector< ObjectOnGPU<std::decay_t<decltype(hmbOpSpace)>> > dmbOpSpace(
		    GPU::MAGMA::ngpus());
		std::vector< ObjectOnGPU<SparseMatrix<Scalar>> > dAdBasis(GPU::MAGMA::ngpus());
		std::vector< cudaStream_t >                      priorityStreams(GPU::MAGMA::ngpus());
		std::vector<int> nThreads1(GPU::MAGMA::ngpus()), nThreads2(GPU::MAGMA::ngpus()),
		    nThreads3(GPU::MAGMA::ngpus());
		std::vector<int> shMem1(GPU::MAGMA::ngpus()), shMem2(GPU::MAGMA::ngpus()),
		    shMem3(GPU::MAGMA::ngpus());
		std::vector< thrust::device_vector<unsigned long long> > d_sm_slots(GPU::MAGMA::ngpus());
		std::vector< thrust::device_vector<Scalar> >             d_data(GPU::MAGMA::ngpus());
		std::vector< thrust::device_vector<Scalar*> >            d_gWorks(GPU::MAGMA::ngpus());

#pragma omp parallel for ordered num_threads(GPU::MAGMA::ngpus())
		for(int dev = 0; dev < GPU::MAGMA::ngpus(); ++dev) {
			cuCHECK(cudaSetDevice(dev));
			int const numSMs = GPU::MAGMA::prop(dev).multiProcessorCount;

			dEigVals[dev] = thrust::device_vector<RealScalar>(dsolver.eigenvalues().begin(),
			                                                  dsolver.eigenvalues().end());
			dEigVecs[dev] = GPU::MatrixGPU<std::decay_t<decltype(dsolver.eigenvectors())>>(
			    dsolver.eigenvectors());
			dmbOpSpace[dev] = ObjectOnGPU<std::decay_t<decltype(hmbOpSpace)>>(hmbOpSpace);
			dAdBasis[dev]   = ObjectOnGPU<SparseMatrix<Scalar>>(adBasis);
			int leastPriority, greatestPriority;
			cuCHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
			cuCHECK(cudaStreamCreateWithPriority(&priorityStreams[dev], cudaStreamNonBlocking,
			                                     greatestPriority));

			// Configure the kernels
			int const maxShMemPerBlock = GPU::MAGMA::prop(dev).sharedMemPerBlockOptin;
			{
				auto const&        kernel = calculateGramMat_kernel<StateSpace, Scalar, RealScalar>;
				cudaFuncAttributes attr;
				cuCHECK(cudaFuncGetAttributes(&attr, kernel));
				cuCHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
				                             maxShMemPerBlock - attr.sharedSizeBytes));
				int const shMemPerBlock = 0;
				int const shMemPerThread
				    = sizeof(RealScalar) + sizeof(int) * hmbOpSpace.actionWorkSize();
				configureKernel(nThreads1[dev], shMem1[dev], shMemPerBlock, shMemPerThread, kernel);
				cuCHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
				                             shMem1[dev]));
			}
			{
				auto const&        kernel = calculateEEV_kernel<StateSpace, Scalar, RealScalar>;
				cudaFuncAttributes attr;
				cuCHECK(cudaFuncGetAttributes(&attr, kernel));
				cuCHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
				                             maxShMemPerBlock - attr.sharedSizeBytes));
				int const shMemPerBlock  = 0;  //hSector.dim() * sizeof(Scalar);
				int const shMemPerThread = hmbOpSpace.actionWorkSize() * sizeof(int);
				configureKernel(nThreads2[dev], shMem2[dev], shMemPerBlock, shMemPerThread, kernel);
				cuCHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
				                             shMem2[dev]));
			}
			{
				auto const& kernel
				    = calculatePartialNormForBlockSq_kernel<StateSpace, Scalar, RealScalar>;
				cudaFuncAttributes attr;
				cuCHECK(cudaFuncGetAttributes(&attr, kernel));
				cuCHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
				                             maxShMemPerBlock - attr.sharedSizeBytes));
				int const shMemPerBlock = 0;
				int const shMemPerThread
				    = sizeof(RealScalar) + sizeof(int) * hmbOpSpace.actionWorkSize();
				configureKernel(nThreads3[dev], shMem3[dev], shMemPerBlock, shMemPerThread, kernel);
				cuCHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
				                             shMem3[dev]));
			}  // (END) Configure the kernel

			// Prepare the work storages on the global memory
			int minBGPU = workloads[dev].period.size();
			for(uint b = 0; b < workloads[dev].period.size(); ++b) {
				Index const blockSize = workloads[dev].offset[b + 1] - workloads[dev].offset[b];
				if(blockSize < minBlockSizeCPU) {
					minBGPU = b;
					break;
				}
			}
			int const numBGPU      = workloads[dev].period.size() - minBGPU;
			int const maxBlockSize = (numBGPU == 0 ? 1
			                                       : workloads[dev].offset[minBGPU + 1]
			                                             - workloads[dev].offset[minBGPU]);
			int const workSize_per_block
			    = maxBlockSize * maxBlockSize + 2 * hSector.dim() * maxBlockSize;
#pragma omp critical
			std::cout << "# (Device " << dev << ")\t minBGPU = " << minBGPU
			          << ", maxBlockSize = " << maxBlockSize
			          << ", workSize_per_block = " << workSize_per_block << ", allocated = "
			          << numSMs * slots_per_sm * workSize_per_block * sizeof(Scalar) / 1024
			          << " KiB" << std::endl;
			d_sm_slots[dev] = thrust::device_vector<unsigned long long>(numSMs, 0);
			d_data[dev]
			    = thrust::device_vector<Scalar>(numSMs * slots_per_sm * workSize_per_block, 0);
			thrust::host_vector<Scalar*> gWorks(numSMs * slots_per_sm);
			// #pragma omp parallel for
			for(uint i = 0; i < gWorks.size(); ++i) {
				gWorks[i] = d_data[dev].data().get() + i * workSize_per_block;
			}
			d_gWorks[dev] = gWorks;
			// (END) Prepare the work storages on the global memory
		}
		// (END) Prepare the data on GPU

		// omp_set_max_active_levels(2);
		std::cout << "# omp_get_max_active_levels() = " << omp_get_max_active_levels() << std::endl;
		double const startT = omp_get_wtime();
// #pragma omp parallel
// #pragma omp for reduction(task, + : resCPU, resGPU)
#pragma omp parallel reduction(task, + : resCPU, resGPU)
#pragma omp single nowait
#pragma omp taskloop grainsize(1) in_reduction(+ : resCPU, resGPU) num_tasks(GPU::MAGMA::ngpus())
		for(auto dev = 0; dev < GPU::MAGMA::ngpus(); ++dev) {
			// Calculate the assignments of each GPU
			// 0 <= b < minBGPU: executed on CPU + GPU
			// minBGPU <= b < workloads[dev].period.size(): executed on GPU
			int minBGPU = workloads[dev].period.size();
			for(uint b = 0; b < workloads[dev].period.size(); ++b) {
				Index const blockSize = workloads[dev].offset[b + 1] - workloads[dev].offset[b];
				if(blockSize < minBlockSizeCPU) {
					minBGPU = b;
					break;
				}
			}
			// minBGPU           = workloads[dev].period.size();
			int const numBGPU = workloads[dev].period.size() - minBGPU;
#pragma omp critical
			std::cout << "# (Device " << dev << ")\t minBGPU = " << minBGPU
			          << ", numBGPU = " << numBGPU << std::endl;

#pragma omp task in_reduction(+ : resGPU)
			if(numBGPU > 0) {
				cuCHECK(cudaSetDevice(dev));  // Set CUDA context on a thread
				cudaStream_t const stream = cudaStreamPerThread;
				int const          numSMs = GPU::MAGMA::prop(dev).multiProcessorCount;

				//----- 3. Calculate the partial norm within each block -----//
				//----- (Verified 2024/09/25 with "calculatePartialNormForBlockSq_kernel_test.cu") -----//
				thrust::device_vector<RealScalar> dPartialNorm1(hSector.dim() * numSMs, 0);
				// Prepare workloads on the GPU
				thrust::host_vector<int> h_offsets(workloads[dev].offset.begin() + minBGPU,
				                                   workloads[dev].offset.end());
#pragma omp parallel for
				for(uint b = 0; b < h_offsets.size(); ++b)
					h_offsets[b] -= workloads[dev].offset[minBGPU];
				thrust::device_vector<int> const d_offsets(h_offsets);
				thrust::device_vector<int> const d_transPeriod(
				    workloads[dev].period.begin() + minBGPU, workloads[dev].period.end());
				thrust::device_vector<Index> const d_eqClassReps(
				    workloads[dev].elems.begin() + workloads[dev].offset[minBGPU],
				    workloads[dev].elems.end());
				// (END) Prepare workloads on the GPU

				int const nBlocks = d_transPeriod.size();
				calculatePartialNormForBlockSq_kernel<<<nBlocks, nThreads3[dev], shMem3[dev],
				                                        stream>>>(
				    dPartialNorm1.data().get(), d_sm_slots[dev].data().get(),
				    d_gWorks[dev].data().get(), dEigVals[dev].data().get(), shWidth,
				    dEigVecs[dev].data(), dEigVecs[dev].LD(), dAdBasis[dev].ptr(),
				    dmbOpSpace[dev].ptr(), d_eqClassReps.data().get(), d_offsets.data().get(),
				    d_transPeriod.data().get(), d_transPeriod.size());
				// cuCHECK(cudaGetLastError());

				cuCHECK(cudaStreamSynchronize(stream));
				thrust::host_vector<RealScalar> partialNorm1(dPartialNorm1);
#pragma omp parallel for
				for(auto alpha = 0; alpha < hSector.dim(); ++alpha)
					for(auto sm = 0; sm < numSMs; ++sm) {
						resGPU(alpha) += partialNorm1[alpha + hSector.dim() * sm];
					}
			}  // (END) #pragma omp task in_reduction(+ : resGPU)

// Workloads on the CPU and the GPU
#pragma omp taskloop nogroup grainsize(1) in_reduction(+ : resCPU)
			for(int b = 0; b < minBGPU; ++b) {
				double const startT = omp_get_wtime();
				// Set CUDA context on a thread
				cuCHECK(cudaSetDevice(dev));
				cudaStream_t stream        = cudaStreamPerThread;
				int          leastPriority = 0, greatestPriority = 0;
				cuCHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
				cuCHECK(
				    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority));
				int const numSMs = GPU::MAGMA::prop(dev).multiProcessorCount;

				int const blockSize   = workloads[dev].offset[b + 1] - workloads[dev].offset[b];
				int const transPeriod = workloads[dev].period[b];

				static_assert(
				    std::is_same_v<Index, std::decay_t<decltype(workloads[dev].elems[0])>>);
				Index* dEqClassRep = nullptr;
				cuCHECK(cudaMallocAsync(&dEqClassRep, blockSize * sizeof(Index), stream));
				cuCHECK(cudaMemcpyAsync(dEqClassRep,
				                        &workloads[dev].elems[workloads[dev].offset[b]],
				                        blockSize * sizeof(Index), cudaMemcpyHostToDevice, stream));

				// 1-1. Calculate the Gram matrix
				RealScalar* dGramMat = nullptr;
				cuCHECK(
				    cudaMallocAsync(&dGramMat, blockSize * blockSize * sizeof(RealScalar), stream));
				int const nBlocks1 = blockSize * (blockSize + 1) / 2;
				calculateGramMat_kernel<<<nBlocks1, nThreads1[dev], shMem1[dev], stream>>>(
				    dGramMat, dmbOpSpace[dev].ptr(), blockSize, dEqClassRep);

				// 1-2. Calculate the eigenstate expectation values
				Scalar* dEEV = nullptr;
				cuCHECK(cudaMallocAsync(&dEEV, hSector.dim() * blockSize * sizeof(Scalar), stream));
				int const nBlocks2 = blockSize;
				calculateEEV_kernel<<<nBlocks2, nThreads2[dev], shMem2[dev], stream>>>(
				    dEEV, d_sm_slots[dev].data().get(), d_gWorks[dev].data().get(),
				    dEigVals[dev].data().get(), shWidth, dEigVecs[dev].data(), dEigVecs[dev].LD(),
				    dAdBasis[dev].ptr(), dmbOpSpace[dev].ptr(), blockSize, dEqClassRep);
				/** Reference implementation on CPU */
				// Eigen::MatrixX<std::complex<RealScalar>> eev(hSector.dim(), blockSize);
				// #pragma omp parallel
				// 				for(int idx = 0; idx < blockSize; ++idx) {
				// 					Index const opIdx   = workloads[dev].elems[idx + workloads[dev].offset[b]];
				// 					auto const  basisOp = hmbOpSpace.basisOp(opIdx).eval();
				// #pragma omp for
				// 					for(auto alpha = 0; alpha < hSector.dim(); ++alpha) {
				// 						eev(alpha, idx) = (eigVecs.col(alpha).adjoint() * adBasis * basisOp
				// 						                   * hSector.basis() * eigVecs.col(alpha))(0);
				// 					}
				// 				}
				// 				eev -= MCAverages(dsolver.eigenvalues(), shWidth, dsolver.eigenvalues(), eev)
				// 				           .eval();

				// 2. Calculate the partial norm within each block
				Eigen::MatrixX<RealScalar> gramMat(blockSize, blockSize);
				cuCHECK(cudaMemcpyAsync(gramMat.data(), dGramMat,
				                        gramMat.size() * sizeof(RealScalar), cudaMemcpyDeviceToHost,
				                        stream));
				Eigen::MatrixX<std::complex<RealScalar>> eev(hSector.dim(), blockSize);
				cuCHECK(cudaMemcpyAsync(eev.data(), dEEV, eev.size() * sizeof(Scalar),
				                        cudaMemcpyDeviceToHost, stream));
				cuCHECK(cudaStreamSynchronize(stream));
				// #ifndef NDEBUG
				// 				{
				// 					double const endT = omp_get_wtime();
				// 	#pragma omp critical
				// 					{
				// 						std::cout << "# (CPU + Dev " << dev << ") Block " << b << "/" << minBGPU
				// 						          << ": blockSize = " << blockSize
				// 						          << ",\t elapsed = " << endT - startT << " (sec)" << std::endl;
				// 					}
				// 				}
				// #else
				// 				{
				// 					double const p = (minBGPU == 1 ? 1 : b / double(minBGPU - 1)) * 100;
				// 					if(abs(p - int(p / 5) * 5.0) < 100.0 / minBGPU) {
				// 						double const endT = omp_get_wtime();
				// 	#pragma omp critical
				// 						std::cout << "# (CPU + Dev " << dev
				// 						          << ")\t Completed GPU tasks: Progress: " << std::setw(4)
				// 						          << int(p / 5) * 5 << "%,\t Block " << b << "/" << minBGPU
				// 						          << ": blockSize = " << blockSize
				// 						          << ",\t Elapsed = " << endT - startT << " (sec)" << std::endl;
				// 					}
				// 				}
				// #endif

				Eigen::MatrixXcd const gramC = gramMat;
				Eigen::MatrixXcd const x     = gramC.selfadjointView<Eigen::Lower>()
				                               .llt()
				                               .solve(eev.transpose())
				                               .transpose();
#pragma omp parallel for
				for(int alpha = 0; alpha < hSector.dim(); ++alpha) {
					double sum = 0.0;
					for(int idx = 0; idx < blockSize; ++idx)
						sum += real(conj(eev(alpha, idx)) * x(alpha, idx));
					resCPU(alpha) += transPeriod * sum;
				}

#ifndef NDEBUG
				double const endT = omp_get_wtime();
	#pragma omp critical
				{
					std::cout << "# (CPU + Dev " << dev << ") Block " << b << "/" << minBGPU
					          << ": blockSize = " << blockSize << ",\t elapsed = " << endT - startT
					          << " (sec)" << std::endl;
				}
#else
				double const p = (minBGPU == 1 ? 1 : b / double(minBGPU - 1)) * 100;
				if(abs(p - int(p / 5) * 5.0) < 100.0 / minBGPU) {
					double const endT = omp_get_wtime();
	#pragma omp critical
					std::cout << "# (CPU + Dev " << dev << ")\t Progress: " << std::setw(4)
					          << int(p / 5) * 5 << "%,\t Block " << b << "/" << minBGPU
					          << ": blockSize = " << blockSize << ",\t Elapsed = " << endT - startT
					          << " (sec)" << std::endl;
				}
#endif

				cuCHECK(cudaFreeAsync(dEqClassRep, stream));
				cuCHECK(cudaFreeAsync(dGramMat, stream));
				cuCHECK(cudaFreeAsync(dEEV, stream));
				cuCHECK(cudaStreamDestroy(stream));
			}  // (END) for(int b = 0; b < minBGPU; ++b)
		}  // (END) for(auto dev = 0;dev < GPU::MAGMA::ngpus(); ++dev)
		double const endT = omp_get_wtime();
		std::cout << "# Total elapsed time on core tasks = " << endT - startT << " (sec)"
		          << std::endl;

		//----- 5. Aggregate the results -----//
		Eigen::ArrayXd res = resCPU + resGPU;

		return res;
	}
}  // namespace StatMech

// 2024/09/26
// The following tests passed:
//         quasiETHmeasure_Boson_onGPU_test

// 100% tests passed, 0 tests failed out of 1

// Total Test time (real) =  51.07 sec