#include "calculateEEV.cuh"
#include "Microcanonical.hpp"
#include <HilbertSpace>
#include <MatrixGPU>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <stdio.h>

#ifdef FLOAT
	using RealScalar               = float;
	RealScalar constexpr precision = 1.0e-6;
#else
	using RealScalar               = double;
	RealScalar constexpr precision = 1.0e-12;
#endif
using Scalar = cuda::std::complex<RealScalar>;
using namespace StatMech;

__global__ void test_kernel(Scalar* __restrict__ res, unsigned long long* sm_slots,
                            Scalar* __restrict__* gWorks, RealScalar const* __restrict__ dEigValPtr,
                            RealScalar const      dE, Scalar const* __restrict__ dEigVecPtr,
                            Index const LD, SparseMatrix<Scalar> const* __restrict__ adBasisPtr,
                            mBodyOpSpace<ManyBodyBosonSpace, Scalar> const* __restrict__ opSpacePtr,
                            int const* __restrict__ transSizePtr,
                            Index const* __restrict__ transEqClassRepPtr) {
	if(blockIdx.x < 1) return;
	SparseMatrix<Scalar> const& adBasis   = *adBasisPtr;
	int const                   sectorDim = adBasis.rows();
	auto const&                 opSpace   = *opSpacePtr;
	Index const* __restrict__ eqClassRep  = transEqClassRepPtr + transSizePtr[blockIdx.x - 1];
	int const             transSize       = transSizePtr[blockIdx.x] - transSizePtr[blockIdx.x - 1];
	extern __shared__ int sm_data[];

	// Get a work slot for this block
	int __shared__                  my_sm, my_slot;
	Scalar* __restrict__ __shared__ workPtr;
	if(threadIdx.x == 0) {
		printf("# %s: block = %d/%d,\t transSize = %d\n", __func__, blockIdx.x, gridDim.x,
		       transSize);
		my_sm   = get_smid();
		my_slot = get_slot(sm_slots + my_sm);
		workPtr = gWorks[my_sm * slots_per_sm + my_slot];
	}
	__syncthreads();
	assert(my_sm >= 0);
	assert(0 <= my_slot && my_slot < slots_per_sm);
	assert(workPtr != nullptr);

	/** To fulfil the alignment requirements, we should first assign larger-sized pointers (i.e., Scalar pointers) and then assign smaller-sized pointers (i.e., RealScalar pointers). */
	Scalar* __restrict__ const eevPtr = reinterpret_cast<Scalar*>(workPtr);
	Eigen::Map<Eigen::MatrixX<Scalar>> eev(eevPtr, sectorDim, transSize);
	Scalar* __restrict__ expvalPtr = eev.data() + eev.size();

	StatMech::calculateEEV(eev, dEigValPtr, dE, dEigVecPtr, LD, adBasis, opSpace, transSize,
	                       eqClassRep, expvalPtr, sm_data);
	__syncthreads();

	// Copy the expectation values to the global memory for verification on the host.
	for(auto alpha = threadIdx.x; alpha < sectorDim; alpha += blockDim.x) {
		for(auto opIdx = 0; opIdx < transSize; ++opIdx) {
			auto const idx = alpha + sectorDim * (opIdx + transSizePtr[blockIdx.x - 1]);
			res[idx]       = eev(alpha, opIdx);
			// res[idx] = 1;
		}
	}

	__syncthreads();
	if(threadIdx.x == 0) {
		release_slot(sm_slots + my_sm, my_slot);
		printf("# (END) %s: block = %d/%d\n\n", __func__, blockIdx.x, gridDim.x);
	}
}

int main(int argc, char* argv[]) {
	if(argc < 4) {
		std::cerr << "Usage: 0.(This) 1.(L) 2.(N) 3.(m)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	GPU::MAGMA::get_controller();  // Initialize MAGMA
	int const     L        = std::atoi(argv[1]);
	int const     N        = std::atoi(argv[2]);
	int const     m        = std::atoi(argv[3]);
	constexpr int momentum = 0;
	std::cout << "# L: " << L << ", N: " << N << ", m: " << m << std::endl;

	mBodyOpSpace<ManyBodyBosonSpace, Scalar>                           mbOpSpace(m, L, N);
	TransSector<std::decay_t<decltype(mbOpSpace.baseSpace())>, Scalar> hSector(
	    momentum, mbOpSpace.baseSpace());
	auto const adBasis      = hSector.basis().adjoint().eval();
	auto const blocks       = blocksInGramMat(mbOpSpace);
	int        maxBlockSize = 0;
	// Cumulative number of sizes of blocks in the gram matrix
	thrust::host_vector<int> h_transSize(blocks.size(), 0);
	// #pragma omp parallel for reduction(max : maxBlockSize)
	for(uint b = 1; b < blocks.size(); ++b) {
		maxBlockSize   = std::max(maxBlockSize, int(blocks[b][0].size()));
		h_transSize[b] = h_transSize[b - 1] + int(blocks[b][0].size());
	}
	int const transSizeSum = h_transSize[blocks.size() - 1];
	std::cout << "# mbOpSpace.dim(): " << mbOpSpace.dim() << ", hSector.dim() = " << hSector.dim()
	          << std::endl;
	std::cout << "# blocks.size(): " << blocks.size() << ", blocks[0].size(): " << blocks[0].size()
	          << ", maxBlockSize: " << maxBlockSize << ", transSizeSum = " << transSizeSum
	          << std::endl;

	// Prepare random matrix
	int const                            seed = 0;
	std::mt19937                         mt(seed);
	std::normal_distribution<RealScalar> Gaussian(0.0, 1.0);
	Eigen::MatrixX<Scalar>               mat = Eigen::MatrixX<Scalar>::NullaryExpr(
        hSector.dim(), hSector.dim(), [&]() { return Scalar(Gaussian(mt), Gaussian(mt)); });
	GPU::MatrixGPU<decltype(mat)>               dmat(mat);
	GPU::SelfAdjointEigenSolver<decltype(dmat)> solver(dmat);
	auto const energyRange   = solver.eigenvalues().maxCoeff() - solver.eigenvalues().minCoeff();
	RealScalar const shWidth = 0.2 * energyRange;

	for(auto dev = 0; dev < GPU::MAGMA::ngpus(); ++dev) {
		std::cout << "# dev: " << dev
		          << ",\t multiProcessorCount: " << GPU::MAGMA::prop(dev).multiProcessorCount
		          << ",\t maxBlocksPerMultiProcessor: "
		          << GPU::MAGMA::prop(dev).maxBlocksPerMultiProcessor
		          << ",\t maxThreadsPerMultiProcessor: "
		          << GPU::MAGMA::prop(dev).maxThreadsPerMultiProcessor
		          << ",\t maxThreadsPerBlock: " << GPU::MAGMA::prop(dev).maxThreadsPerBlock
		          << ",\t sharedMemPerBlockOptin: " << GPU::MAGMA::prop(dev).sharedMemPerBlockOptin
		          << std::endl;

		cuCHECK(cudaSetDevice(dev));
		/** Prepare working area for each active blocks */
		int const num_sms = GPU::MAGMA::prop(dev).multiProcessorCount;
		int const workSize_per_block = hSector.dim() * maxBlockSize;
		std::cout << "#\t\t workSize_per_block: " << workSize_per_block << std::endl;
		thrust::device_vector<unsigned long long> d_sm_slots(num_sms, 0);
		thrust::device_vector<Scalar> d_data(num_sms * slots_per_sm * workSize_per_block, 0);
		thrust::host_vector<Scalar*>  gWorks(num_sms * slots_per_sm);
#pragma omp parallel for
		for(uint i = 0; i < gWorks.size(); ++i) {
			gWorks[i] = d_data.data().get() + i * workSize_per_block;
		}
		thrust::device_vector<Scalar*> d_gWorks(gWorks);

		/** Copy representatives to device */
		thrust::device_vector<int>   d_transSize(h_transSize);
		thrust::device_vector<Index> d_eqClassReps(transSizeSum, 0);
#pragma omp parallel for
		for(uint b = 1; b < blocks.size(); ++b) {
			cuCHECK(cudaMemcpyAsync(d_eqClassReps.data().get() + h_transSize[b - 1],
			                        blocks[b][0].data(), blocks[b][0].size() * sizeof(Index),
			                        cudaMemcpyHostToDevice, cudaStreamPerThread));
			assert(blocks[b][0].size() == h_transSize[b] - h_transSize[b - 1]);
		}
		cuCHECK(cudaDeviceSynchronize());
		// Check if the data for eqClassReps is correctly copied to the device.
		{
			thrust::device_vector<Index> h_eqClassReps(d_eqClassReps);
#pragma omp parallel for
			for(uint b = 1; b < blocks.size(); ++b) {
				for(uint i = 0; i < blocks[b][0].size(); ++i) {
					auto elem = h_eqClassReps[h_transSize[b - 1] + i];
					if(elem != blocks[b][0][i]) {
						std::cout << "# Error: b = " << b << ", i = " << i << ", elem = " << elem
						          << ", blocks[b][0][i] = " << blocks[b][0][i] << std::endl;
					}
					assert(elem == blocks[b][0][i]);
				}
			}
		}

		thrust::device_vector<Scalar>                        d_res(hSector.dim() * transSizeSum, 0);
		ObjectOnGPU<std::decay_t<decltype(mbOpSpace)>> const dmbOpSpace(mbOpSpace);
		ObjectOnGPU<SparseMatrix<Scalar>> const              dAdBasis(adBasis);
		thrust::device_vector<RealScalar> const              d_eigVals(solver.eigenvalues().begin(),
		                                                               solver.eigenvalues().end());
		GPU::MatrixGPU<std::decay_t<decltype(solver.eigenvectors())>> const d_eigVecs(
		    solver.eigenvectors());
		int nBlocks = blocks.size(), nThreads = 0, shMem = 0;
		{
			int       nThreads1 = 0, shMem1 = 0;
			int const shMemPerBlock1 = 0;
			int const shMemPerThread1
			    = sizeof(RealScalar) + sizeof(int) * mbOpSpace.actionWorkSize();
			configureKernel(nThreads1, shMem1, shMemPerBlock1, shMemPerThread1, test_kernel);

			int       nThreads2 = 0, shMem2 = 0;
			int const shMemPerBlock2  = sizeof(Scalar) * hSector.dim();
			int const shMemPerThread2 = sizeof(int) * mbOpSpace.actionWorkSize();
			configureKernel(nThreads2, shMem2, shMemPerBlock2, shMemPerThread2, test_kernel);

			nThreads = (nThreads1 <= nThreads2 ? nThreads1 : nThreads2);
			shMem1   = shMemPerBlock1 + nThreads * shMemPerThread1;
			shMem2   = shMemPerBlock2 + nThreads * shMemPerThread2;
			shMem    = (shMem1 >= shMem2 ? shMem1 : shMem2);
			std::cout << "#\t nThreads1 = " << nThreads1 << ", shMem1 = " << shMem1 << "\n"
			          << "#\t nThreads2 = " << nThreads2 << ", shMem2 = " << shMem2 << "\n"
			          << "#\t nThreads  = " << nThreads << ", shMem  = " << shMem << "\n"
			          << std::endl;
		}
		test_kernel<<<nBlocks, nThreads, shMem>>>(
		    d_res.data().get(), d_sm_slots.data().get(), d_gWorks.data().get(),
		    d_eigVals.data().get(), shWidth, d_eigVecs.data(), d_eigVecs.LD(), dAdBasis.ptr(),
		    dmbOpSpace.ptr(), d_transSize.data().get(), d_eqClassReps.data().get());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
		if(cudaGetLastError() != cudaSuccess) {
			std::cout << "failure" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::cout << "success" << std::endl;

		// Verify the results
		thrust::host_vector<Scalar>        h_res(d_res);
		Eigen::Map<Eigen::MatrixX<Scalar>> res(h_res.data(), hSector.dim(), transSizeSum);
		// std::cout << "# res: \n" << res.cast<std::complex<RealScalar>>() << std::endl;

		auto const             eigVecFull = (hSector.basis() * solver.eigenvectors()).eval();
		Eigen::MatrixX<Scalar> eevRef = Eigen::MatrixX<Scalar>::Zero(hSector.dim(), transSizeSum);
#pragma omp parallel for schedule(dynamic, 1)
		for(uint b = 1; b < blocks.size(); ++b) {
			for(uint idx = 0; idx < blocks[b][0].size(); ++idx) {
				Index const opIdx   = blocks[b][0][idx];
				auto const  j       = idx + h_transSize[b - 1];
				auto const  basisOp = mbOpSpace.basisOp(opIdx).eval();
				for(auto alpha = 0; alpha < hSector.dim(); ++alpha) {
					eevRef(alpha, j)
					    = (eigVecFull.col(alpha).adjoint() * basisOp * eigVecFull.col(alpha))(0);
				}
			}
		}
		eevRef -= MCAverages(solver.eigenvalues().cast<double>(), shWidth,
		                     solver.eigenvalues().cast<double>(), eevRef)
		              .eval();
		auto const shellDims = get_shellDims(solver.eigenvalues(), shWidth, solver.eigenvalues());

#pragma omp parallel for schedule(dynamic, 1)
		for(uint b = 1; b < blocks.size(); ++b) {
			for(uint idx = 0; idx < blocks[b][0].size(); ++idx) {
				auto const j = idx + h_transSize[b - 1];
				for(auto alpha = 0; alpha < hSector.dim(); ++alpha) {
					double const diff = cuda::std::abs(res(alpha, j) - eevRef(alpha, j));
					if(diff >= precision) {
#pragma omp critical
						std::cerr << "# Error: (block, idx, alpha) = (" << b << ", " << idx << ", "
						          << alpha << "), shellDim = " << shellDims(alpha)
						          << ", diff = " << diff << std::endl;
					}
				}
			}
		}

		double const diff = (res - eevRef).cwiseAbs().maxCoeff();
		if(diff >= precision) {
			for(auto j = 0; j < transSizeSum; ++j) {
				double const err = (res.col(j) - eevRef.col(j)).cwiseAbs().maxCoeff();
				if(err >= precision) {
					std::cout << "# j = " << j << ", err = " << err << std::endl;
					std::cout << "# res.col(j):\n"
					          << res.col(j).transpose().cast<std::complex<RealScalar>>()
					          << std::endl;
					std::cout << "# eevRef.col(j):\n"
					          << eevRef.col(j).transpose().cast<std::complex<RealScalar>>()
					          << std::endl;
					std::cout
					    << "# res.col(j) - eevRef.col(j):\n"
					    << (res.col(j) - eevRef.col(j)).transpose().cast<std::complex<RealScalar>>()
					    << std::endl;
				}
			}
		}
		std::cout << "# diff = " << diff << std::endl;
		assert(diff < precision);
	}

	return EXIT_SUCCESS;
}