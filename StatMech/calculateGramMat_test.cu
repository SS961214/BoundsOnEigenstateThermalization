#include "../calculateGramMat.cuh"
#include <HilbertSpace>
#include <MatrixGPU>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifdef FLOAT
	using RealScalar               = float;
	RealScalar constexpr precision = 1.0e-6;
#else
	using RealScalar               = double;
	RealScalar constexpr precision = 1.0e-12;
#endif
using Scalar = cuda::std::complex<RealScalar>;
using namespace StatMech;

__global__ void test_kernel(RealScalar* __restrict__ res, unsigned long long* sm_slots,
                            RealScalar* __restrict__* gWorks,
                            mBodyOpSpace<ManyBodyBosonSpace, Scalar> const* __restrict__ opSpacePtr,
                            int const* __restrict__ transSizePtr,
                            Index const* __restrict__ transEqClassRepPtr, int const maxBlockSize) {
	if(blockIdx.x < 1) return;
	auto const& opSpace                  = *opSpacePtr;
	Index const* __restrict__ eqClassRep = transEqClassRepPtr + transSizePtr[blockIdx.x - 1];
	int const             transSize      = transSizePtr[blockIdx.x] - transSizePtr[blockIdx.x - 1];
	extern __shared__ int sm_data[];

	// Get a work slot for this block
	int __shared__                      my_sm, my_slot;
	RealScalar* __restrict__ __shared__ workPtr;
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

	Eigen::Map<Eigen::MatrixX<RealScalar>> gramMat(workPtr + maxBlockSize * maxBlockSize, transSize,
	                                               transSize);
	StatMech::calculateGramMat(gramMat, opSpace, transSize, eqClassRep, sm_data);
	__syncthreads(); // Need to synchronize before accumulating results

	// Accumulate results for verification
	for(int j = threadIdx.x; j < maxBlockSize * maxBlockSize; j += blockDim.x)
		workPtr[j] += gramMat.data()[j];

	// int64_t start = clock64();
	// while(clock64() < start + DELAY_T);
	// assert(gWorks[my_sm * slots_per_sm + my_slot] == blockIdx.x);
	__syncthreads();
	if(threadIdx.x == 0) {
		res[blockIdx.x] = gramMat.norm();
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
	int const L = std::atoi(argv[1]);
	int const N = std::atoi(argv[2]);
	int const m = std::atoi(argv[3]);
	std::cout << "# L: " << L << ", N: " << N << ", m: " << m << std::endl;

	mBodyOpSpace<ManyBodyBosonSpace, Scalar> mbOpSpace(m, L, N);
	auto const                               blocks       = blocksInGramMat(mbOpSpace);
	int                                      maxBlockSize = 0;
	// Cumulative number of sizes of blocks in the gram matrix
	thrust::host_vector<int> h_transSize(blocks.size(), 0);
	// #pragma omp parallel for reduction(max : maxBlockSize)
	for(uint b = 1; b < blocks.size(); ++b) {
		maxBlockSize   = std::max(maxBlockSize, int(blocks[b][0].size()));
		h_transSize[b] = h_transSize[b - 1] + int(blocks[b][0].size());
	}
	int const transSizeSum = h_transSize[blocks.size() - 1];
	std::cout << "# mbOpSpace.dim(): " << mbOpSpace.dim() << std::endl;
	std::cout << "# blocks.size(): " << blocks.size() << ", blocks[0].size(): " << blocks[0].size()
	          << ", maxBlockSize: " << maxBlockSize << ", transSizeSum = " << transSizeSum
	          << std::endl;

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
		int const num_sms = GPU::MAGMA::prop(dev).multiProcessorCount;
		int const workSize_per_block = 2 * maxBlockSize * maxBlockSize;
		std::cout << "#\t\t workSize_per_block: " << workSize_per_block << std::endl;
		thrust::device_vector<unsigned long long> d_sm_slots(num_sms, 0);
		thrust::device_vector<RealScalar> d_data(num_sms * slots_per_sm * workSize_per_block, 0);
		thrust::host_vector<RealScalar*>  gWorks(num_sms * slots_per_sm);
#pragma omp parallel for
		for(uint i = 0; i < gWorks.size(); ++i) {
			gWorks[i] = d_data.data().get() + i * workSize_per_block;
		}
		thrust::device_vector<RealScalar*> d_gWorks(gWorks);

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

		thrust::device_vector<RealScalar>              d_res(blocks.size(), 0);
		ObjectOnGPU<std::decay_t<decltype(mbOpSpace)>> dmbOpSpace(mbOpSpace);
		int nBlocks = blocks.size(), nThreads = 0, shMem = 0;
		{
			int const shMemPerBlock = 0;
			int const shMemPerThread
			    = sizeof(RealScalar) + sizeof(int) * mbOpSpace.actionWorkSize();
			configureKernel(nThreads, shMem, shMemPerBlock, shMemPerThread, test_kernel);
		}
		test_kernel<<<nBlocks, nThreads, shMem>>>(
		    d_res.data().get(), d_sm_slots.data().get(), d_gWorks.data().get(), dmbOpSpace.ptr(),
		    d_transSize.data().get(), d_eqClassReps.data().get(), maxBlockSize);
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
		if(cudaGetLastError() != cudaSuccess) {
			std::cout << "failure" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::cout << "success" << std::endl;

		Eigen::ArrayXd refData  = Eigen::ArrayXd::Zero(maxBlockSize * maxBlockSize);
		Eigen::ArrayXd refNorms = Eigen::ArrayXd::Zero(blocks.size());
		for(uint b = 1; b < blocks.size(); ++b) {
			int const transSize = blocks[b][0].size();

			double                      norm = 0;
			Eigen::Map<Eigen::MatrixXd> gramMat(refData.data(), transSize, transSize);
			for(int idx1 = 0; idx1 < gramMat.cols(); ++idx1) {
				auto const   j          = blocks[b][0][idx1];
				auto const&  adBasisOp1 = mbOpSpace.basisOp(j).adjoint().eval();
				double const res        = adBasisOp1.squaredNorm();
				gramMat(idx1, idx1) += res;
				norm += res * res;

				for(int idx2 = 0; idx2 < idx1; ++idx2) {
					auto const k        = blocks[b][0][idx2];
					auto const basisOp2 = mbOpSpace.basisOp(k).eval();

					auto const res
					    = cuda::std::real((adBasisOp1 * basisOp2).eval().diagonal().sum());
					gramMat(idx1, idx2) += res;
					gramMat(idx2, idx1) += res;
					norm += 2 * res * res;
				}
			}
			refNorms(b) = std::sqrt(norm);
		}

		thrust::host_vector<RealScalar> norms(d_res);
#pragma omp parallel for
		for(uint b = 1; b < blocks.size(); ++b) {
			if(std::abs((norms[b] - refNorms[b]) / (norms[b] + refNorms[b])) > precision) {
#pragma omp critical
				std::cout << "# Error: block = " << b << " (size=" << blocks[b][0].size() << ")"
				          << ", res = " << norms[b] << ", refNorm = " << refNorms[b]
				          << ", diff = " << norms[b] - refNorms[b] << std::endl;
			}
			assert(std::abs((norms[b] - refNorms[b]) / (norms[b] + refNorms[b])) < precision);
		}
		thrust::host_vector<RealScalar> data(d_data);
#pragma omp parallel for
		for(int j = 0; j < maxBlockSize * maxBlockSize; ++j) {
			double res = 0, prev = 0;
			for(int i = 1; i < num_sms * slots_per_sm; ++i) {
				prev = data[j];
				data[j] += data[i * workSize_per_block + j];
				res += (data[j] - prev) - data[i * workSize_per_block + j];
			}
			data[j] += res;
		}
		Eigen::VectorXd diff
		    = Eigen::Map<Eigen::ArrayX<RealScalar>>(data.data(), maxBlockSize * maxBlockSize)
		          .cast<double>()
		      - refData;
		int          maxPos;
		double const maxDiff = diff.cwiseAbs().maxCoeff(&maxPos);

		std::cout << "# sum: \n"
		          << Eigen::Map<Eigen::ArrayX<RealScalar>>(data.data(), maxBlockSize * maxBlockSize)
		                 .transpose()
		          << std::endl;
		std::cout << "# sum:ref: \n" << refData.transpose() << std::endl;
		std::cout << "# diff: \n" << diff.transpose() << std::endl;
		std::cout
		    << "# sum = "
		    << Eigen::Map<Eigen::ArrayX<RealScalar>>(data.data(), maxBlockSize * maxBlockSize).sum()
		    << ", refData.sum() = " << refData.sum() << ", diff = " << diff.cwiseAbs().sum()
		    << ", maxDiff = " << maxDiff << ", maxPos = " << maxPos << std::endl;
	}

	return EXIT_SUCCESS;
}