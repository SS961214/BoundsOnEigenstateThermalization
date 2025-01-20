#include "tests.hpp"
#include "mBodyOpSpace_Fermion_GramMat.hpp"
#include <complex>
#include <cassert>
#include <sys/time.h>
#include <omp.h>

using Scalar = std::complex<double>;

static inline double getETtime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

TEST_CASE("mBodyOpSpace_Fermion_GramMat", "test") {
	constexpr int LMax = 8;
	// constexpr int NMax = 6;

	for(auto L = LMax; L <= LMax; ++L)
		for(auto N = 1; N <= L; ++N)
			for(auto m = 1; m <= std::min(N, L - N); ++m) {
				std::cout << "# L = " << L << ", N = " << N << ", m = " << m << std::endl;
				mBodyOpSpace<ManyBodyFermionSpace, Scalar> opSpace(m, L, N);

				double     blockTime = getETtime();
				auto const blocks    = blocksInGramMat(opSpace);
				blockTime            = getETtime() - blockTime;
				std::cout << "# blockTime = " << blockTime << "\n";

				// for(Index b = 0; b < blocks.size(); ++b) {
				// 	for(Index j = 0; j < blocks[b].size(); ++j) { std::cout << blocks[b][j].size() << " "; }
				// 	std::cout << std::endl;
				// }

				int dim = 0;
#pragma omp parallel for reduction(+ : dim)
				for(auto j = 0; j < Index(blocks.size()); ++j)
					for(auto k = 0; k < Index(blocks[j].size()); ++k) dim += blocks[j][k].size();

				REQUIRE(opSpace.dim() == dim);
				if(opSpace.dim() > 4000) continue;

				double          gramTime = getETtime();
				Eigen::MatrixXd gramMat  = Eigen::MatrixXd::Zero(opSpace.dim(), opSpace.dim());
#pragma omp parallel for schedule(dynamic, 10)
				for(auto rep = 0; rep < Index(blocks[0].size()); ++rep) {
					auto const   j = blocks[0][rep][0];
					double const coeff
					    = std::real((opSpace.basisOp(j).adjoint() * opSpace.basisOp(j))
					                    .eval()
					                    .diagonal()
					                    .sum());
					for(auto idx = 0; idx < Index(blocks[0][rep].size()); ++idx) {
						auto const j  = blocks[0][rep][idx];
						gramMat(j, j) = coeff;
					}
				}

				int zeros = 0;
#pragma omp parallel for schedule(dynamic, 10)
				for(auto b = 1; b < Index(blocks.size()); ++b) {
					Eigen::MatrixXd smallGMat(blocks[b][0].size(), blocks[b][0].size());
					for(auto idx1 = 0; idx1 < Index(blocks[b][0].size()); ++idx1) {
						auto const j          = blocks[b][0][idx1];
						auto const adBasisOp1 = opSpace.basisOp(j).adjoint().eval();
						for(auto idx2 = 0; idx2 <= idx1; ++idx2) {
							auto const k        = blocks[b][0][idx2];
							auto const basisOp2 = opSpace.basisOp(k);
							double     coeff
							    = std::real((adBasisOp1 * basisOp2).eval().diagonal().sum());

							smallGMat(idx1, idx2) = coeff;
							smallGMat(idx2, idx1) = smallGMat(idx1, idx2);
							for(auto trans = 0; trans < Index(blocks[b].size()); ++trans) {
								auto const translated1            = opSpace.translate(j, trans);
								auto const translated2            = opSpace.translate(k, trans);
								gramMat(translated1, translated2) = coeff;
								gramMat(translated2, translated1)
								    = gramMat(translated1, translated2);
							}
						}
					}

					Eigen::SelfAdjointEigenSolver<decltype(smallGMat)> solver(
					    smallGMat, Eigen::EigenvaluesOnly);
					solver.eigenvalues();

					for(auto l = 0; l < solver.eigenvalues().size(); ++l) {
						if(std::abs(solver.eigenvalues()(l)) < 1.0e-6) zeros += 1;
					}
				}
				gramTime = getETtime() - gramTime;
				std::cout << "#  gramTime = " << gramTime << "\n";
				// std::cout << gramMat << std::endl;
				if(zeros > 0) std::cout << "#\t zeros = " << zeros << std::endl;
				REQUIRE(zeros == 0);

				double          reftime    = getETtime();
				Eigen::MatrixXd gramMatRef = Eigen::MatrixXd::Zero(opSpace.dim(), opSpace.dim());
#pragma omp parallel for schedule(dynamic, 10)
				for(auto j = 0; j < opSpace.dim(); ++j) {
					auto const adBasisOp1 = opSpace.basisOp(j).adjoint().eval();
					for(auto k = 0; k <= j; ++k) {
						auto const basisOp2 = opSpace.basisOp(k);
						gramMatRef(j, k)
						    = std::real((adBasisOp1 * basisOp2).eval().diagonal().sum());
						gramMatRef(k, j) = gramMatRef(j, k);
					}
				}
				reftime = getETtime() - reftime;
				REQUIRE(gramMatRef.isApprox(gramMat));
				std::cout << "#   refTime = " << reftime << "\n" << std::endl;
			}
}