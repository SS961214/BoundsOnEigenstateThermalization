#pragma once
#include "../ManyBodySpaceBase.hpp"
#include "../SubSpace.hpp"

template<class TotalSpace_, typename Scalar,
         typename std::enable_if_t<
             std::is_convertible_v<TotalSpace_, ManyBodySpaceBase<TotalSpace_>> >* = nullptr>
class ParitySector : public SubSpace<TotalSpace_, Scalar> {
	private:
		using TotalSpace = TotalSpace_;
		int m_parity;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param parity
		 * @param sysSize
		 * @param dimLoc
		 */
		template<typename... Args>
		__host__ ParitySector(int parity = +1, int sysSize = 0, Args... args)
		    : ParitySector(parity, TotalSpace(sysSize, args...)) {}

		__host__ __device__ int parity() const { return m_parity; }

		__host__ __device__ int repState(Index j, int trans = 0) const {
			Index innerId = this->basis().outerIndexPtr()[j] + (trans % this->period(j));
			assert(innerId < this->basis().nonZeros() && "innerId < this->basis().nonZeros()");
			return this->basis().innerIndexPtr()[innerId];
		}

		__host__ __device__ int period(Index j) const {
			assert(j < this->dim());
			return this->basis().outerIndexPtr()[j + 1] - this->basis().outerIndexPtr()[j];
		}

		__host__ ParitySector(int parity, TotalSpace const& mbHSpace)
		    : SubSpace<TotalSpace, Scalar>{mbHSpace}, m_parity(parity) {
			Index const L = this->totalSpace().sysSize();
			if(L == 0) return;

			Index dim = 0, numParityEigens;
#pragma omp parallel for reduction(+ : dim)
			for(Index j = 0; j < this->totalSpace().dim(); ++j) {
				if(this->totalSpace().reverse(j) == j) continue;
				dim += 1;
			}
			if(m_parity == 1) {
				dim             = this->totalSpace().dim() - dim / 2;
				numParityEigens = 2 * dim - this->totalSpace().dim();
			}
			else {
				dim             = dim / 2;
				numParityEigens = 0;
			}

			this->basis().resize(this->totalSpace().dim(), dim);
			this->basis().reserve(Eigen::VectorXi::Constant(dim, 2));
			Index eigNum = 0, pairNum = 0;
			for(Index j = 0; j < this->totalSpace().dim(); ++j) {
				auto const reversed = this->totalSpace().reverse(j);
				if(reversed == j && m_parity == 1) {
					this->basis().insert(j, eigNum) = 1;
					eigNum += 1;
				}
				else if(reversed > j) {
					this->basis().insert(j, numParityEigens + pairNum) = 1 / std::sqrt(2.0);
					this->basis().insert(reversed, numParityEigens + pairNum)
					    = m_parity / std::sqrt(2.0);
					pairNum += 1;
				}
			}
			assert(dim == eigNum + pairNum);
			this->basis().makeCompressed();
		}
};
