#pragma once

#include "typedefs.hpp"
#include "ManyBodySpaceBase.hpp"
#include "SubSpace.hpp"

template<class TotalSpace_, typename Scalar>
class TransParitySector : public SubSpace<TotalSpace_, Scalar> {
	private:
		using TotalSpace     = TotalSpace_;
		int const m_momentum = 0;
		int       m_parity;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param k
		 * @param sysSize
		 * @param dimLoc
		 */
		template<typename... Args>
		__host__ TransParitySector(int parity = 0, int sysSize = 0, Args... args)
		    : TransParitySector(parity, TotalSpace(sysSize, args...)) {}

		__host__ __device__ int momentum() const { return m_momentum; }
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

		__host__ TransParitySector(int parity, TotalSpace const& mbHSpace)
		    : SubSpace<TotalSpace, Scalar>{mbHSpace}, m_parity(parity) {
			static_assert(std::is_convertible_v<TotalSpace_, ManyBodySpaceBase<TotalSpace_>>);
			static_assert(Eigen::NumTraits<Scalar>::IsComplex);

			using Real           = typename Eigen::NumTraits<Scalar>::Real;
			auto const& totSpace = this->totalSpace();
			Index const L        = totSpace.sysSize();

			if(L == 0) return;
			if(m_parity != +1 && m_parity != -1) {
				std::cerr << "Error: " << __FILE__ << ":" << __LINE__
				          << ": Parity must be either +1 or -1: parity(" << m_parity << ")"
				          << std::endl;
				std::exit(EXIT_FAILURE);
			}

			totSpace.compute_transEqClass();
			std::vector<Index> state_to_transEqClass(totSpace.dim());
#pragma omp parallel for
			for(Index eqClass = 0; eqClass < totSpace.transEqDim(); ++eqClass) {
				for(auto trans = 0; trans < totSpace.transPeriod(eqClass); ++trans) {
					auto const rep                  = totSpace.transEqClassRep(eqClass);
					auto const stateNum             = totSpace.translate(rep, trans);
					state_to_transEqClass[stateNum] = eqClass;
				}
			}
#ifndef NDEBUG
			for(Index stateNum = 0; stateNum < totSpace.dim(); ++stateNum) {
				auto const eqClass = state_to_transEqClass[stateNum];
				assert(eqClass < totSpace.transEqDim());
				auto const rep      = totSpace.transEqClassRep(eqClass);
				bool       appeared = false;
				for(auto trans = 0; trans < totSpace.transPeriod(eqClass); ++trans) {
					auto const state = totSpace.translate(rep, trans);
					if(state == stateNum) appeared = true;
				}
				assert(appeared);
			}
#endif

			// Calculate the dimension of the subspace
			Index                                nParityEigens = 0, nParityPairs = 0;
			std::vector<Index>                   parityEigens(totSpace.transEqDim());
			std::vector<std::pair<Index, Index>> parityPairs(totSpace.transEqDim());
			for(Index eqClass = 0; eqClass < totSpace.transEqDim(); ++eqClass) {
				auto const rep        = totSpace.transEqClassRep(eqClass);
				auto const reversed   = totSpace.reverse(rep);
				auto const revEqClass = state_to_transEqClass[reversed];
				if(eqClass > revEqClass) continue;
				if(eqClass == revEqClass) { parityEigens[nParityEigens++] = eqClass; }
				else { parityPairs[nParityPairs++] = std::make_pair(eqClass, revEqClass); }
			}
			assert(nParityEigens + 2 * nParityPairs == totSpace.transEqDim());

			Index const dim = nParityPairs + (m_parity == +1 ? nParityEigens : 0);
			this->basis().resize(totSpace.dim(), dim);
			this->basis().reserve(Eigen::VectorXi::Constant(dim, 2 * L));
#pragma omp parallel for
			for(Index basisNum = 0; basisNum < nParityPairs; ++basisNum) {
				auto const [eqClass, revEqClass] = parityPairs[basisNum];
				auto const eqClassRep            = totSpace.transEqClassRep(eqClass);
				auto const revEqClassRep         = totSpace.transEqClassRep(revEqClass);
				Real const norm                  = Real(sqrt(2.0 * totSpace.transPeriod(eqClass)));

				for(auto trans = 0; trans != totSpace.transPeriod(eqClass); ++trans) {
					auto const stateNum = totSpace.translate(eqClassRep, trans);
					this->basis().insert(stateNum, basisNum) = 1.0 / norm;

					auto const revStateNum = totSpace.translate(revEqClassRep, trans);
					this->basis().insert(revStateNum, basisNum) = m_parity / norm;
				}
			}

			if(m_parity == +1) {
				for(Index basisNum = nParityPairs; basisNum < dim; ++basisNum) {
					auto const eqClass = parityEigens[basisNum - nParityPairs];
					Real const norm    = Real(sqrt(Real(totSpace.transPeriod(eqClass))));
					for(auto trans = 0; trans != totSpace.transPeriod(eqClass); ++trans) {
						auto const eqClassRep = totSpace.transEqClassRep(eqClass);
						auto const stateNum   = totSpace.translate(eqClassRep, trans);
						this->basis().insert(stateNum, basisNum) = 1.0 / norm;
					}
				}
			}
			this->basis().makeCompressed();
		}
};