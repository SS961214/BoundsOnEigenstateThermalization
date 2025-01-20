#include <HilbertSpace>
#include <iostream>

using Scalar = std::complex<double>;

int main(int argc, char* argv[]) {
	if(argc != 6) {
		std::cerr << "Usage: 0.(This) 1.(m) 2.(Lloc) 3.(Nloc) 4.(Lglob) 5.(Nglob)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	constexpr double precision = 1e-12;
	Index const      m         = std::atoi(argv[1]);
	Index const      Lloc      = std::atoi(argv[2]);
	Index const      Nloc      = std::atoi(argv[3]);
	Index const      Lglob     = std::atoi(argv[4]);
	Index const      Nglob     = std::atoi(argv[5]);

	ManyBodyFermionSpace                       locSpace(Lloc, Nloc);
	mBodyOpSpace<ManyBodyFermionSpace, Scalar> locOpSpace(m, locSpace);
	Eigen::MatrixX<Scalar>                     locGramMat(locOpSpace.dim(), locOpSpace.dim());
	std::cout << "# locOpSpace.dim() = " << locOpSpace.dim() << std::endl;
#pragma omp parallel for
	for(Index i = 0; i < locOpSpace.dim(); ++i) {
		auto const basisOp1 = locOpSpace.basisOp(i);
		locGramMat(i, i)    = basisOp1.squaredNorm();
		for(Index j = 0; j < i; ++j) {
			auto const basisOp2 = locOpSpace.basisOp(j);
			locGramMat(i, j)    = (basisOp1.adjoint() * basisOp2).eval().diagonal().sum();
			locGramMat(j, i)    = conj(locGramMat(i, j));
		}
	}

	ManyBodyFermionSpace                       globSpace(Lglob, Nglob);
	mBodyOpSpace<ManyBodyFermionSpace, Scalar> globOpSpace(m, globSpace);
	Eigen::MatrixX<Scalar>                     globGramMat(locOpSpace.dim(), locOpSpace.dim());
	std::cout << "# globOpSpace.dim() = " << globOpSpace.dim() << std::endl;

#pragma omp parallel
	{
		Eigen::ArrayXi config = Eigen::ArrayXi::Zero(globSpace.sysSize());
#pragma omp for ordered
		for(Index i = 0; i < locOpSpace.dim(); ++i) {
			locOpSpace.ordinal_to_config(config.head(locOpSpace.sysSize()), i);
#pragma omp ordered
			{ std::cout << "# i = " << i << ", config = " << config.transpose() << std::endl; }
			auto const opIdx1   = globOpSpace.config_to_ordinal(config);
			auto const basisOp1 = globOpSpace.basisOp(opIdx1);
			globGramMat(i, i)   = basisOp1.squaredNorm();
			for(Index j = 0; j < i; ++j) {
				locOpSpace.ordinal_to_config(config.head(locOpSpace.sysSize()), j);
				auto const opIdx2   = globOpSpace.config_to_ordinal(config);
				auto const basisOp2 = globOpSpace.basisOp(opIdx2);
				globGramMat(i, j)   = (basisOp1.adjoint() * basisOp2).eval().diagonal().sum();
				globGramMat(j, i)   = conj(globGramMat(i, j));
			}
		}
	}

	std::cout << "# locGramMat:\n" << locGramMat << std::endl;
	std::cout << "# globGramMat:\n" << globGramMat << std::endl;
	// double const diff = (locGramMat - globGramMat).cwiseAbs().maxCoeff();
	// std::cout << "# diff = " << diff << std::endl;
	// if(diff > precision) {
	// 	std::cerr << "Error: diff = " << diff << " > " << precision << std::endl;
	// 	return EXIT_FAILURE;
	// }

	return EXIT_SUCCESS;
}