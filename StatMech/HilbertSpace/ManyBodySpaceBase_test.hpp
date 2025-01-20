#pragma once

#include "tests.hpp"
#include "ManyBodySpaceBase.hpp"
#include <iostream>
#include <random>

template<class Derived>
void test_ManyBodySpaceBase(ManyBodySpaceBase<Derived> const& mbSpace, Index sysSize) {
	std::cout << "# test_ManyBodySpaceBase" << std::endl;
	std::cout << "mbSpace.dim() = " << mbSpace.dim() << std::endl;
	if(sysSize == 0) REQUIRE(mbSpace.dim() == 0);
	REQUIRE(mbSpace.sysSize() == sysSize);

	std::random_device                   seed_gen;
	std::default_random_engine           engine(seed_gen());
	std::uniform_int_distribution<Index> dist(0, mbSpace.dim() - 1);
	constexpr int                        nSample = 100;
	Eigen::ArrayX<Index>                 index;
	if(nSample > mbSpace.dim()) {
		index.resize(mbSpace.dim());
		for(Index j = 0; j != mbSpace.dim(); ++j) index(j) = j;
	}
	else {
		index = index.NullaryExpr(nSample, [&]() { return dist(engine); });
	}

// test locState
// test ordinal_to_config
// test config_to_ordinal
#pragma omp parallel
	{
		Eigen::ArrayX<Index> config(mbSpace.sysSize());
#pragma omp for
		for(auto sample = 0; sample < index.size(); ++sample) {
			Index stateNum = index(sample);
			mbSpace.ordinal_to_config(config, stateNum);
			REQUIRE(stateNum == mbSpace.config_to_ordinal(config));
			for(auto pos = 0; pos < mbSpace.sysSize(); ++pos) {
				REQUIRE(config(pos) == mbSpace.locState(stateNum, pos));
			}
		}
	}
	std::cout << "-- Passed tests for config <-> ordinal conversion" << std::endl;

	// test for translation operations
	mbSpace.compute_transEqClass();
	std::cout << "transEqDim() = " << mbSpace.transEqDim() << std::endl;

	Eigen::ArrayXi appeared = Eigen::ArrayXi::Zero(mbSpace.dim());
#pragma omp parallel for
	for(Index eqClassNum = 0; eqClassNum != mbSpace.transEqDim(); ++eqClassNum) {
		auto stateNum = mbSpace.transEqClassRep(eqClassNum);
		appeared(stateNum) += 1;
		for(auto trans = 1; trans != mbSpace.transPeriod(eqClassNum); ++trans) {
			auto translated = mbSpace.translate(stateNum, trans);
			appeared(translated) += 1;
		}
	}
#pragma omp parallel for
	for(Index stateNum = 0; stateNum != mbSpace.dim(); ++stateNum) REQUIRE(appeared(stateNum) == 1);

	// 		// test for state_to_transEqClass
	// 		// test for state_to_transShift
	// #pragma omp parallel for
	// 	for(auto sample = 0; sample < index.size(); ++sample) {
	// 		Index     stateNum   = index(sample);
	// 		auto const eqClass    = mbSpace.state_to_transEqClass(stateNum);
	// 		auto const eqClassRep = mbSpace.transEqClassRep(eqClass);
	// 		auto const trans      = mbSpace.state_to_transShift(stateNum);
	// 		REQUIRE(stateNum == mbSpace.translate(eqClassRep, trans));
	// 	}
	std::cout << "-- Passed tests for translation operations" << std::endl;

// test for reverse()
#pragma omp parallel
	{
		Eigen::ArrayX<Index> config(mbSpace.sysSize());
#pragma omp for
		for(auto sample = 0; sample < index.size(); ++sample) {
			Index stateNum = index(sample);
			mbSpace.ordinal_to_config(config, stateNum);
			auto reversed = mbSpace.config_to_ordinal(config.reverse());
			REQUIRE(reversed == mbSpace.reverse(stateNum));
		}
	}
	std::cout << "-- Passed tests for reverse operations" << std::endl;
	std::cout << "# Passed test_ManyBodySpaceBase\n" << std::endl;
}

#ifdef __NVCC__
	#include "ManyBodySpaceBase_test.cuh"
#endif