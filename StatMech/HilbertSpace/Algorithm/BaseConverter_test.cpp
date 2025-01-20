#include "tests.hpp"
#include "BaseConverter.hpp"
#include <iostream>
#include <random>

template<typename Integer>
void test_BaseConverter(BaseConverter<Integer> const& bConv, Index dLoc, Index L, Index max) {
	REQUIRE(dLoc == bConv.radix());
	REQUIRE(L == bConv.length());
	REQUIRE(max == bConv.max());

	std::random_device                   seed_gen;
	std::default_random_engine           engine(seed_gen());
	std::uniform_int_distribution<Index> dist(0, bConv.max() - 1);
	constexpr Index                      nSample = 100000;
	Eigen::ArrayX<Index>                 index;
	if(nSample >= bConv.max()) {
		index.resize(bConv.max());
		for(Index j = 0; j != bConv.max(); ++j) index(j) = j;
	}
	else {
		index = index.NullaryExpr(nSample, [&]() { return dist(engine); });
	}

#pragma omp parallel for
	for(Index sample = 0; sample < index.size(); ++sample) {
		auto j      = index(sample);
		auto config = bConv.number_to_config(j);
		REQUIRE(j == bConv.config_to_number(config));
		for(auto pos = 0; pos < bConv.length(); ++pos) {
			REQUIRE(config(pos) == bConv.digit(j, pos));
		}
	}
}

TEST_CASE("BaseConverter", "test") {
	auto powi = [](Index radix, Index expo) {
		Index res = 1;
		for(Index j = 0; j != expo; ++j) res *= radix;
		return res;
	};

	constexpr int LMax = 20;

	{
		// test Default constructor
		BaseConverter<Index> bConv;
		test_BaseConverter(bConv, Index(0), Index(0), Index(1));
	}
	{
		constexpr int dLoc = 2;
		for(auto L = 0; L <= LMax; ++L) {
			std::cout << "dloc = " << dLoc << ", L = " << L << std::endl;
			BaseConverter<Index> bConv(dLoc, L);
			test_BaseConverter(bConv, dLoc, L, powi(dLoc, L));
		}
	}
	{
		constexpr int dLoc = 4;
		for(auto L = 0; L <= LMax; ++L) {
			std::cout << "dloc = " << dLoc << ", L = " << L << std::endl;
			BaseConverter<Index> bConv(dLoc, L);
			test_BaseConverter(bConv, dLoc, L, powi(dLoc, L));
		}
	}
}