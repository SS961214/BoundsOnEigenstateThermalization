#include "tests.hpp"
#include "HilbertSpace.hpp"
#include <random>
#include <iostream>

TEST_CASE("HilbertSpace", "test") {
	std::random_device              seed_gen;
	std::default_random_engine      engine(seed_gen());
	std::uniform_int_distribution<> dist(0, 100000);

	Index testLoop = 100;

	{
		// Default constructor
		{
			HilbertSpace<int> hSpace;
			REQUIRE(hSpace.dim() == 0);
		}
		for(Index n = 0; n != testLoop; ++n) {
			Index             dim = dist(engine);
			HilbertSpace<int> hSpace(dim);
			REQUIRE(hSpace.dim() == dim);
		}
	}
	{
		// Copy constructor
		for(Index n = 0; n != testLoop; ++n) {
			Index             dim = dist(engine);
			HilbertSpace<int> hSpace1(dim);
			HilbertSpace<int> hSpace2(hSpace1);
			REQUIRE(hSpace1.dim() == hSpace2.dim());
		}
	}
	{
		// Move constructor
		for(Index n = 0; n != testLoop; ++n) {
			Index             dim = dist(engine);
			HilbertSpace<int> hSpace1(dim);
			HilbertSpace<int> hSpace2(std::move(hSpace1));
			REQUIRE(hSpace2.dim() == dim);
		}
	}

	{
		// Copy assignment operator
		for(Index n = 0; n != testLoop; ++n) {
			auto              dim = dist(engine);
			HilbertSpace<int> hSpace1(dim);
			HilbertSpace<int> hSpace2;
			hSpace2 = hSpace1;
			REQUIRE(hSpace1.dim() == hSpace2.dim());
		}
	}
	{
		// Move assignment operator
		for(Index n = 0; n != testLoop; ++n) {
			Index             dim = dist(engine);
			HilbertSpace<int> hSpace1(dim);
			HilbertSpace<int> hSpace2;
			hSpace2 = std::move(hSpace1);
			REQUIRE(hSpace2.dim() == dim);
		}
	}

	{
		// Equality operator
		for(Index n = 0; n != testLoop; ++n) {
			Index             dim = dist(engine);
			HilbertSpace<int> hSpace1(dim);
			HilbertSpace<int> hSpace2(hSpace1);
			REQUIRE(hSpace1.dim() == hSpace2.dim());
			REQUIRE(hSpace1 == hSpace2);
		}
	}
}