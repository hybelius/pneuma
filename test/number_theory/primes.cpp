#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <utility>

#include "number_theory/primes.h"

using namespace pneuma::primes;

TEST_CASE("Prime factorization", "[primes]")
{
    std::vector< std::pair< int, std::vector<prime_factor<int>> > > test_values = {
        { 6, {{2, 1}, {3, 1}} },
        { 1, {} },
        { 25, {{5, 2}} },
        { 1001, {{7, 1}, {11, 1}, {13, 1}} },
        { 10001, {{73, 1}, {137, 1}} },
        { 931, {{7, 2}, {19, 1}} },
        { 1024, {{2, 10}} }
    };

    for (auto& [num, factors_ref] : test_values)
    {
        auto factors = get_prime_factors(num);
        REQUIRE(factors.size() == factors_ref.size());

        for (int i = 0; i < factors.size(); i++)
        {
            REQUIRE(factors[i].factor == factors_ref[i].factor);
            REQUIRE(factors[i].multiplicity == factors_ref[i].multiplicity);
        }
    }
}
