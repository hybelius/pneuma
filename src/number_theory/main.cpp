#include <iostream>
#include <print>
#include <string>

#include "primes.h"

using namespace numeric;

template <typename T>
void print_factors(std::vector<primes::prime_factor<T>> factors)
{
    std::string out;
    T number = 1;
    for (auto [factor, multiplicity] : factors)
    {
        for (uint32_t i = 0; i < multiplicity; i++)
        {
            number *= factor;
        }
        out += std::format("({}^{})", factor, multiplicity);
    }
    std::println("{} = {}", number, out);
}

int main()
{
    print_factors(primes::get_prime_factors(10));
    print_factors(primes::get_prime_factors(100));
    print_factors(primes::get_prime_factors(1000));
    print_factors(primes::get_prime_factors(1234321));
}
