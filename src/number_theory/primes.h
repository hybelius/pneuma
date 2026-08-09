#include <cstdint>
#include <print>
#include <type_traits>
#include <utility>
#include <vector>

namespace pneuma::primes
{

template<typename T>
struct prime_factor
{
    static_assert(std::is_integral_v<T>);
    T factor;
    uint32_t multiplicity;
};


template<typename T>
std::vector<prime_factor<T>> get_prime_factors(T number)
{
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_signed_v<T>)
    {
        number = number > 0 ? number : -number;
    }

    // Trivial case
    if (number == 0 || number == 1)
    {
        return {};
    }

    // Vector to fill
    std::vector<prime_factor<T>> factors;

    // Handle factors of 2 separately
    uint32_t multiplicity_2 = 0;
    while ((number & 1) == 0)
    {
        multiplicity_2++;
        number >>= 1;
    }
    if (multiplicity_2 > 0)
    {
        factors.emplace_back(static_cast<T>(2), multiplicity_2);
        if (number == 1)
        {
            return factors;
        }
    }

    // Only need to look for factors up to sqrt(number)
    uint32_t log2_sqrt = 1; // smallest possible number at this point is 3
    while (number > (1<<(log2_sqrt<<1)))
    {
        log2_sqrt++;
    }
    T sqrt_est = 1 << log2_sqrt;

    // Sieve of eratoshenes to keep track of prime factors
    // std::vector<bool> prime_mask(static_cast<size_t>(sqrt_est), false);
    // prime_mask[0] = true;
    // prime_mask[1] = true;

    // The bool at index i is false if p = 3 + 2 * i is potentially a prime
    std::vector<bool> prime_mask(static_cast<size_t>(sqrt_est) >> 1, false);

    // for (T prime = 2; prime < sqrt_est; prime++)
    for (uint32_t i = 0; i < prime_mask.size(); i++)
    {
        if (prime_mask[i])
        {
            // Number is marked as not prime
            continue;
        }
        T prime = 3 + 2 * i;

        uint32_t multiplicity = 0;
        while (number % prime == 0)
        {
            multiplicity++;
            number /= prime;
        }
        if (multiplicity > 0)
        {
            // Found a new factor
            factors.emplace_back(prime, multiplicity);
        }
        if (number == 1)
        {
            // All factors found
            break;
        }

        // Update table of primes
        // Multiples less than prime ** 2 are guaranteed to already be marked
        // Even multiples are not represented in the mask, as it only handles odd primes
        for (uint32_t j = (prime * prime - 3) / 2; j < prime_mask.size(); j += prime)
        {
            prime_mask[j] = true;
        }
    }

    if (number > 1)
    {
        // One factor is greater than sqrt(number), so hasn't been found yet
        // Since all other factor have been divided away, what remains is the last factor,
        // with multiplicity 1.
        factors.emplace_back(number, 1);
    }

    return factors;
}

} // namespace numeric::primes
