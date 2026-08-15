#include <cmath>
#include <cassert>
#include <exception>

#include "fft.h"
#include "number_theory/primes.h"

namespace pneuma::transforms
{

struct sin_and_cos_trf
{
    float sin;
    float cos;
};

inline size_t floor_log2(size_t number)
{
    if (number <= 1)
    {
        return 0;
    }

    size_t log2 = 0;
    for (size_t n_bits = 32; n_bits > 0; n_bits >>= 1)
    {
        size_t upper_bits = number >> n_bits;
        if (upper_bits)
        {
            log2 += n_bits;
            number = upper_bits;
        }
    }
    return log2;
}

// Calculates the discrete sine and cosine transforms of type 2
// for odd and even inputs of the input data.
// This corresponds to all but the "top" function call in a recursive implementation.
// The final stage is done in separate sine and cosine functions to avoid calculating the transform
// that remains unused.
inline std::vector<sin_and_cos_trf> _discrete_sine_and_cosine_pow_2_type_2_common_stages(const std::span<const float> input)
{
    size_t trf_size = input.size();
    size_t log2_size = floor_log2(trf_size);
    assert(trf_size == (1 << log2_size) && "Input must be of a power-of-two size");

    // Construct two compute buffers:
    // buffer_1 is initialized to equal `input` (in both .sin and .cos).
    // buffer_2 is default initialized to have the same size as input.
    // The computation writes back and forth between buffer_1 and buffer_2
    std::vector<sin_and_cos_trf> buffer_1;
    std::vector<sin_and_cos_trf> buffer_2(input.size());
    buffer_1.reserve(input.size());
    for (auto val : input)
    {
        buffer_1.emplace_back(val, val);
    }
    std::span<sin_and_cos_trf> stage_input(buffer_1);
    std::span<sin_and_cos_trf> stage_output(buffer_2);

    for (uint32_t stage = 0; stage < log2_size - 1; stage++)
    {
        uint32_t stride = (trf_size >> (stage + 1));
        uint32_t block_size = 1 << (stage + 1);

// Helper define that converts a `k` index inside a "block" to a vector index in the total input/output vector
#define IDX_CONV(k) (block_idx + (k) * stride)

        auto half_size = block_size / 2;
        for (uint32_t block_idx = 0; block_idx < stride; block_idx++)
        {
            // Special handling of k = 0 for cos transform and k == size - 1 for sin transform
            // Cosine transform values at index k correspond to Sine transform values at index k - 1,
            // so the iteration over k starts at 1 and therefore excludes cos[0] and sin[size - 1]
            auto cos_even_k0 = stage_input[IDX_CONV(0)].cos;
            auto cos_odd_k0  = stage_input[IDX_CONV(1)].cos;

            stage_output[IDX_CONV(0)].cos              = cos_even_k0 + cos_odd_k0;
            stage_output[IDX_CONV(block_size - 1)].sin = cos_even_k0 - cos_odd_k0;
        }

        for (uint32_t k = 1; k < half_size; k++)
        {
            for (uint32_t block_idx = 0; block_idx < stride; block_idx++)
            {
                // TODO: implement custom sin/cos for constrained input angles
                float angle = M_PIf * float(k) / (2 * block_size);
                float twiddle_sin, twiddle_cos;
                sincosf(angle, &twiddle_sin, &twiddle_cos);

                auto cos_even = stage_input[IDX_CONV(2 * k)].cos;
                auto cos_odd  = stage_input[IDX_CONV(2 * k + 1)].cos;
                auto sin_even = stage_input[IDX_CONV(2 * (k - 1))].sin;
                auto sin_odd  = stage_input[IDX_CONV(2 * (k - 1) + 1)].sin;

                auto cos_sum  = cos_even + cos_odd;
                auto cos_diff = cos_even - cos_odd;
                auto sin_sum  = sin_even + sin_odd;
                auto sin_diff = sin_even - sin_odd;

                stage_output[IDX_CONV(k)].cos   = cos_sum * twiddle_cos + sin_diff * twiddle_sin;
                stage_output[IDX_CONV(k-1)].sin = sin_sum * twiddle_cos - cos_diff * twiddle_sin;

                // Input block has half the size of output block (size N)
                // Extract values at N/2 + k by "reflecting" indices across N/2
                stage_output[IDX_CONV(block_size - k)].cos     = - cos_sum * twiddle_sin + sin_diff * twiddle_cos;
                stage_output[IDX_CONV(block_size - k - 1)].sin =   sin_sum * twiddle_sin + cos_diff * twiddle_cos;
            }
        }

        for (uint32_t block_idx = 0; block_idx < stride; block_idx++)
        {
            // Separate handling of k = size // 2 because it reflects to itself
            auto sin_even_khalf = stage_input[IDX_CONV(2 * (half_size - 1))].sin;
            auto sin_odd_khalf  = stage_input[IDX_CONV(2 * (half_size - 1) + 1)].sin;
            stage_output[IDX_CONV(half_size)].cos         = (sin_even_khalf - sin_odd_khalf) * M_SQRT1_2;
            stage_output[IDX_CONV(half_size - 1)].sin     = (sin_even_khalf + sin_odd_khalf) * M_SQRT1_2;
        }

        // Switch input/output role between buffer_1 and buffer_2 by swapping the spans
        std::swap(stage_input, stage_output);
#undef IDX_CONV
    }

    // Last iteration was stage == log2_size - 2.
    if ((log2_size % 2) == 0)
    {
        // If log2_size is even, buffer_2 was used as the last output
        return buffer_2;
    }
    else
    {
        // If log2_size is odd,  buffer_1 was used as the last output
        return buffer_1;
    }
}


std::vector<float> _discrete_cosine_transform_pow_2_type_2(const std::span<const float> input)
{
    auto trf_size = input.size();
    if (trf_size == 1)
    {
        // Trivial base case
        return std::vector<float>({input[0]});
    }

    auto stage_input = _discrete_sine_and_cosine_pow_2_type_2_common_stages(input);
    std::vector<float> output(input.size());

    auto half_size = trf_size / 2;
    auto stride = 1;

    // Special handling of k = 0
    auto cos_even_k0 = stage_input[0].cos;
    auto cos_odd_k0  = stage_input[1].cos;

    output[0] = cos_even_k0 + cos_odd_k0;

    for (uint32_t k = 1; k < half_size; k++)
    {
        float angle = M_PIf * float(k) / (2 * trf_size);
        float twiddle_sin, twiddle_cos;
        sincosf(angle, &twiddle_sin, &twiddle_cos);

        auto cos_even = stage_input[2 * k].cos;
        auto cos_odd  = stage_input[2 * k + 1].cos;
        auto sin_even = stage_input[2 * (k - 1)].sin;
        auto sin_odd  = stage_input[2 * (k - 1) + 1].sin;

        auto cos_sum  = cos_even + cos_odd;
        auto sin_diff = sin_even - sin_odd;

        output[k]            =   cos_sum * twiddle_cos + sin_diff * twiddle_sin;
        output[trf_size - k] = - cos_sum * twiddle_sin + sin_diff * twiddle_cos;
    }

    // Separate handling of k = size // 2 because it reflects to itself
    auto sin_even_k_half = stage_input[2 * (half_size - 1)].sin;
    auto sin_odd_k_half  = stage_input[2 * (half_size - 1) + 1].sin;
    output[half_size] = (sin_even_k_half - sin_odd_k_half) * M_SQRT1_2;

    return output;
}

std::vector<float> _discrete_cosine_transform_pow_2_type_3(const std::span<const float> input)
{
    auto trf_size = input.size();
    if (trf_size == 1)
    {
        // Trivial base case
        return std::vector<float>({0.5f * input[0]});
    }
    size_t log2_size = floor_log2(trf_size);
    assert(trf_size == (1 << log2_size) && "Input must be of a power-of-two size");

    // Construct two compute buffers:
    // buffer_1 is initialized to equal `input` in .cos.
    // buffer_2 is default initialized to have the same size as input.
    // The computation writes back and forth between buffer_1 and buffer_2
    std::vector<sin_and_cos_trf> buffer_1;
    std::vector<sin_and_cos_trf> buffer_2(input.size());
    buffer_1.reserve(input.size());
    for (auto val : input)
    {
        buffer_1.emplace_back(0, val);
    }
    // Cosine transform type 3 is defined as 0.5 * x[0] + a sum over x[n], n=1..N-1
    // Multiplying x[0] by 0.5 lets us include it as the first term in the sum, simplifying the algorithm
    buffer_1[0].cos *= 0.5f;
    std::span<sin_and_cos_trf> stage_input(buffer_1);
    std::span<sin_and_cos_trf> stage_output(buffer_2);

    for (size_t stage = 0; stage < log2_size - 1; stage++)
    {
        size_t stride = (trf_size >> (stage + 1));
        size_t block_size = 1 << (stage + 1);
        size_t half_size = block_size / 2;

        for (size_t k = 0; k < half_size; k++)
        {
            float angle = M_PI * (k + 0.5f) / block_size;
            float twiddle_sin, twiddle_cos;
            sincosf(angle, &twiddle_sin, &twiddle_cos);

            // block_idx = 0 never needs the even sine transform
            auto cos_even = stage_input[(2 * k) * stride].cos;
            auto [sin_odd, cos_odd] = stage_input[(2 * k + 1) * stride];
            auto cos_odd_term = cos_odd * twiddle_cos - sin_odd * twiddle_sin;

            // Note: the sine transform at these indices is never used, so we skip calculating them
            stage_output[k * stride].cos                    = cos_even + cos_odd_term;
            stage_output[(block_size - k - 1) * stride].cos = cos_even - cos_odd_term;

            for (size_t block_idx = 1; block_idx < stride; block_idx++)
            {
                auto [sin_even, cos_even] = stage_input[block_idx + (2 * k) * stride];
                auto [sin_odd,  cos_odd]  = stage_input[block_idx + (2 * k + 1) * stride];
                auto sin_odd_term = sin_odd * twiddle_cos + cos_odd * twiddle_sin;
                auto cos_odd_term = cos_odd * twiddle_cos - sin_odd * twiddle_sin;

                stage_output[block_idx + k * stride] = {
                    sin_even + sin_odd_term,
                    cos_even + cos_odd_term
                };
                // Input block has half the size of output block (size N)
                // Extract values at N/2 + k by "reflecting" indices across N/2
                stage_output[block_idx + (block_size - k - 1) * stride] = {
                    - sin_even + sin_odd_term,
                      cos_even - cos_odd_term
                };
            }
        }

        std::swap(stage_input, stage_output);
    }

    // Final iteration: stage = log2_size - 1
    // `stride` is 1 and only the cosine transform output needs to be calculated
    // This is the "block_idx == 0" step from previous stages
    size_t half_size = trf_size / 2;
    std::vector<float> output(trf_size);

    for (size_t k = 0; k < half_size; k++)
    {
        float angle = M_PI * (k + 0.5f) / trf_size;
        float twiddle_sin, twiddle_cos;
        sincosf(angle, &twiddle_sin, &twiddle_cos);

        auto cos_even = stage_input[2 * k].cos;
        auto [sin_odd, cos_odd] = stage_input[2 * k + 1];
        auto cos_odd_term = cos_odd * twiddle_cos - sin_odd * twiddle_sin;

        output[k]                = cos_even + cos_odd_term;
        output[trf_size - k - 1] = cos_even - cos_odd_term;
    }

    return output;
}

std::vector<float> _discrete_sine_transform_pow_2_type_2(const std::span<const float> input)
{
    auto trf_size = input.size();
    if (trf_size == 1)
    {
        // Trivial base case
        return std::vector<float>({input[0]});
    }

    auto stage_input = _discrete_sine_and_cosine_pow_2_type_2_common_stages(input);
    std::vector<float> output(input.size());

    auto half_size = trf_size / 2;
    auto stride = 1;

    // Special handling of k = 0
    auto cos_even_k0 = stage_input[0].cos;
    auto cos_odd_k0  = stage_input[1].cos;

    output[trf_size - 1] = cos_even_k0 - cos_odd_k0;

    for (uint32_t k = 1; k < half_size; k++)
    {
        float angle = M_PIf * float(k) / (2 * trf_size);
        float twiddle_sin, twiddle_cos;
        sincosf(angle, &twiddle_sin, &twiddle_cos);

        auto cos_even = stage_input[2 * k].cos;
        auto cos_odd  = stage_input[2 * k + 1].cos;
        auto sin_even = stage_input[2 * (k - 1)].sin;
        auto sin_odd  = stage_input[2 * (k - 1) + 1].sin;

        auto sin_sum  = sin_even + sin_odd;
        auto cos_diff = cos_even - cos_odd;

        output[k - 1]            = sin_sum * twiddle_cos - cos_diff * twiddle_sin;
        output[trf_size - k - 1] = sin_sum * twiddle_sin + cos_diff * twiddle_cos;
    }

    // Separate handling of k = size // 2 because it reflects to itself
    auto sin_even_k_half = stage_input[2 * (half_size - 1)].sin;
    auto sin_odd_k_half  = stage_input[2 * (half_size - 1) + 1].sin;
    output[half_size - 1] = (sin_even_k_half + sin_odd_k_half) * M_SQRT1_2;

    return output;
}

std::vector<float> _discrete_sine_transform_pow_2_type_3(const std::span<const float> input)
{
    auto trf_size = input.size();
    if (trf_size == 1)
    {
        // Trivial base case
        return std::vector<float>({0.5f * input[0]});
    }
    size_t log2_size = floor_log2(trf_size);
    assert(trf_size == (1 << log2_size) && "Input must be of a power-of-two size");

    // Construct two compute buffers:
    // buffer_1 is initialized to equal `input` in .sin.
    // buffer_2 is default initialized to have the same size as input.
    // The computation writes back and forth between buffer_1 and buffer_2
    std::vector<sin_and_cos_trf> buffer_1;
    std::vector<sin_and_cos_trf> buffer_2(input.size());
    buffer_1.reserve(input.size());
    for (auto val : input)
    {
        buffer_1.emplace_back(val, 0);
    }
    // Sine transform type 3 is defined as 0.5 * x[N-1] + a sum over x[n], n=0..N-2
    // Multiplying x[N-1] by 0.5 lets us include it as the last term in the sum, simplifying the algorithm
    buffer_1[trf_size - 1].sin *= 0.5f;
    std::span<sin_and_cos_trf> stage_input(buffer_1);
    std::span<sin_and_cos_trf> stage_output(buffer_2);

    for (size_t stage = 0; stage < log2_size - 1; stage++)
    {
        size_t stride = (trf_size >> (stage + 1));
        size_t block_size = 1 << (stage + 1);
        size_t half_size = block_size / 2;

        for (size_t k = 0; k < half_size; k++)
        {
            float angle = M_PI * (k + 0.5f) / block_size;
            float twiddle_sin, twiddle_cos;
            sincosf(angle, &twiddle_sin, &twiddle_cos);

            for (size_t block_idx = 0; block_idx < stride - 1; block_idx++)
            {
                auto [sin_even, cos_even] = stage_input[block_idx + (2 * k) * stride];
                auto [sin_odd,  cos_odd]  = stage_input[block_idx + (2 * k + 1) * stride];
                auto sin_even_term = sin_even * twiddle_cos - cos_even * twiddle_sin;
                auto cos_even_term = cos_even * twiddle_cos + sin_even * twiddle_sin;

                stage_output[block_idx + k * stride] = {
                    sin_odd + sin_even_term,
                    cos_odd + cos_even_term
                };
                // Input block has half the size of output block (size N)
                // Extract values at N/2 + k by "reflecting" indices across N/2
                stage_output[block_idx + (block_size - k - 1) * stride] = {
                    - sin_odd + sin_even_term,
                      cos_odd - cos_even_term
                };
            }

            // block_idx = stride-1 never needs the odd cosine transform
            size_t block_idx = stride - 1;
            auto sin_odd = stage_input[block_idx + (2 * k + 1) * stride].sin;
            auto [sin_even, cos_even] = stage_input[block_idx + (2 * k) * stride];
            auto sin_even_term = sin_even * twiddle_cos - cos_even * twiddle_sin;

            // Note: the cos transform at these indices is never used, so we skip calculating them
            stage_output[block_idx + k * stride].sin                    =   sin_odd + sin_even_term;
            stage_output[block_idx + (block_size - k - 1) * stride].sin = - sin_odd + sin_even_term;
        }

        std::swap(stage_input, stage_output);
    }

    // Final iteration: stage = log2_size - 1
    // `stride` is 1 and only the cosine transform output needs to be calculated
    // This is the "block_idx == N-1" step from previous stages
    // block_idx = stride - 1 == 0
    size_t half_size = trf_size / 2;
    std::vector<float> output(trf_size);

    for (size_t k = 0; k < half_size; k++)
    {
        float angle = M_PI * (k + 0.5f) / trf_size;
        float twiddle_sin, twiddle_cos;
        sincosf(angle, &twiddle_sin, &twiddle_cos);

        auto sin_odd = stage_input[2 * k + 1].sin;
        auto [sin_even, cos_even] = stage_input[2 * k];
        auto sin_even_term = sin_even * twiddle_cos - cos_even * twiddle_sin;

        output[k]                =   sin_odd + sin_even_term;
        output[trf_size - k - 1] = - sin_odd + sin_even_term;
    }

    return output;
}

std::vector<float> discrete_cosine_transform_pow_2(const std::span<const float> input, SIN_COS_TRF_TYPE type)
{
    switch(type)
    {
        case SIN_COS_TRF_TYPE::II:
            return _discrete_cosine_transform_pow_2_type_2(input);
        case SIN_COS_TRF_TYPE::III:
            return _discrete_cosine_transform_pow_2_type_3(input);
        default:
            throw std::invalid_argument("Currently only the Cosine type II and III transforms are implemented.");
    }
}

std::vector<float> discrete_sine_transform_pow_2(const std::span<const float> input, SIN_COS_TRF_TYPE type)
{
    switch(type)
    {
        case SIN_COS_TRF_TYPE::II:
            return _discrete_sine_transform_pow_2_type_2(input);
        case SIN_COS_TRF_TYPE::III:
            return _discrete_sine_transform_pow_2_type_3(input);
        default:
            throw std::invalid_argument("Currently only the Sine type II and III transforms are implemented.");
    }
}

std::vector<float> discrete_cosine_transform(const std::span<const float> input, SIN_COS_TRF_TYPE type)
{
    auto factors = primes::get_prime_factors(input.size());
    if (factors.size() != 0 && (factors.size() > 1 || factors[0].factor != 2))
    {
        throw std::domain_error("Cosine transform is currently only implemented for power-of-two sized input.");
    }

    return discrete_cosine_transform_pow_2(input, type);
}

std::vector<float> discrete_sine_transform(const std::span<const float> input, SIN_COS_TRF_TYPE type)
{
    auto factors = primes::get_prime_factors(input.size());
    if (factors.size() != 0 && (factors.size() != 1 || factors[0].factor != 2))
    {
        throw std::domain_error("Sine transform is currently only implemented for power-of-two sized input");
    }

    return discrete_sine_transform_pow_2(input, type);
}

} // namespace numeric::transforms