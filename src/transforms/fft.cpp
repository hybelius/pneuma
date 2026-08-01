#include <cmath>
#include <cassert>
#include <exception>
#include <span>

#include "fft.h"
#include "primes.h"

namespace numeric::transforms
{

struct sin_and_cos_trf
{
    float sin;
    float cos;
};


// Calculates the discrete sine and cosine transforms of type 2
// for odd and even inputs of the input data.
// This corresponds to all but the "top" function call in a recursive implementation.
// The final stage is done in separate sine and cosine functions to avoid calculating the transform
// that remains unused.
inline std::vector<sin_and_cos_trf> _discrete_sine_and_cosine_pow_2_type_2_common_stages(const std::span<const float> input)
{
    size_t trf_size = input.size();
    size_t log2_size = 1;
    while ((trf_size >> log2_size) > 1)
    {
        log2_size++;
    }
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
#define IDX_CONV(k) (block_idx + k * stride)

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


std::vector<float> discrete_cosine_transform_pow_2(const std::span<const float> input, SIN_COS_TRF_TYPE type)
{
    if (type == SIN_COS_TRF_TYPE::II)
    {
        return _discrete_cosine_transform_pow_2_type_2(input);
    }
    else
    {
        throw std::invalid_argument("Currently only the Cosine type II transform is implemented.");
    }
}


std::vector<float> discrete_sine_transform_pow_2(const std::span<const float> input, SIN_COS_TRF_TYPE type)
{
    if (type == SIN_COS_TRF_TYPE::II)
    {
        return _discrete_sine_transform_pow_2_type_2(input);
    }
    else
    {
        throw std::invalid_argument("Currently only the Sine type II transform is implemented.");
    }
}


std::vector<float> discrete_cosine_transform(const std::span<const float> input, SIN_COS_TRF_TYPE type)
{
    auto factors = primes::get_prime_factors(input.size());
    if (factors.size() != 1 || factors[0].factor != 2)
    {
        throw std::domain_error("Cosine transform is currently only implemented for power-of-two sized input.");
    }

    return discrete_cosine_transform_pow_2(input, type);
}


std::vector<float> discrete_sine_transform(const std::span<const float> input, SIN_COS_TRF_TYPE type)
{
    auto factors = primes::get_prime_factors(input.size());
    if (factors.size() != 1 || factors[0].factor != 2)
    {
        throw std::domain_error("Sine transform is currently only implemented for power-of-two sized input");
    }

    return discrete_sine_transform_pow_2(input, type);
}

} // namespace numeric::transforms