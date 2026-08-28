#include <cmath>
#include <cassert>
#include <exception>

#include "fft.h"
#include "number_theory/primes.h"

namespace pneuma::transforms
{

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
// for a single factor "2".
inline void _discrete_sine_and_cosine_type_2_factor_2(const std::span<const float> input_sin, const std::span<const float> input_cos,
                                                      const std::span<float> output_sin, const std::span<float> output_cos,
                                                      size_t stride, size_t block_size)
{
    auto half_size = block_size / 2;
    for (uint32_t block_idx = 0; block_idx < stride; block_idx++)
    {
        // Special handling of k = 0 for cos transform and k == size - 1 for sin transform
        // Cosine transform values at index k correspond to Sine transform values at index k - 1,
        // so the iteration over k starts at 1 and therefore excludes cos[0] and sin[size - 1]
        auto cos_even_k0 = input_cos[block_idx];
        auto cos_odd_k0  = input_cos[block_idx + stride];

        output_cos[block_idx]                             = cos_even_k0 + cos_odd_k0;
        output_sin[block_idx + stride * (block_size - 1)] = cos_even_k0 - cos_odd_k0;
    }

    for (uint32_t k = 1; k < half_size; k++)
    {
        for (uint32_t block_idx = 0; block_idx < stride; block_idx++)
        {
            // TODO: implement custom sin/cos for constrained input angles
            float angle = M_PIf * float(k) / (2 * block_size);
            float twiddle_sin, twiddle_cos;
            sincosf(angle, &twiddle_sin, &twiddle_cos);

            auto cos_even = input_cos[block_idx + stride * (2 * k)];
            auto cos_odd  = input_cos[block_idx + stride * (2 * k + 1)];
            auto sin_even = input_sin[block_idx + stride * (2 * (k - 1))];
            auto sin_odd  = input_sin[block_idx + stride * (2 * (k - 1) + 1)];

            auto cos_sum  = cos_even + cos_odd;
            auto cos_diff = cos_even - cos_odd;
            auto sin_sum  = sin_even + sin_odd;
            auto sin_diff = sin_even - sin_odd;

            output_cos[block_idx + stride * (k)]   = cos_sum * twiddle_cos + sin_diff * twiddle_sin;
            output_sin[block_idx + stride * (k-1)] = sin_sum * twiddle_cos - cos_diff * twiddle_sin;

            // Input block has half the size of output block (size N)
            // Extract values at N/2 + k by "reflecting" indices across N/2
            output_cos[block_idx + stride * (block_size - k)]     = - cos_sum * twiddle_sin + sin_diff * twiddle_cos;
            output_sin[block_idx + stride * (block_size - k - 1)] =   sin_sum * twiddle_sin + cos_diff * twiddle_cos;
        }
    }

    for (uint32_t block_idx = 0; block_idx < stride; block_idx++)
    {
        // Separate handling of k = size // 2 because it reflects to itself
        auto sin_even_khalf = input_sin[block_idx + stride * (2 * (half_size - 1))];
        auto sin_odd_khalf  = input_sin[block_idx + stride * (2 * (half_size - 1) + 1)];
        output_cos[block_idx + stride * (half_size)]         = (sin_even_khalf - sin_odd_khalf) * M_SQRT1_2;
        output_sin[block_idx + stride * (half_size - 1)]     = (sin_even_khalf + sin_odd_khalf) * M_SQRT1_2;
    }
}

// Calculates the discrete sine and cosine transforms of type 2
// for a single factor "3".
inline void _discrete_sine_and_cosine_type_2_factor_3(const std::span<const float> input_sin, const std::span<const float> input_cos,
                                                      const std::span<float> output_sin, const std::span<float> output_cos,
                                                      size_t stride, size_t block_size)
{
    // Input size is N = 3 Q
    auto third_size = block_size / 3;
    const float sin_py_by_3 = sqrtf(3.0f) * 0.5f; // sin(pi / 3) == sin(2 pi / 3) == sqrt(3)/2 TODO: replace with constant
    for (uint32_t block_idx = 0; block_idx < stride; block_idx++)
    {
        // Special handling of k = 0 and k = 2 Q
        auto cos_mod0_k0 = input_cos[block_idx];
        auto cos_mod1_k0 = input_cos[block_idx + stride];
        auto cos_mod2_k0 = input_cos[block_idx + stride * 2];

        output_cos[block_idx]                                 =   cos_mod1_k0 +        cos_mod0_k0 + cos_mod2_k0;
        output_cos[block_idx + stride * (2 * third_size)]     = - cos_mod1_k0 + 0.5 * (cos_mod0_k0 + cos_mod2_k0);
        output_sin[block_idx + stride * (2 * third_size - 1)] =         sin_py_by_3 * (cos_mod0_k0 - cos_mod2_k0);
    }

    // Cosine transform values at index k correspond to Sine transform values at index k - 1
    for (uint32_t k = 1; k < third_size; k++)
    {
        for (uint32_t block_idx = 0; block_idx < stride; block_idx++)
        {
            // TODO: implement custom sin/cos for constrained input angles
            float angle = M_PIf * float(k) / (3 * block_size);
            float twiddle_sin, twiddle_cos;
            sincosf(angle, &twiddle_sin, &twiddle_cos);

            auto cos_mod0 = input_cos[block_idx + stride * (3 * k)];
            auto cos_mod1 = input_cos[block_idx + stride * (3 * k + 1)];
            auto cos_mod2 = input_cos[block_idx + stride * (3 * k + 2)];
            auto sin_mod0 = input_sin[block_idx + stride * (3 * (k - 1))];
            auto sin_mod1 = input_sin[block_idx + stride * (3 * (k - 1) + 1)];
            auto sin_mod2 = input_sin[block_idx + stride * (3 * (k - 1) + 2)];

            auto cos_sum  = cos_mod0 + cos_mod2;
            auto cos_diff = cos_mod0 - cos_mod2;
            auto sin_sum  = sin_mod0 + sin_mod2;
            auto sin_diff = sin_mod0 - sin_mod2;

            output_cos[block_idx + stride * (k)]   = cos_mod1 + twiddle_cos * cos_sum + twiddle_sin * sin_diff;
            output_sin[block_idx + stride * (k-1)] = sin_mod1 + twiddle_cos * sin_sum - twiddle_sin * cos_diff;

            // output at k > Q == N/3 is acquired by reflecting k across Q and across 2Q
            // input at k is used for output at k' == 2Q - k and at k'' = 2Q + k
            // angle is (pi / N) (2 Q - k) == (2 pi / 3 - pi k / N)
            auto twiddle_cos_2Qmk =      - 0.5f * twiddle_cos + sin_py_by_3 * twiddle_sin;
            auto twiddle_sin_2Qmk = sin_py_by_3 * twiddle_cos +        0.5f * twiddle_sin;

            // angle is (pi / N) (2 Q + k) == (2 pi / 3 + pi k / N)
            auto twiddle_cos_2Qpk =      - 0.5f * twiddle_cos - sin_py_by_3 * twiddle_sin;
            auto twiddle_sin_2Qpk = sin_py_by_3 * twiddle_cos -        0.5f * twiddle_sin;

            output_cos[block_idx + stride * (2 * third_size - k)]     = - cos_mod1 - twiddle_cos_2Qmk * cos_sum
                                                                                   + twiddle_sin_2Qmk * sin_diff;
            output_sin[block_idx + stride * (2 * third_size - k - 1)] =   sin_mod1 + twiddle_cos_2Qmk * sin_sum
                                                                                   + twiddle_sin_2Qmk * cos_diff;

            output_cos[block_idx + stride * (2 * third_size + k)]     = - cos_mod1 - twiddle_cos_2Qpk * cos_sum
                                                                                   - twiddle_sin_2Qpk * sin_diff;
            output_sin[block_idx + stride * (2 * third_size + k - 1)] = - sin_mod1 - twiddle_cos_2Qpk * sin_sum
                                                                                   + twiddle_sin_2Qpk * cos_diff;
        }
    }

    for (uint32_t block_idx = 0; block_idx < stride; block_idx++)
    {
        // Special handling of k = N - 1 and k = Q
        auto sin_mod0_kQ = input_sin[block_idx + stride * (3 * (third_size - 1))];
        auto sin_mod1_kQ = input_sin[block_idx + stride * (3 * (third_size - 1) + 1)];
        auto sin_mod2_kQ = input_sin[block_idx + stride * (3 * (third_size - 1) + 2)];

        output_sin[block_idx + stride * (block_size - 1)] = - sin_mod1_kQ +        sin_mod0_kQ + sin_mod2_kQ;
        output_sin[block_idx + stride * (third_size - 1)] =   sin_mod1_kQ + 0.5 * (sin_mod0_kQ + sin_mod2_kQ);
        output_cos[block_idx + stride * third_size]       =         sin_py_by_3 * (sin_mod0_kQ - sin_mod2_kQ);
    }
}

// Calculates the discrete sine and cosine transforms of type 2
// for odd and even inputs of the input data.
// This corresponds to all but the "top" function call in a recursive implementation.
// The final stage is done in separate sine and cosine functions to avoid calculating the transform
// that remains unused.
inline std::pair<std::vector<float>, std::vector<float>> _discrete_sine_and_cosine_pow_2_type_2_common_stages(const std::span<const float> input)
{
    size_t trf_size = input.size();
    size_t log2_size = floor_log2(trf_size);
    assert(trf_size == (1 << log2_size) && "Input must be of a power-of-two size");

    // Construct four compute buffers:
    // buffer_1_sin and buffer_1_cos are initialized to equal `input`.
    // buffer_2_sin and buffer_2_cos are default initialized with the same size as input.
    // The computation writes back and forth between buffer_1_[sin|cos] and buffer_2_[sin|cos]
    std::vector<float> buffer_1_sin, buffer_1_cos;
    std::vector<float> buffer_2_sin(trf_size), buffer_2_cos(trf_size);
    buffer_1_sin.reserve(trf_size);
    buffer_1_cos.reserve(trf_size);
    for (auto val : input)
    {
        buffer_1_sin.emplace_back(val);
        buffer_1_cos.emplace_back(val);
    }
    std::span<float> stage_input_sin(buffer_1_sin);
    std::span<float> stage_input_cos(buffer_1_cos);
    std::span<float> stage_output_sin(buffer_2_sin);
    std::span<float> stage_output_cos(buffer_2_cos);

    for (uint32_t stage = 0; stage < log2_size - 1; stage++)
    {
        uint32_t stride = (trf_size >> (stage + 1));
        uint32_t block_size = 1 << (stage + 1);

        _discrete_sine_and_cosine_type_2_factor_2(stage_input_sin, stage_input_cos, stage_output_sin, stage_output_cos, stride, block_size);

        std::swap(stage_input_sin, stage_output_sin);
        std::swap(stage_input_cos, stage_output_cos);
    }

    // Last iteration was stage == log2_size - 2.
    if ((log2_size % 2) == 0)
    {
        // If log2_size is even, buffer_2_[sin|cos] was used as the last output
        return std::make_pair(buffer_2_sin, buffer_2_cos);
    }
    else
    {
        // If log2_size is odd,  buffer_1_[sin|cos] was used as the last output
        return std::make_pair(buffer_1_sin, buffer_1_cos);
    }
}

inline void _discrete_sine_type_3_factor_2(const std::span<const float> input_sin, const std::span<const float> input_cos,
                                           const std::span<float> output_sin, const std::span<float> output_cos,
                                           size_t stride, size_t block_size)
{
    for (size_t k = 0; k < block_size / 2; k++)
    {
        float angle = M_PI * (k + 0.5f) / block_size;
        float twiddle_sin, twiddle_cos;
        sincosf(angle, &twiddle_sin, &twiddle_cos);

        for (size_t block_idx = 0; block_idx < stride - 1; block_idx++)
        {
            auto sin_even = input_sin[block_idx + stride * (2 * k)];
            auto sin_odd  = input_sin[block_idx + stride * (2 * k + 1)];
            auto cos_even = input_cos[block_idx + stride * (2 * k)];
            auto cos_odd  = input_cos[block_idx + stride * (2 * k + 1)];
            auto sin_even_term = sin_even * twiddle_cos - cos_even * twiddle_sin;
            auto cos_even_term = cos_even * twiddle_cos + sin_even * twiddle_sin;

            output_sin[block_idx + stride * k] = sin_odd + sin_even_term;
            output_cos[block_idx + stride * k] = cos_odd + cos_even_term;

            // Input block has half the size of output block (size N)
            // Extract values at N/2 + k by "reflecting" indices across N/2
            output_sin[block_idx + stride * (block_size - k - 1)] = - sin_odd + sin_even_term;
            output_cos[block_idx + stride * (block_size - k - 1)] =   cos_odd - cos_even_term;
        }

        // block_idx = stride-1 never needs the odd cosine transform
        size_t block_idx = stride - 1;
        auto sin_odd  = input_sin[block_idx + stride * (2 * k + 1)];
        auto sin_even = input_sin[block_idx + stride * (2 * k)];
        auto cos_even = input_cos[block_idx + stride * (2 * k)];
        auto sin_even_term = sin_even * twiddle_cos - cos_even * twiddle_sin;

        // Note: the cos transform at these indices is never used, so we skip calculating them
        output_sin[block_idx + stride * k]                    =   sin_odd + sin_even_term;
        output_sin[block_idx + stride * (block_size - k - 1)] = - sin_odd + sin_even_term;
    }
}

inline void _discrete_cosine_type_3_factor_2(const std::span<const float> input_sin, const std::span<const float> input_cos,
                                             const std::span<float> output_sin, const std::span<float> output_cos,
                                             size_t stride, size_t block_size)
{
    for (size_t k = 0; k < block_size / 2; k++)
    {
        float angle = M_PI * (k + 0.5f) / block_size;
        float twiddle_sin, twiddle_cos;
        sincosf(angle, &twiddle_sin, &twiddle_cos);

        // block_idx = 0 never needs the even sine transform
        auto cos_even = input_cos[stride * (2 * k)];
        auto cos_odd  = input_cos[stride * (2 * k + 1)];
        auto sin_odd  = input_sin[stride * (2 * k + 1)];
        auto cos_odd_term = cos_odd * twiddle_cos - sin_odd * twiddle_sin;

        // Note: the sine transform at these indices is never used, so we skip calculating them
        output_cos[stride * k]                    = cos_even + cos_odd_term;
        output_cos[stride * (block_size - k - 1)] = cos_even - cos_odd_term;

        for (size_t block_idx = 1; block_idx < stride; block_idx++)
        {
            auto cos_even = input_cos[block_idx + stride * (2 * k)];
            auto cos_odd  = input_cos[block_idx + stride * (2 * k + 1)];
            auto sin_even = input_sin[block_idx + stride * (2 * k)];
            auto sin_odd  = input_sin[block_idx + stride * (2 * k + 1)];
            auto sin_odd_term = sin_odd * twiddle_cos + cos_odd * twiddle_sin;
            auto cos_odd_term = cos_odd * twiddle_cos - sin_odd * twiddle_sin;

            output_cos[block_idx + stride * k] = cos_even + cos_odd_term;
            output_sin[block_idx + stride * k] = sin_even + sin_odd_term;
            // Input block has half the size of output block (size N)
            // Extract values at N/2 + k by "reflecting" indices across N/2
            output_cos[block_idx + stride * (block_size - k - 1)] =   cos_even - cos_odd_term;
            output_sin[block_idx + stride * (block_size - k - 1)] = - sin_even + sin_odd_term;
        }
    }
}

std::vector<float> _discrete_cosine_transform_pow_2_type_2(const std::span<const float> input)
{
    size_t trf_size = input.size();
    if (trf_size == 1)
    {
        // Trivial base case
        return std::vector<float>({input[0]});
    }

    const auto [stage_input_sin, stage_input_cos] = _discrete_sine_and_cosine_pow_2_type_2_common_stages(input);
    std::vector<float> output(input.size());
    auto half_size = trf_size / 2;

    // Special handling of k = 0
    auto cos_even_k0 = stage_input_cos[0];
    auto cos_odd_k0  = stage_input_cos[1];
    output[0] = cos_even_k0 + cos_odd_k0;

    for (uint32_t k = 1; k < half_size; k++)
    {
        float angle = M_PIf * float(k) / (2 * trf_size);
        float twiddle_sin, twiddle_cos;
        sincosf(angle, &twiddle_sin, &twiddle_cos);

        auto cos_even = stage_input_cos[2 * k];
        auto cos_odd  = stage_input_cos[2 * k + 1];
        auto sin_even = stage_input_sin[2 * (k - 1)];
        auto sin_odd  = stage_input_sin[2 * (k - 1) + 1];

        auto cos_sum  = cos_even + cos_odd;
        auto sin_diff = sin_even - sin_odd;

        output[k]            =   cos_sum * twiddle_cos + sin_diff * twiddle_sin;
        output[trf_size - k] = - cos_sum * twiddle_sin + sin_diff * twiddle_cos;
    }

    // Separate handling of k = size // 2 because it reflects to itself
    auto sin_even_k_half = stage_input_sin[2 * (half_size - 1)];
    auto sin_odd_k_half  = stage_input_sin[2 * (half_size - 1) + 1];
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
    std::vector<float> buffer_1_cos, buffer_1_sin(trf_size);
    std::vector<float> buffer_2_cos(trf_size), buffer_2_sin(trf_size);
    buffer_1_cos.reserve(input.size());
    for (auto val : input)
    {
        buffer_1_cos.emplace_back(val);
    }
    // Cosine transform type 3 is defined as 0.5 * x[0] + a sum over x[n], n=1..N-1
    // Multiplying x[0] by 0.5 lets us include it as the first term in the sum, simplifying the algorithm
    buffer_1_cos[0] *= 0.5f;
    std::span<float> stage_input_sin(buffer_1_sin);
    std::span<float> stage_input_cos(buffer_1_cos);
    std::span<float> stage_output_sin(buffer_2_sin);
    std::span<float> stage_output_cos(buffer_2_cos);

    for (size_t stage = 0; stage < log2_size - 1; stage++)
    {
        size_t stride = (trf_size >> (stage + 1));
        size_t block_size = 1 << (stage + 1);

        _discrete_cosine_type_3_factor_2(stage_input_sin, stage_input_cos, stage_output_sin, stage_output_cos, stride, block_size);

        std::swap(stage_input_sin, stage_output_sin);
        std::swap(stage_input_cos, stage_output_cos);
    }

    // Final iteration: stage = log2_size - 1
    // `stride` is 1 and only the cosine transform output needs to be calculated
    // This is the "block_idx == 0" step from previous stages
    size_t half_size = trf_size / 2;
    for (size_t k = 0; k < half_size; k++)
    {
        float angle = M_PI * (k + 0.5f) / trf_size;
        float twiddle_sin, twiddle_cos;
        sincosf(angle, &twiddle_sin, &twiddle_cos);

        auto cos_even = stage_input_cos[2 * k];
        auto cos_odd  = stage_input_cos[2 * k + 1];
        auto sin_odd  = stage_input_sin[2 * k + 1];
        auto cos_odd_term = cos_odd * twiddle_cos - sin_odd * twiddle_sin;

        stage_output_cos[k]                = cos_even + cos_odd_term;
        stage_output_cos[trf_size - k - 1] = cos_even - cos_odd_term;
    }

    return stage_output_cos.data() == buffer_1_cos.data() ? buffer_1_cos : buffer_2_cos;
}

std::vector<float> _discrete_sine_transform_pow_2_type_2(const std::span<const float> input)
{
    size_t trf_size = input.size();
    if (trf_size == 1)
    {
        // Trivial base case
        return std::vector<float>({input[0]});
    }

    const auto [stage_input_sin, stage_input_cos] = _discrete_sine_and_cosine_pow_2_type_2_common_stages(input);
    std::vector<float> output(input.size());
    auto half_size = trf_size / 2;

    // Special handling of k = 0
    auto cos_even_k0 = stage_input_cos[0];
    auto cos_odd_k0  = stage_input_cos[1];

    output[trf_size - 1] = cos_even_k0 - cos_odd_k0;

    for (uint32_t k = 1; k < half_size; k++)
    {
        float angle = M_PIf * float(k) / (2 * trf_size);
        float twiddle_sin, twiddle_cos;
        sincosf(angle, &twiddle_sin, &twiddle_cos);

        auto cos_even = stage_input_cos[2 * k];
        auto cos_odd  = stage_input_cos[2 * k + 1];
        auto sin_even = stage_input_sin[2 * (k - 1)];
        auto sin_odd  = stage_input_sin[2 * (k - 1) + 1];

        auto sin_sum  = sin_even + sin_odd;
        auto cos_diff = cos_even - cos_odd;

        output[k - 1]            = sin_sum * twiddle_cos - cos_diff * twiddle_sin;
        output[trf_size - k - 1] = sin_sum * twiddle_sin + cos_diff * twiddle_cos;
    }

    // Separate handling of k = size // 2 because it reflects to itself
    auto sin_even_k_half = stage_input_sin[2 * (half_size - 1)];
    auto sin_odd_k_half  = stage_input_sin[2 * (half_size - 1) + 1];
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
    std::vector<float> buffer_1_sin, buffer_1_cos(trf_size);
    std::vector<float> buffer_2_sin(trf_size), buffer_2_cos(trf_size);
    buffer_1_sin.reserve(trf_size);
    for (auto val : input)
    {
        buffer_1_sin.emplace_back(val);
    }
    // Sine transform type 3 is defined as 0.5 * x[N-1] + a sum over x[n], n=0..N-2
    // Multiplying x[N-1] by 0.5 lets us include it as the last term in the sum, simplifying the algorithm
    buffer_1_sin[trf_size - 1] *= 0.5f;
    std::span<float> stage_input_sin(buffer_1_sin);
    std::span<float> stage_input_cos(buffer_1_cos);
    std::span<float> stage_output_sin(buffer_2_sin);
    std::span<float> stage_output_cos(buffer_2_cos);

    for (size_t stage = 0; stage < log2_size - 1; stage++)
    {
        size_t stride = (trf_size >> (stage + 1));
        size_t block_size = 1 << (stage + 1);

        _discrete_sine_type_3_factor_2(stage_input_sin, stage_input_cos, stage_output_sin, stage_output_cos, stride, block_size);

        std::swap(stage_input_sin, stage_output_sin);
        std::swap(stage_input_cos, stage_output_cos);
    }

    // Final iteration: stage = log2_size - 1
    // `stride` is 1 and only the sine transform output needs to be calculated
    // This is the "block_idx == N-1" step from previous stages
    // block_idx = stride - 1 == 0
    size_t half_size = trf_size / 2;
    for (size_t k = 0; k < half_size; k++)
    {
        float angle = M_PI * (k + 0.5f) / trf_size;
        float twiddle_sin, twiddle_cos;
        sincosf(angle, &twiddle_sin, &twiddle_cos);

        auto sin_odd  = stage_input_sin[2 * k + 1];
        auto sin_even = stage_input_sin[2 * k];
        auto cos_even = stage_input_cos[2 * k];
        auto sin_even_term = sin_even * twiddle_cos - cos_even * twiddle_sin;

        stage_output_sin[k]                =   sin_odd + sin_even_term;
        stage_output_sin[trf_size - k - 1] = - sin_odd + sin_even_term;
    }

    return stage_output_sin.data() == buffer_1_sin.data() ? buffer_1_sin : buffer_2_sin;
}

std::vector<float> _discrete_cosine_transform_multi_radix_type_2(const std::span<const float> input, const std::span<const primes::prime_factor<size_t>> factors)
{
    size_t trf_size = input.size();
    size_t block_size = 1;
    size_t stride = trf_size;

    size_t pow_2, pow_3;

    for (auto [prime, pow] : factors)
    {
        switch (prime)
        {
            case 2:
                pow_2 = pow;
                break;
            case 3:
                pow_3 = pow;
                break;
            default:
                throw std::domain_error(std::format("Unsupported prime factor: {}", std::to_string(prime)));
        }
    }

    // Construct four compute buffers:
    // buffer_1_sin and buffer_1_cos are initialized to equal `input`.
    // buffer_2_sin and buffer_2_cos are default initialized with the same size as input.
    // The computation writes back and forth between buffer_1_[sin|cos] and buffer_2_[sin|cos]
    std::vector<float> buffer_1_sin(input.begin(), input.end());
    std::vector<float> buffer_1_cos(buffer_1_sin);
    std::vector<float> buffer_2_sin(trf_size), buffer_2_cos(trf_size);
    std::span<float> stage_input_sin(buffer_1_sin);
    std::span<float> stage_input_cos(buffer_1_cos);
    std::span<float> stage_output_sin(buffer_2_sin);
    std::span<float> stage_output_cos(buffer_2_cos);

    for (size_t i = 0; i < pow_3; i++)
    {
        block_size *= 3;
        stride /= 3;
        _discrete_sine_and_cosine_type_2_factor_3(stage_input_sin, stage_input_cos, stage_output_sin, stage_output_cos, stride, block_size);
        std::swap(stage_input_sin, stage_output_sin);
        std::swap(stage_input_cos, stage_output_cos);
    }

    for (size_t i = 0; i < pow_2; i++)
    {
        block_size <<= 1;
        stride >>= 1;
        _discrete_sine_and_cosine_type_2_factor_2(stage_input_sin, stage_input_cos, stage_output_sin, stage_output_cos, stride, block_size);
        std::swap(stage_input_sin, stage_output_sin);
        std::swap(stage_input_cos, stage_output_cos);
    }

    assert(block_size == trf_size);
    if ((pow_2 + pow_3) % 2 == 0)
    {
        return buffer_1_cos;
    }
    else
    {
        return buffer_2_cos;
    }
}

std::vector<float> _discrete_sine_transform_multi_radix_type_2(const std::span<const float> input, const std::span<const primes::prime_factor<size_t>> factors)
{
    size_t trf_size = input.size();
    size_t block_size = 1;
    size_t stride = trf_size;

    size_t pow_2, pow_3;

    for (auto [prime, pow] : factors)
    {
        switch (prime)
        {
            case 2:
                pow_2 = pow;
                break;
            case 3:
                pow_3 = pow;
                break;
            default:
                throw std::domain_error(std::format("Unsupported prime factor: {}", std::to_string(prime)));
        }
    }

    // Construct four compute buffers:
    // buffer_1_sin and buffer_1_cos are initialized to equal `input`.
    // buffer_2_sin and buffer_2_cos are default initialized with the same size as input.
    // The computation writes back and forth between buffer_1_[sin|cos] and buffer_2_[sin|cos]
    std::vector<float> buffer_1_sin(input.begin(), input.end());
    std::vector<float> buffer_1_cos(buffer_1_sin);
    std::vector<float> buffer_2_sin(trf_size), buffer_2_cos(trf_size);
    std::span<float> stage_input_sin(buffer_1_sin);
    std::span<float> stage_input_cos(buffer_1_cos);
    std::span<float> stage_output_sin(buffer_2_sin);
    std::span<float> stage_output_cos(buffer_2_cos);

    for (size_t i = 0; i < pow_3; i++)
    {
        block_size *= 3;
        stride /= 3;
        _discrete_sine_and_cosine_type_2_factor_3(stage_input_sin, stage_input_cos, stage_output_sin, stage_output_cos, stride, block_size);
        std::swap(stage_input_sin, stage_output_sin);
        std::swap(stage_input_cos, stage_output_cos);
    }

    for (size_t i = 0; i < pow_2; i++)
    {
        block_size <<= 1;
        stride >>= 1;
        _discrete_sine_and_cosine_type_2_factor_2(stage_input_sin, stage_input_cos, stage_output_sin, stage_output_cos, stride, block_size);
        std::swap(stage_input_sin, stage_output_sin);
        std::swap(stage_input_cos, stage_output_cos);
    }

    assert(block_size == trf_size);
    if ((pow_2 + pow_3) % 2 == 0)
    {
        return buffer_1_sin;
    }
    else
    {
        return buffer_2_sin;
    }
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
    if (factors.size() == 0 || (factors.size() == 1 && factors[0].factor == 2))
    {
        return discrete_cosine_transform_pow_2(input, type);
    }
    if (type == SIN_COS_TRF_TYPE::II)
    {
        return _discrete_cosine_transform_multi_radix_type_2(input, factors);
    }
    throw std::domain_error("Cosine transform type 3 is currently only implemented for power-of-two sized input.");
}

std::vector<float> discrete_sine_transform(const std::span<const float> input, SIN_COS_TRF_TYPE type)
{
    auto factors = primes::get_prime_factors(input.size());
    if (factors.size() == 0 || (factors.size() == 1 && factors[0].factor == 2))
    {
        return discrete_sine_transform_pow_2(input, type);
    }
    if (type == SIN_COS_TRF_TYPE::II)
    {
        return _discrete_sine_transform_multi_radix_type_2(input, factors);
    }
    throw std::domain_error("Sine transform type 3 is currently only implemented for power-of-two sized input.");
}

} // namespace numeric::transforms