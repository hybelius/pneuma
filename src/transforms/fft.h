#include <cstdint>
#include <vector>
#include <span>

namespace pneuma::transforms
{

enum class SIN_COS_TRF_TYPE
{
    I,
    II,
    III,
    IV
};

std::vector<float> discrete_cosine_transform(const std::span<const float> input, SIN_COS_TRF_TYPE type);
std::vector<float> discrete_cosine_transform_pow_2(const std::span<const float> input, SIN_COS_TRF_TYPE type);

std::vector<float> discrete_sine_transform(const std::span<const float> input, SIN_COS_TRF_TYPE type);

} // namespace numeric::transforms
