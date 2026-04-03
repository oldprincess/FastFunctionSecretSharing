#ifndef FAST_FSS_TESTS_COMMON_TEST_UTILS_HPP
#define FAST_FSS_TESTS_COMMON_TEST_UTILS_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace FastFss::tests {

struct WidthConfig
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

inline std::mt19937_64 &rng()
{
    static std::mt19937_64 engine(0x5A17C3D49E8B102Full);
    return engine;
}

template <typename T>
constexpr std::size_t bit_size_v = sizeof(T) * 8;

template <typename T>
constexpr T mod_bits(T value, std::size_t bitWidth) noexcept
{
    if (bitWidth >= bit_size_v<T>)
    {
        return value;
    }
    return value & ((T(1) << static_cast<unsigned int>(bitWidth)) - T(1));
}

template <typename T>
T random_value(std::size_t bitWidth)
{
    std::uint64_t low  = rng()();
    std::uint64_t high = rng()();
    T             out  = T(low);
    if constexpr (sizeof(T) > sizeof(std::uint64_t))
    {
        out = out | (T(high) << 64U);
    }
    if constexpr (sizeof(T) > sizeof(std::uint64_t) * 2)
    {
        out = out | (T(rng()()) << 128U);
    }
    if constexpr (sizeof(T) > sizeof(std::uint64_t) * 3)
    {
        out = out | (T(rng()()) << 192U);
    }
    return mod_bits(out, bitWidth);
}

template <typename T>
std::vector<T> random_vector(std::size_t size, std::size_t bitWidth)
{
    std::vector<T> values(size);
    for (auto &value : values)
    {
        value = random_value<T>(bitWidth);
    }
    return values;
}

inline std::array<std::size_t, 3> element_counts()
{
    return {1, 1024, 17};
}

template <typename T>
std::vector<WidthConfig> eval_width_configs()
{
    const std::size_t maxBits  = bit_size_v<T>;
    const std::size_t nonFull  = maxBits > 1 ? maxBits - 1 : 1;
    const std::size_t smallIn  = std::max<std::size_t>(1, maxBits / 2);
    const std::size_t smallOut = std::min<std::size_t>(maxBits, std::max<std::size_t>(smallIn + 1, 2));
    const std::size_t largeOut = std::max<std::size_t>(1, nonFull > 2 ? nonFull - 2 : 1);
    return {
        {maxBits, maxBits},
        {nonFull, largeOut},
        {smallIn, smallOut},
        {nonFull, nonFull},
    };
}

template <typename T>
std::array<std::vector<std::uint8_t>, 2> make_seeds(std::size_t  elementNum,
                                                    std::uint8_t mul0,
                                                    std::uint8_t add0,
                                                    std::uint8_t mul1,
                                                    std::uint8_t add1)
{
    std::array<std::vector<std::uint8_t>, 2> seeds{
        std::vector<std::uint8_t>(16 * elementNum),
        std::vector<std::uint8_t>(16 * elementNum),
    };
    for (std::size_t i = 0; i < seeds[0].size(); ++i)
    {
        seeds[0][i] = static_cast<std::uint8_t>((mul0 * i + add0) & 0xffU);
        seeds[1][i] = static_cast<std::uint8_t>((mul1 * i + add1) & 0xffU);
    }
    return seeds;
}

template <typename T>
std::vector<T> reconstruct_arithmetic(const std::vector<T> &share0, const std::vector<T> &share1, std::size_t bitWidth)
{
    std::vector<T> out(share0.size());
    for (std::size_t i = 0; i < share0.size(); ++i)
    {
        out[i] = mod_bits<T>(share0[i] + share1[i], bitWidth);
    }
    return out;
}

template <typename T>
std::vector<T> reconstruct_boolean(const std::vector<T> &share0, const std::vector<T> &share1)
{
    std::vector<T> out(share0.size());
    for (std::size_t i = 0; i < share0.size(); ++i)
    {
        out[i] = (share0[i] + share1[i]) & T(1);
    }
    return out;
}

} // namespace FastFss::tests

#endif
