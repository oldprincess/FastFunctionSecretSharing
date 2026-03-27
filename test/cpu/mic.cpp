// clang-format off
// g++ -I include -I third_party/googletest/googletest/include -I third_party/googletest/googletest src/cpu/config.cpp src/cpu/mic.cpp test/cpu/mic.cpp third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cpu_mic.exe -std=c++17 -maes -fopenmp
// clang-format on
#include <FastFss/cpu/mic.h>
#include <FastFss/errors.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "wideint/wideint.hpp"

namespace {

using uint128_t = wideint::uint<2>;

template <typename T>
constexpr T modBits(T x, std::size_t bitWidth) noexcept
{
    if (bitWidth == sizeof(T) * 8)
    {
        return x;
    }
    return x & ((T(1) << static_cast<unsigned int>(bitWidth)) - T(1));
}

template <typename T>
std::array<std::vector<std::uint8_t>, 2> makeSeeds(std::size_t elementNum)
{
    std::array<std::vector<std::uint8_t>, 2> seeds{
        std::vector<std::uint8_t>(16 * elementNum),
        std::vector<std::uint8_t>(16 * elementNum),
    };
    for (std::size_t i = 0; i < seeds[0].size(); ++i)
    {
        seeds[0][i] = static_cast<std::uint8_t>((31 * i + 3) & 0xff);
        seeds[1][i] = static_cast<std::uint8_t>((17 * i + 9) & 0xff);
    }
    return seeds;
}

template <typename T>
void makeKeyAndCache(std::size_t                      bitWidthIn,
                     std::size_t                      bitWidthOut,
                     std::size_t                      elementNum,
                     const std::vector<T>            &alpha,
                     const std::vector<T>            &leftEndpoints,
                     const std::vector<T>            &rightEndpoints,
                     const std::vector<std::uint8_t> &seed0,
                     const std::vector<std::uint8_t> &seed1,
                     std::vector<std::uint8_t>       &key,
                     std::vector<std::uint8_t>       &cache,
                     std::vector<T>                  &z)
{
    std::size_t keyDataSize = 0;
    ASSERT_EQ(FastFss_cpu_dcfMICGetKeyDataSize(
                  &keyDataSize, bitWidthIn, bitWidthOut, sizeof(T), elementNum),
              FAST_FSS_SUCCESS);
    ASSERT_GT(keyDataSize, 0u);

    std::size_t cacheDataSize = 0;
    ASSERT_EQ(
        FastFss_cpu_dcfMICGetCacheDataSize(&cacheDataSize, bitWidthIn,
                                           bitWidthOut, sizeof(T), elementNum),
        FAST_FSS_SUCCESS);

    key.assign(keyDataSize, 0);
    cache.assign(cacheDataSize, 0);
    z.assign(elementNum * leftEndpoints.size(), 0);

    ASSERT_EQ(FastFss_cpu_dcfMICKeyGen(
                  key.data(), key.size(), z.data(), z.size() * sizeof(T),
                  alpha.data(), alpha.size() * sizeof(T), seed0.data(),
                  seed0.size(), seed1.data(), seed1.size(),
                  leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
                  rightEndpoints.data(), rightEndpoints.size() * sizeof(T),
                  bitWidthIn, bitWidthOut, sizeof(T), elementNum),
              FAST_FSS_SUCCESS);
}

template <typename T>
class MicCpuTypedTest : public ::testing::Test
{
};

using MicElementTypes = ::testing::
    Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, uint128_t>;

TYPED_TEST_SUITE(MicCpuTypedTest, MicElementTypes);

TYPED_TEST(MicCpuTypedTest, IntervalEvaluationMatchesExpectedMembership)
{
    using T = TypeParam;
    constexpr std::size_t bitWidthIn =
        sizeof(T) == 1 ? 8 : (sizeof(T) == 2 ? 12 : (sizeof(T) == 4 ? 24 : 32));
    constexpr std::size_t bitWidthOut    = sizeof(T) == 16 ? 127 : 8;
    const std::vector<T>  leftEndpoints  = {T(1), T(4), T(7)};
    const std::vector<T>  rightEndpoints = {T(2), T(6), T(8)};
    const std::vector<T>  alpha          = {T(5), T(11), T(19), T(2)};
    const std::vector<T>  x              = {T(1), T(5), T(9), T(7)};
    const std::size_t     elementNum     = alpha.size();
    auto                  seeds          = makeSeeds<T>(elementNum);

    std::vector<T> maskedX(elementNum);
    for (std::size_t i = 0; i < elementNum; ++i)
    {
        maskedX[i] = modBits<T>(x[i] + alpha[i], bitWidthIn);
    }

    std::vector<std::uint8_t> key;
    std::vector<std::uint8_t> cache;
    std::vector<T>            z;
    makeKeyAndCache(bitWidthIn, bitWidthOut, elementNum, alpha, leftEndpoints,
                    rightEndpoints, seeds[0], seeds[1], key, cache, z);

    std::vector<T> sharedZ0(z.size());
    std::vector<T> sharedZ1(z.size());
    for (std::size_t i = 0; i < z.size(); ++i)
    {
        sharedZ0[i] = static_cast<T>(i + 1);
        sharedZ1[i] = z[i] - sharedZ0[i];
    }

    std::vector<T> share0(z.size());
    std::vector<T> share1(z.size());
    ASSERT_EQ(
        FastFss_cpu_dcfMICEval(
            share0.data(), share0.size() * sizeof(T), maskedX.data(),
            maskedX.size() * sizeof(T), key.data(), key.size(), sharedZ0.data(),
            sharedZ0.size() * sizeof(T), seeds[0].data(), seeds[0].size(), 0,
            leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
            rightEndpoints.data(), rightEndpoints.size() * sizeof(T),
            bitWidthIn, bitWidthOut, sizeof(T), elementNum, nullptr, 0),
        FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cpu_dcfMICEval(
                  share1.data(), share1.size() * sizeof(T), maskedX.data(),
                  maskedX.size() * sizeof(T), key.data(), key.size(),
                  sharedZ1.data(), sharedZ1.size() * sizeof(T), seeds[1].data(),
                  seeds[1].size(), 1, leftEndpoints.data(),
                  leftEndpoints.size() * sizeof(T), rightEndpoints.data(),
                  rightEndpoints.size() * sizeof(T), bitWidthIn, bitWidthOut,
                  sizeof(T), elementNum, cache.data(), cache.size()),
              FAST_FSS_SUCCESS);

    for (std::size_t i = 0; i < elementNum; ++i)
    {
        for (std::size_t j = 0; j < leftEndpoints.size(); ++j)
        {
            const T reconstructed =
                modBits<T>(share0[i * leftEndpoints.size() + j] +
                               share1[i * leftEndpoints.size() + j],
                           bitWidthOut);
            const bool inRange =
                leftEndpoints[j] <= x[i] && x[i] <= rightEndpoints[j];
            EXPECT_EQ(reconstructed, inRange ? T(1) : T(0));
        }
    }
}

} // namespace
