// clang-format off
// g++ -I include -I third_party/googletest/googletest/include -I third_party/googletest/googletest src/cpu/config.cpp src/cpu/grotto.cpp test/cpu/grotto.cpp third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cpu_grotto_gtest.exe -std=c++17 -maes -fopenmp
// clang-format on
#include <FastFss/cpu/grotto.h>
#include <FastFss/errors.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "uint128_t.h"

namespace {

template <typename T>
constexpr T modBits(T x, std::size_t bitWidth) noexcept
{
    if (bitWidth == sizeof(T) * 8)
    {
        return x;
    }
    return x & ((((T)1) << bitWidth) - 1);
}

template <typename T>
constexpr std::size_t defaultBitWidthIn() noexcept
{
    if constexpr (sizeof(T) == 1)
    {
        return 8;
    }
    else if constexpr (sizeof(T) == 2)
    {
        return 12;
    }
    else if constexpr (sizeof(T) == 4)
    {
        return 18;
    }
    else if constexpr (sizeof(T) == 8)
    {
        return 32;
    }
    else
    {
        return 64;
    }
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
        seeds[0][i] = static_cast<std::uint8_t>((11 * i + 5) & 0xff);
        seeds[1][i] = static_cast<std::uint8_t>((19 * i + 9) & 0xff);
    }
    return seeds;
}

template <typename T>
std::vector<T> reconstructBooleanShares(const std::vector<T> &share0,
                                        const std::vector<T> &share1)
{
    EXPECT_EQ(share0.size(), share1.size());
    std::vector<T> result(share0.size());
    for (std::size_t i = 0; i < share0.size(); ++i)
    {
        result[i] = (share0[i] + share1[i]) & 1;
    }
    return result;
}

template <typename T>
void makeKeyAndCache(std::size_t                      bitWidthIn,
                     std::size_t                      elementNum,
                     const std::vector<T>            &alpha,
                     const std::vector<std::uint8_t> &seed0,
                     const std::vector<std::uint8_t> &seed1,
                     std::vector<std::uint8_t>       &key,
                     std::vector<std::uint8_t>       &cache)
{
    std::size_t keyDataSize = 0;
    ASSERT_EQ(FastFss_cpu_grottoGetKeyDataSize(&keyDataSize, bitWidthIn,
                                               sizeof(T), elementNum),
              FAST_FSS_SUCCESS);
    ASSERT_GT(keyDataSize, 0u);

    std::size_t cacheDataSize = 0;
    ASSERT_EQ(FastFss_cpu_grottoGetCacheDataSize(&cacheDataSize, bitWidthIn,
                                                 sizeof(T), elementNum),
              FAST_FSS_SUCCESS);

    key.assign(keyDataSize, 0);
    cache.assign(cacheDataSize, 0);

    ASSERT_EQ(FastFss_cpu_grottoKeyGen(key.data(), key.size(), alpha.data(),
                                       alpha.size() * sizeof(T), seed0.data(),
                                       seed0.size(), seed1.data(), seed1.size(),
                                       bitWidthIn, sizeof(T), elementNum),
              FAST_FSS_SUCCESS);
}

template <typename T>
class GrottoCpuTypedTest : public ::testing::Test
{
};

using GrottoElementTypes = ::testing::
    Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, uint128_t>;

TYPED_TEST_SUITE(GrottoCpuTypedTest, GrottoElementTypes);

TYPED_TEST(GrottoCpuTypedTest, EqEvalReconstructsZeroTest)
{
    using T                         = TypeParam;
    const std::size_t    bitWidthIn = defaultBitWidthIn<T>();
    const std::vector<T> alpha      = {T(5), T(9), T(3), T(12)};
    const std::vector<T> x          = {T(0), T(4), T(0), T(7)};
    const std::size_t    elementNum = alpha.size();
    auto                 seeds      = makeSeeds<T>(elementNum);

    std::vector<T> maskedX(elementNum);
    for (std::size_t i = 0; i < elementNum; ++i)
    {
        maskedX[i] = modBits<T>(alpha[i] + x[i], bitWidthIn);
    }

    std::vector<std::uint8_t> key;
    std::vector<std::uint8_t> cache;
    makeKeyAndCache(bitWidthIn, elementNum, alpha, seeds[0], seeds[1], key,
                    cache);

    std::vector<T>            share0(elementNum);
    std::vector<T>            share1(elementNum);
    std::vector<std::uint8_t> cache1 = cache;

    ASSERT_EQ(FastFss_cpu_grottoEqEval(
                  share0.data(), share0.size() * sizeof(T), maskedX.data(),
                  maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[0].data(), seeds[0].size(), 0, bitWidthIn, sizeof(T),
                  elementNum, cache.data(), cache.size()),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cpu_grottoEqEval(
                  share1.data(), share1.size() * sizeof(T), maskedX.data(),
                  maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[1].data(), seeds[1].size(), 1, bitWidthIn, sizeof(T),
                  elementNum, cache1.data(), cache1.size()),
              FAST_FSS_SUCCESS);

    const auto reconstructed = reconstructBooleanShares(share0, share1);
    EXPECT_EQ(reconstructed[0], T(1));
    EXPECT_EQ(reconstructed[1], T(0));
    EXPECT_EQ(reconstructed[2], T(1));
    EXPECT_EQ(reconstructed[3], T(0));
}

TYPED_TEST(GrottoCpuTypedTest, EvalMatchesStrictAndClosedBounds)
{
    using T                         = TypeParam;
    const std::size_t    bitWidthIn = defaultBitWidthIn<T>();
    const std::vector<T> alpha      = {T(3), T(9), T(15), T(1)};
    const std::vector<T> x          = {T(1), T(0), T(5), T(2)};
    const std::size_t    elementNum = alpha.size();
    auto                 seeds      = makeSeeds<T>(elementNum);

    std::vector<T> maskedX(elementNum);
    std::vector<T> expectedStrict(elementNum);
    std::vector<T> expectedClosed(elementNum);
    for (std::size_t i = 0; i < elementNum; ++i)
    {
        maskedX[i]        = modBits<T>(alpha[i] + x[i], bitWidthIn);
        expectedStrict[i] = (maskedX[i] < alpha[i]) ? T(1) : T(0);
        expectedClosed[i] = (maskedX[i] <= alpha[i]) ? T(1) : T(0);
    }

    std::vector<std::uint8_t> key;
    std::vector<std::uint8_t> cache;
    makeKeyAndCache(bitWidthIn, elementNum, alpha, seeds[0], seeds[1], key,
                    cache);

    std::vector<T> share0(elementNum), share1(elementNum);
    std::vector<T> share0eq(elementNum), share1eq(elementNum);

    ASSERT_EQ(FastFss_cpu_grottoEval(
                  share0.data(), share0.size() * sizeof(T), maskedX.data(),
                  maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[0].data(), seeds[0].size(), false, 0, bitWidthIn,
                  sizeof(T), elementNum, cache.data(), cache.size()),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cpu_grottoEval(share1.data(), share1.size() * sizeof(T),
                                     maskedX.data(), maskedX.size() * sizeof(T),
                                     key.data(), key.size(), seeds[1].data(),
                                     seeds[1].size(), false, 1, bitWidthIn,
                                     sizeof(T), elementNum, nullptr, 0),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cpu_grottoEval(
                  share0eq.data(), share0eq.size() * sizeof(T), maskedX.data(),
                  maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[0].data(), seeds[0].size(), true, 0, bitWidthIn,
                  sizeof(T), elementNum, nullptr, 0),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cpu_grottoEval(
                  share1eq.data(), share1eq.size() * sizeof(T), maskedX.data(),
                  maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[1].data(), seeds[1].size(), true, 1, bitWidthIn,
                  sizeof(T), elementNum, nullptr, 0),
              FAST_FSS_SUCCESS);

    EXPECT_EQ(reconstructBooleanShares(share0, share1), expectedStrict);
    EXPECT_EQ(reconstructBooleanShares(share0eq, share1eq), expectedClosed);
}

TYPED_TEST(GrottoCpuTypedTest, EqEvalMultiMatchesSelectedPoints)
{
    using T                         = TypeParam;
    const std::size_t    bitWidthIn = defaultBitWidthIn<T>();
    const std::vector<T> alpha      = {T(8), T(2), T(11)};
    const std::vector<T> x          = {T(1), T(4), T(7)};
    const std::vector<T> points     = {T(7), T(1), T(3), T(4)};
    const std::size_t    elementNum = alpha.size();
    auto                 seeds      = makeSeeds<T>(elementNum);

    std::vector<T> maskedX(elementNum);
    for (std::size_t i = 0; i < elementNum; ++i)
    {
        maskedX[i] = modBits<T>(alpha[i] + x[i], bitWidthIn);
    }

    std::vector<std::uint8_t> key;
    std::vector<std::uint8_t> cache;
    makeKeyAndCache(bitWidthIn, elementNum, alpha, seeds[0], seeds[1], key,
                    cache);

    std::vector<T> share0(elementNum * points.size());
    std::vector<T> share1(elementNum * points.size());

    ASSERT_EQ(
        FastFss_cpu_grottoEqEvalMulti(
            share0.data(), share0.size() * sizeof(T), maskedX.data(),
            maskedX.size() * sizeof(T), key.data(), key.size(), seeds[0].data(),
            seeds[0].size(), 0, points.data(), points.size() * sizeof(T),
            bitWidthIn, sizeof(T), elementNum, cache.data(), cache.size()),
        FAST_FSS_SUCCESS);
    ASSERT_EQ(
        FastFss_cpu_grottoEqEvalMulti(
            share1.data(), share1.size() * sizeof(T), maskedX.data(),
            maskedX.size() * sizeof(T), key.data(), key.size(), seeds[1].data(),
            seeds[1].size(), 1, points.data(), points.size() * sizeof(T),
            bitWidthIn, sizeof(T), elementNum, nullptr, 0),
        FAST_FSS_SUCCESS);

    const auto reconstructed = reconstructBooleanShares(share0, share1);
    for (std::size_t i = 0; i < elementNum; ++i)
    {
        for (std::size_t j = 0; j < points.size(); ++j)
        {
            EXPECT_EQ(reconstructed[i * points.size() + j],
                      x[i] == points[j] ? T(1) : T(0));
        }
    }
}

TEST(GrottoCpuTest, MICEvalMatchesIntervals)
{
    using T                          = std::uint16_t;
    constexpr std::size_t bitWidthIn = 12;
    const std::vector<T>  alpha      = {10, 20, 30};
    const std::vector<T>  x          = {5, 23, 35};
    const std::vector<T>  left       = {1, 20, 30};
    const std::vector<T>  right      = {10, 25, 34};
    const std::size_t     elementNum = alpha.size();
    auto                  seeds      = makeSeeds<T>(elementNum);

    std::vector<T> maskedX(elementNum);
    for (std::size_t i = 0; i < elementNum; ++i)
    {
        maskedX[i] = modBits<T>(alpha[i] + x[i], bitWidthIn);
    }

    std::vector<std::uint8_t> key;
    std::vector<std::uint8_t> cache;
    makeKeyAndCache(bitWidthIn, elementNum, alpha, seeds[0], seeds[1], key,
                    cache);

    std::vector<T> share0(elementNum * left.size());
    std::vector<T> share1(elementNum * left.size());

    ASSERT_EQ(
        FastFss_cpu_grottoMICEval(
            share0.data(), share0.size() * sizeof(T), maskedX.data(),
            maskedX.size() * sizeof(T), key.data(), key.size(), seeds[0].data(),
            seeds[0].size(), 0, left.data(), left.size() * sizeof(T),
            right.data(), right.size() * sizeof(T), bitWidthIn, sizeof(T),
            elementNum, cache.data(), cache.size()),
        FAST_FSS_SUCCESS);
    ASSERT_EQ(
        FastFss_cpu_grottoMICEval(
            share1.data(), share1.size() * sizeof(T), maskedX.data(),
            maskedX.size() * sizeof(T), key.data(), key.size(), seeds[1].data(),
            seeds[1].size(), 1, left.data(), left.size() * sizeof(T),
            right.data(), right.size() * sizeof(T), bitWidthIn, sizeof(T),
            elementNum, nullptr, 0),
        FAST_FSS_SUCCESS);

    const auto reconstructed = reconstructBooleanShares(share0, share1);
    for (std::size_t i = 0; i < elementNum; ++i)
    {
        for (std::size_t j = 0; j < left.size(); ++j)
        {
            const bool inRange = left[j] <= x[i] && x[i] <= right[j];
            EXPECT_EQ(reconstructed[i * left.size() + j],
                      inRange ? T(1) : T(0));
        }
    }
}

TEST(GrottoCpuTest, LutEvalReconstructsLookupValue)
{
    using T                          = std::uint8_t;
    constexpr std::size_t bitWidthIn = 7;
    const std::size_t     tableSize  = 1ULL << bitWidthIn;
    const std::vector<T>  alpha      = {3, 17, 9, 12};
    const std::vector<T>  x          = {0, 5, 19, 63};
    const std::size_t     elementNum = alpha.size();
    auto                  seeds      = makeSeeds<T>(elementNum);

    std::vector<T> table(tableSize);
    for (std::size_t i = 0; i < tableSize; ++i)
    {
        table[i] = static_cast<T>(tableSize - i - 1);
    }

    std::vector<T> maskedX(elementNum);
    for (std::size_t i = 0; i < elementNum; ++i)
    {
        maskedX[i] = modBits<T>(alpha[i] + x[i], bitWidthIn);
    }

    std::vector<std::uint8_t> key;
    std::vector<std::uint8_t> cache;
    makeKeyAndCache(bitWidthIn, elementNum, alpha, seeds[0], seeds[1], key,
                    cache);

    std::vector<T> shareE0(elementNum), shareE1(elementNum);
    std::vector<T> shareT0(elementNum), shareT1(elementNum);

    ASSERT_EQ(FastFss_cpu_grottoLutEval(
                  shareE0.data(), shareE0.size() * sizeof(T), shareT0.data(),
                  shareT0.size() * sizeof(T), maskedX.data(),
                  maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[0].data(), seeds[0].size(), 0, table.data(),
                  table.size() * sizeof(T), bitWidthIn, sizeof(T) * 8,
                  sizeof(T), elementNum, cache.data(), cache.size()),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(
        FastFss_cpu_grottoLutEval(
            shareE1.data(), shareE1.size() * sizeof(T), shareT1.data(),
            shareT1.size() * sizeof(T), maskedX.data(),
            maskedX.size() * sizeof(T), key.data(), key.size(), seeds[1].data(),
            seeds[1].size(), 1, table.data(), table.size() * sizeof(T),
            bitWidthIn, sizeof(T) * 8, sizeof(T), elementNum, nullptr, 0),
        FAST_FSS_SUCCESS);

    for (std::size_t i = 0; i < elementNum; ++i)
    {
        T e = (shareE0[i] + shareE1[i]) & 1;
        T t = shareT0[i] + shareT1[i];
        T v = modBits<T>(t + e * t * (T)(-2), bitWidthIn);
        EXPECT_EQ(v, table[x[i]]);
    }
}

TEST(GrottoCpuTest, SizeQueriesValidateArguments)
{
    std::size_t keyDataSize   = 0;
    std::size_t cacheDataSize = 0;

    EXPECT_EQ(FastFss_cpu_grottoGetKeyDataSize(&keyDataSize, 8,
                                               sizeof(std::uint16_t), 4),
              FAST_FSS_SUCCESS);
    EXPECT_GT(keyDataSize, 0u);

    EXPECT_EQ(FastFss_cpu_grottoGetCacheDataSize(&cacheDataSize, 8,
                                                 sizeof(std::uint16_t), 4),
              FAST_FSS_SUCCESS);
    EXPECT_GT(cacheDataSize, 0u);

    EXPECT_EQ(FastFss_cpu_grottoGetKeyDataSize(&keyDataSize, 5,
                                               sizeof(std::uint16_t), 4),
              FAST_FSS_INVALID_BITWIDTH_ERROR);
}

} // namespace
