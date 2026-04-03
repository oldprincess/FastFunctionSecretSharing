#include <FastFss/cpu/grotto.h>
#include <FastFss/errors.h>
#include <FastFss/grotto.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/test_utils.hpp"

namespace FastFss::tests {
namespace {

TEST(CpuGrottoIntervalLutEvalTest, IntervalLutEvalReconstructsLookupValue)
{
    using T                             = std::uint8_t;
    constexpr std::size_t bitWidthIn    = 7;
    constexpr std::size_t bitWidthOut   = 8;
    constexpr std::size_t elementNum    = 17;
    constexpr std::size_t intervalCount = 3;
    constexpr std::size_t lutNum        = 2;

    auto           seeds = make_seeds<T>(elementNum, 43, 1, 47, 9);
    auto           alpha = random_vector<T>(elementNum, bitWidthIn);
    auto           x     = random_vector<T>(elementNum, bitWidthIn);
    std::vector<T> maskedX(elementNum);
    for (std::size_t i = 0; i < elementNum; ++i)
    {
        maskedX[i] = mod_bits<T>(alpha[i] + x[i], bitWidthIn);
    }

    std::vector<T> left  = {5, 17, 42};
    std::vector<T> right = {12, 33, 63};
    std::vector<T> table = {11, 23, 37, 3, 5, 7};

    std::size_t keySize = 0;
    ASSERT_EQ(FastFss_grottoGetKeyDataSize(&keySize, bitWidthIn, sizeof(T), elementNum), FAST_FSS_SUCCESS);
    std::size_t cacheSize = 0;
    ASSERT_EQ(FastFss_grottoGetCacheDataSize(&cacheSize, bitWidthIn, sizeof(T), elementNum), FAST_FSS_SUCCESS);
    std::vector<std::uint8_t> key(keySize);
    ASSERT_EQ(
        FastFss_cpu_grottoKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), seeds[0].data(),
                                 seeds[0].size(), seeds[1].data(), seeds[1].size(), bitWidthIn, sizeof(T), elementNum),
        FAST_FSS_SUCCESS);

    std::vector<T>            shareE0(elementNum), shareE1(elementNum);
    std::vector<T>            shareT0(elementNum * lutNum), shareT1(elementNum * lutNum);
    std::vector<std::uint8_t> cache(cacheSize);
    ASSERT_EQ(FastFss_cpu_grottoIntervalLutEval(
                  shareE0.data(), shareE0.size() * sizeof(T), shareT0.data(), shareT0.size() * sizeof(T),
                  maskedX.data(), maskedX.size() * sizeof(T), key.data(), key.size(), seeds[0].data(), seeds[0].size(),
                  0, left.data(), left.size() * sizeof(T), right.data(), right.size() * sizeof(T), table.data(),
                  table.size() * sizeof(T), bitWidthIn, bitWidthOut, sizeof(T), elementNum, cache.data(), cache.size()),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cpu_grottoIntervalLutEval(
                  shareE1.data(), shareE1.size() * sizeof(T), shareT1.data(), shareT1.size() * sizeof(T),
                  maskedX.data(), maskedX.size() * sizeof(T), key.data(), key.size(), seeds[1].data(), seeds[1].size(),
                  1, left.data(), left.size() * sizeof(T), right.data(), right.size() * sizeof(T), table.data(),
                  table.size() * sizeof(T), bitWidthIn, bitWidthOut, sizeof(T), elementNum, nullptr, 0),
              FAST_FSS_SUCCESS);

    for (std::size_t i = 0; i < elementNum; ++i)
    {
        std::size_t matched = intervalCount;
        for (std::size_t k = 0; k < intervalCount; ++k)
        {
            if (left[k] <= x[i] && x[i] <= right[k])
            {
                matched = k;
                break;
            }
        }
        for (std::size_t j = 0; j < lutNum; ++j)
        {
            const T e        = mod_bits<T>(shareE0[i] + shareE1[i], 1);
            const T t        = mod_bits<T>(shareT0[i * lutNum + j] + shareT1[i * lutNum + j], bitWidthOut);
            const T out      = mod_bits<T>(t + e * t * T(-2), bitWidthOut);
            const T expected = matched == intervalCount ? T(0) : table[matched + j * intervalCount];
            EXPECT_EQ(out, expected);
        }
    }
}

} // namespace
} // namespace FastFss::tests
