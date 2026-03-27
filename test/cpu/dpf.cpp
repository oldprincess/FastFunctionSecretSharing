// clang-format off
// g++ -I include -I third_party/googletest/googletest/include -I third_party/googletest/googletest src/cpu/config.cpp src/cpu/dpf.cpp test/cpu/dpf.cpp third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cpu_dpf.exe -std=c++17 -maes -fopenmp
// clang-format on
#include <FastFss/cpu/dpf.h>
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
        seeds[0][i] = static_cast<std::uint8_t>((13 * i + 1) & 0xff);
        seeds[1][i] = static_cast<std::uint8_t>((23 * i + 7) & 0xff);
    }
    return seeds;
}

template <typename T>
std::vector<T> reconstructShares(const std::vector<T> &share0,
                                 const std::vector<T> &share1,
                                 std::size_t           bitWidthOut)
{
    EXPECT_EQ(share0.size(), share1.size());

    std::vector<T> result(share0.size());
    for (std::size_t i = 0; i < share0.size(); ++i)
    {
        result[i] = modBits<T>(share0[i] + share1[i], bitWidthOut);
    }
    return result;
}

template <typename T>
struct DpfFixtureData
{
    std::size_t    bitWidthIn  = 4;
    std::size_t    bitWidthOut = 5;
    std::size_t    groupSize   = 3;
    std::size_t    elementNum  = 4;
    std::vector<T> alpha;
    std::vector<T> x;
    std::vector<T> beta;
    std::vector<T> maskedX;
};

template <typename T>
DpfFixtureData<T> makeFixtureData()
{
    DpfFixtureData<T> data;
    data.alpha = {T(3), T(9), T(5), T(1)};
    data.x     = {T(0), T(2), T(0), T(7)};
    data.beta  = {
        T(1), T(2),  T(3),  T(5),  T(6),  T(7),
        T(9), T(10), T(11), T(13), T(14), T(15),
    };
    data.maskedX.resize(data.elementNum);
    for (std::size_t i = 0; i < data.elementNum; ++i)
    {
        data.maskedX[i] =
            modBits<T>(data.alpha[i] + data.x[i], data.bitWidthIn);
    }
    return data;
}

template <typename T>
void makeKeyAndCache(const DpfFixtureData<T>         &data,
                     const std::vector<std::uint8_t> &seed0,
                     const std::vector<std::uint8_t> &seed1,
                     std::vector<std::uint8_t>       &key,
                     std::vector<std::uint8_t>       &cache)
{
    std::size_t keyDataSize = 0;
    ASSERT_EQ(FastFss_cpu_dpfGetKeyDataSize(&keyDataSize, data.bitWidthIn,
                                            data.bitWidthOut, data.groupSize,
                                            sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);
    ASSERT_GT(keyDataSize, 0u);

    std::size_t cacheDataSize = 0;
    ASSERT_EQ(FastFss_cpu_dpfGetCacheDataSize(&cacheDataSize, data.bitWidthIn,
                                              sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);

    key.assign(keyDataSize, 0);
    cache.assign(cacheDataSize, 0);

    ASSERT_EQ(FastFss_cpu_dpfKeyGen(
                  key.data(), key.size(), data.alpha.data(),
                  data.alpha.size() * sizeof(T), data.beta.data(),
                  data.beta.size() * sizeof(T), seed0.data(), seed0.size(),
                  seed1.data(), seed1.size(), data.bitWidthIn, data.bitWidthOut,
                  data.groupSize, sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);
}

template <typename T>
class DpfCpuTypedTest : public ::testing::Test
{
};

using DpfElementTypes = ::testing::
    Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, uint128_t>;

TYPED_TEST_SUITE(DpfCpuTypedTest, DpfElementTypes);

TYPED_TEST(DpfCpuTypedTest, EvalReturnsBetaOnlyAtZeroOffset)
{
    using T          = TypeParam;
    const auto data  = makeFixtureData<T>();
    auto       seeds = makeSeeds<T>(data.elementNum);

    std::vector<std::uint8_t> key;
    std::vector<std::uint8_t> cache;
    makeKeyAndCache(data, seeds[0], seeds[1], key, cache);

    std::vector<T> share0(data.elementNum * data.groupSize);
    std::vector<T> share1(data.elementNum * data.groupSize);
    std::vector<T> expected(data.elementNum * data.groupSize, T(0));

    for (std::size_t i = 0; i < data.elementNum; ++i)
    {
        if (data.x[i] == T(0))
        {
            for (std::size_t j = 0; j < data.groupSize; ++j)
            {
                expected[i * data.groupSize + j] = modBits<T>(
                    data.beta[i * data.groupSize + j], data.bitWidthOut);
            }
        }
    }

    ASSERT_EQ(FastFss_cpu_dpfEval(
                  share0.data(), share0.size() * sizeof(T), data.maskedX.data(),
                  data.maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[0].data(), seeds[0].size(), 0, data.bitWidthIn,
                  data.bitWidthOut, data.groupSize, sizeof(T), data.elementNum,
                  cache.data(), cache.size()),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cpu_dpfEval(
                  share1.data(), share1.size() * sizeof(T), data.maskedX.data(),
                  data.maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[1].data(), seeds[1].size(), 1, data.bitWidthIn,
                  data.bitWidthOut, data.groupSize, sizeof(T), data.elementNum,
                  nullptr, 0),
              FAST_FSS_SUCCESS);

    EXPECT_EQ(reconstructShares(share0, share1, data.bitWidthOut), expected);
}

TYPED_TEST(DpfCpuTypedTest, EvalAllPlacesBetaAtSelectedPoint)
{
    using T          = TypeParam;
    const auto data  = makeFixtureData<T>();
    auto       seeds = makeSeeds<T>(data.elementNum);

    std::vector<std::uint8_t> key;
    std::vector<std::uint8_t> cache;
    makeKeyAndCache(data, seeds[0], seeds[1], key, cache);

    const std::size_t domainSize = std::size_t{1} << data.bitWidthIn;
    std::vector<T>    share0(data.elementNum * domainSize * data.groupSize);
    std::vector<T>    share1(data.elementNum * domainSize * data.groupSize);

    ASSERT_EQ(FastFss_cpu_dpfEvalAll(
                  share0.data(), share0.size() * sizeof(T), data.maskedX.data(),
                  data.maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[0].data(), seeds[0].size(), 0, data.bitWidthIn,
                  data.bitWidthOut, data.groupSize, sizeof(T), data.elementNum,
                  cache.data(), cache.size()),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cpu_dpfEvalAll(
                  share1.data(), share1.size() * sizeof(T), data.maskedX.data(),
                  data.maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[1].data(), seeds[1].size(), 1, data.bitWidthIn,
                  data.bitWidthOut, data.groupSize, sizeof(T), data.elementNum,
                  nullptr, 0),
              FAST_FSS_SUCCESS);

    const auto reconstructed =
        reconstructShares(share0, share1, data.bitWidthOut);
    for (std::size_t i = 0; i < data.elementNum; ++i)
    {
        for (std::size_t point = 0; point < domainSize; ++point)
        {
            for (std::size_t j = 0; j < data.groupSize; ++j)
            {
                const std::size_t index = i * domainSize * data.groupSize +
                                          point * data.groupSize + j;
                const T expected =
                    point == static_cast<std::size_t>(data.x[i])
                        ? modBits<T>(data.beta[i * data.groupSize + j],
                                     data.bitWidthOut)
                        : T(0);
                EXPECT_EQ(reconstructed[index], expected);
            }
        }
    }
}

TYPED_TEST(DpfCpuTypedTest, EvalMultiMatchesChosenPoints)
{
    using T          = TypeParam;
    const auto data  = makeFixtureData<T>();
    auto       seeds = makeSeeds<T>(data.elementNum);

    std::vector<std::uint8_t> key;
    std::vector<std::uint8_t> cache;
    makeKeyAndCache(data, seeds[0], seeds[1], key, cache);

    const std::vector<T> points = {T(0), T(2), T(4), T(7)};
    std::vector<T> share0(data.elementNum * points.size() * data.groupSize);
    std::vector<T> share1(data.elementNum * points.size() * data.groupSize);

    ASSERT_EQ(FastFss_cpu_dpfEvalMulti(
                  share0.data(), share0.size() * sizeof(T), data.maskedX.data(),
                  data.maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[0].data(), seeds[0].size(), 0, points.data(),
                  points.size() * sizeof(T), data.bitWidthIn, data.bitWidthOut,
                  data.groupSize, sizeof(T), data.elementNum, cache.data(),
                  cache.size()),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cpu_dpfEvalMulti(
                  share1.data(), share1.size() * sizeof(T), data.maskedX.data(),
                  data.maskedX.size() * sizeof(T), key.data(), key.size(),
                  seeds[1].data(), seeds[1].size(), 1, points.data(),
                  points.size() * sizeof(T), data.bitWidthIn, data.bitWidthOut,
                  data.groupSize, sizeof(T), data.elementNum, nullptr, 0),
              FAST_FSS_SUCCESS);

    const auto reconstructed =
        reconstructShares(share0, share1, data.bitWidthOut);
    for (std::size_t i = 0; i < data.elementNum; ++i)
    {
        for (std::size_t pointIndex = 0; pointIndex < points.size();
             ++pointIndex)
        {
            for (std::size_t j = 0; j < data.groupSize; ++j)
            {
                const std::size_t index = i * points.size() * data.groupSize +
                                          pointIndex * data.groupSize + j;
                const T expected =
                    data.x[i] == points[pointIndex]
                        ? modBits<T>(data.beta[i * data.groupSize + j],
                                     data.bitWidthOut)
                        : T(0);
                EXPECT_EQ(reconstructed[index], expected);
            }
        }
    }
}

TYPED_TEST(DpfCpuTypedTest, SizeQueriesReturnSuccessForValidParameters)
{
    std::size_t keyDataSize   = 0;
    std::size_t cacheDataSize = 0;

    EXPECT_EQ(FastFss_cpu_dpfGetKeyDataSize(&keyDataSize, 4, 5, 3,
                                            sizeof(TypeParam), 4),
              FAST_FSS_SUCCESS);
    EXPECT_GT(keyDataSize, 0u);

    EXPECT_EQ(FastFss_cpu_dpfGetCacheDataSize(&cacheDataSize, 4,
                                              sizeof(TypeParam), 4),
              FAST_FSS_SUCCESS);
    EXPECT_GT(cacheDataSize, 0u);
}

TEST(DpfCpuTest, InvalidBitWidthReturnsError)
{
    std::size_t keyDataSize = 0;

    EXPECT_EQ(FastFss_cpu_dpfGetKeyDataSize(&keyDataSize, 17, 8, 1,
                                            sizeof(std::uint16_t), 1),
              FAST_FSS_INVALID_BITWIDTH_ERROR);
    EXPECT_EQ(FastFss_cpu_dpfGetCacheDataSize(&keyDataSize, 9,
                                              sizeof(std::uint8_t), 1),
              FAST_FSS_INVALID_BITWIDTH_ERROR);
}

TEST(DpfCpuTest, InvalidElementSizeReturnsError)
{
    std::size_t keyDataSize = 0;

    EXPECT_EQ(FastFss_cpu_dpfGetKeyDataSize(&keyDataSize, 4, 5, 1, 3, 1),
              FAST_FSS_INVALID_ELEMENT_SIZE_ERROR);
    EXPECT_EQ(FastFss_cpu_dpfGetCacheDataSize(&keyDataSize, 4, 3, 1),
              FAST_FSS_INVALID_ELEMENT_SIZE_ERROR);
}

TEST(DpfCpuTest, ZipAndUnzipReportRuntimeError)
{
    std::uint8_t buffer[8] = {};

    EXPECT_EQ(
        FastFss_cpu_dpfKeyZip(buffer, sizeof(buffer), buffer, sizeof(buffer), 4,
                              5, 1, sizeof(std::uint8_t), 1),
        FAST_FSS_RUNTIME_ERROR);
    EXPECT_EQ(
        FastFss_cpu_dpfKeyUnzip(buffer, sizeof(buffer), buffer, sizeof(buffer),
                                4, 5, 1, sizeof(std::uint8_t), 1),
        FAST_FSS_RUNTIME_ERROR);
}

} // namespace
