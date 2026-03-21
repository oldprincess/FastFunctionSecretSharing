// clang-format off
// g++ -I include -I third_party/googletest/googletest/include -I third_party/googletest/googletest src/cpu/config.cpp src/cpu/dcf.cpp test/cpu/dcf.cpp third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cpu_dcf.exe -std=c++17 -maes -fopenmp
// clang-format on
#include <FastFss/cpu/dcf.h>
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
std::vector<T> reconstructShares(const std::vector<T> &share0,
                                 const std::vector<T> &share1,
                                 std::size_t           bitWidthOut)
{
    EXPECT_EQ(share0.size(), share1.size());

    std::vector<T> reconstructed(share0.size());
    for (std::size_t i = 0; i < share0.size(); ++i)
    {
        reconstructed[i] = modBits<T>(share0[i] + share1[i], bitWidthOut);
    }
    return reconstructed;
}

template <typename T>
struct DcfFixtureData
{
    std::size_t    bitWidthIn;
    std::size_t    bitWidthOut;
    std::size_t    groupSize;
    std::size_t    elementNum;
    std::vector<T> alpha;
    std::vector<T> beta;
    std::vector<T> maskedX;
    std::vector<T> expected;
};

template <typename T>
DcfFixtureData<T> makeVectorBetaCase()
{
    DcfFixtureData<T> data;
    data.bitWidthIn  = 8;
    data.bitWidthOut = 8;
    data.groupSize   = 3;
    data.elementNum  = 4;

    data.alpha   = {5, 10, 2, 200};
    data.maskedX = {3, 10, 1, 250};
    data.beta    = {
        11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43,
    };
    data.expected = {
        11, 12, 13, 0, 0, 0, 31, 32, 33, 0, 0, 0,
    };
    return data;
}

template <typename T>
DcfFixtureData<T> makeNullBetaCase()
{
    DcfFixtureData<T> data;
    data.bitWidthIn  = sizeof(T) == 1 ? 8 : 12;
    data.bitWidthOut = sizeof(T) == 16 ? 16 : 8;
    data.groupSize   = 1;
    data.elementNum  = 4;
    data.alpha       = {T(8), T(4), T(9), T(1)};
    data.maskedX     = {T(2), T(4), T(1), T(6)};
    data.expected    = {T(1), T(0), T(1), T(0)};
    return data;
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
        seeds[0][i] = static_cast<std::uint8_t>((17 * i + 3) & 0xff);
        seeds[1][i] = static_cast<std::uint8_t>((29 * i + 7) & 0xff);
    }
    return seeds;
}

template <typename T>
std::vector<T> evaluateAndReconstruct(const DcfFixtureData<T> &data,
                                      const void              *betaPtr,
                                      std::size_t              betaDataSize)
{
    auto seeds = makeSeeds<T>(data.elementNum);

    std::size_t keyDataSize = 0;
    if (FastFss_cpu_dcfGetKeyDataSize(
            &keyDataSize, data.bitWidthIn, data.bitWidthOut, data.groupSize,
            sizeof(T), data.elementNum) != FAST_FSS_SUCCESS)
    {
        ADD_FAILURE() << "FastFss_cpu_dcfGetKeyDataSize failed";
        return {};
    }
    if (keyDataSize == 0)
    {
        ADD_FAILURE() << "FastFss_cpu_dcfGetKeyDataSize returned zero";
        return {};
    }

    std::size_t cacheDataSize = 0;
    if (FastFss_cpu_dcfGetCacheDataSize(
            &cacheDataSize, data.bitWidthIn, data.bitWidthOut, data.groupSize,
            sizeof(T), data.elementNum) != FAST_FSS_SUCCESS)
    {
        ADD_FAILURE() << "FastFss_cpu_dcfGetCacheDataSize failed";
        return {};
    }

    std::vector<std::uint8_t> key(keyDataSize);
    if (FastFss_cpu_dcfKeyGen(key.data(), key.size(), data.alpha.data(),
                              data.alpha.size() * sizeof(T), betaPtr,
                              betaDataSize, seeds[0].data(), seeds[0].size(),
                              seeds[1].data(), seeds[1].size(), data.bitWidthIn,
                              data.bitWidthOut, data.groupSize, sizeof(T),
                              data.elementNum) != FAST_FSS_SUCCESS)
    {
        ADD_FAILURE() << "FastFss_cpu_dcfKeyGen failed";
        return {};
    }

    std::vector<T>            share0(data.elementNum * data.groupSize);
    std::vector<T>            share1(data.elementNum * data.groupSize);
    std::vector<std::uint8_t> cache0(cacheDataSize);
    std::vector<std::uint8_t> cache1(cacheDataSize);

    if (FastFss_cpu_dcfEval(
            share0.data(), share0.size() * sizeof(T), data.maskedX.data(),
            data.maskedX.size() * sizeof(T), key.data(), key.size(),
            seeds[0].data(), seeds[0].size(), 0, data.bitWidthIn,
            data.bitWidthOut, data.groupSize, sizeof(T), data.elementNum,
            cache0.data(), cache0.size()) != FAST_FSS_SUCCESS)
    {
        ADD_FAILURE() << "FastFss_cpu_dcfEval failed for party 0";
        return {};
    }

    if (FastFss_cpu_dcfEval(
            share1.data(), share1.size() * sizeof(T), data.maskedX.data(),
            data.maskedX.size() * sizeof(T), key.data(), key.size(),
            seeds[1].data(), seeds[1].size(), 1, data.bitWidthIn,
            data.bitWidthOut, data.groupSize, sizeof(T), data.elementNum,
            cache1.data(), cache1.size()) != FAST_FSS_SUCCESS)
    {
        ADD_FAILURE() << "FastFss_cpu_dcfEval failed for party 1";
        return {};
    }

    return reconstructShares(share0, share1, data.bitWidthOut);
}

template <typename T>
class DcfCpuTypedTest : public ::testing::Test
{
};

using DcfElementTypes = ::testing::
    Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, uint128_t>;

TYPED_TEST_SUITE(DcfCpuTypedTest, DcfElementTypes);

TYPED_TEST(DcfCpuTypedTest, VectorBetaUsesGroupSizeAsVectorLength)
{
    const auto data = makeVectorBetaCase<TypeParam>();

    const auto reconstructed = evaluateAndReconstruct<TypeParam>(
        data, data.beta.data(), data.beta.size() * sizeof(TypeParam));

    EXPECT_EQ(reconstructed, data.expected);
}

TYPED_TEST(DcfCpuTypedTest, NullBetaBehavesAsScalarOne)
{
    const auto data = makeNullBetaCase<TypeParam>();

    const auto reconstructed =
        evaluateAndReconstruct<TypeParam>(data, nullptr, 0);

    EXPECT_EQ(reconstructed, data.expected);
}

TYPED_TEST(DcfCpuTypedTest, SizeQueriesReturnSuccessForValidParameters)
{
    constexpr std::size_t bitWidthIn  = sizeof(TypeParam) == 1 ? 8 : 16;
    constexpr std::size_t bitWidthOut = sizeof(TypeParam) == 16 ? 16 : 8;
    constexpr std::size_t groupSize   = 5;
    constexpr std::size_t elementSize = sizeof(TypeParam);
    constexpr std::size_t elementNum  = 7;

    std::size_t keyDataSize   = 0;
    std::size_t cacheDataSize = 0;

    EXPECT_EQ(
        FastFss_cpu_dcfGetKeyDataSize(&keyDataSize, bitWidthIn, bitWidthOut,
                                      groupSize, elementSize, elementNum),
        FAST_FSS_SUCCESS);
    EXPECT_GT(keyDataSize, 0u);

    EXPECT_EQ(
        FastFss_cpu_dcfGetCacheDataSize(&cacheDataSize, bitWidthIn, bitWidthOut,
                                        groupSize, elementSize, elementNum),
        FAST_FSS_SUCCESS);
    EXPECT_GT(cacheDataSize, 0u);
}

TEST(DcfCpuTest, InvalidBitWidthReturnsError)
{
    std::size_t keyDataSize = 0;

    EXPECT_EQ(FastFss_cpu_dcfGetKeyDataSize(&keyDataSize, 0, 8, 1,
                                            sizeof(std::uint32_t), 1),
              FAST_FSS_INVALID_BITWIDTH_ERROR);

    EXPECT_EQ(FastFss_cpu_dcfGetCacheDataSize(&keyDataSize, 33, 8, 1,
                                              sizeof(std::uint32_t), 1),
              FAST_FSS_INVALID_BITWIDTH_ERROR);
}

TEST(DcfCpuTest, InvalidElementSizeReturnsError)
{
    std::size_t keyDataSize = 0;

    EXPECT_EQ(FastFss_cpu_dcfGetKeyDataSize(&keyDataSize, 8, 8, 1, 3, 1),
              FAST_FSS_INVALID_ELEMENT_SIZE_ERROR);
}

} // namespace
