// clang-format off
// nvcc -I include -I third_party/googletest/googletest/include -I third_party/wideint/include -I third_party/googletest/googletest src/cuda/config.cpp src/cuda/dcf.cu test/cuda/dcf.cu third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cuda_dcf.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/dcf.h>
#include <FastFss/errors.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "utils.cuh"
#include "wideint/wideint.hpp"

namespace {

using uint128_t = wideint::uint<2>;

using FastFss::cuda::make_unique_gpu_ptr;
using FastFss::cuda::memcpy_gpu2cpu;

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
        seeds[0][i] = static_cast<std::uint8_t>((17 * i + 3) & 0xff);
        seeds[1][i] = static_cast<std::uint8_t>((29 * i + 7) & 0xff);
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
    data.alpha       = {5, 10, 2, 200};
    data.maskedX     = {3, 10, 1, 250};
    data.beta        = {11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43};
    data.expected    = {11, 12, 13, 0, 0, 0, 31, 32, 33, 0, 0, 0};
    return data;
}

template <typename T>
std::vector<T> evaluateAndReconstruct(const DcfFixtureData<T> &data,
                                      const void              *betaPtr,
                                      std::size_t              betaDataSize)
{
    auto seeds = makeSeeds<T>(data.elementNum);

    std::size_t keyDataSize = 0;
    EXPECT_EQ(FastFss_cuda_dcfGetKeyDataSize(&keyDataSize, data.bitWidthIn,
                                             data.bitWidthOut, data.groupSize,
                                             sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);
    std::size_t cacheDataSize = 0;
    EXPECT_EQ(FastFss_cuda_dcfGetCacheDataSize(&cacheDataSize, data.bitWidthIn,
                                               data.bitWidthOut, data.groupSize,
                                               sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);

    auto dKey     = make_unique_gpu_ptr(keyDataSize);
    auto dAlpha   = make_unique_gpu_ptr(const_cast<T *>(data.alpha.data()),
                                        data.alpha.size() * sizeof(T));
    auto dMaskedX = make_unique_gpu_ptr(const_cast<T *>(data.maskedX.data()),
                                        data.maskedX.size() * sizeof(T));
    auto dSeed0   = make_unique_gpu_ptr(seeds[0].data(), seeds[0].size());
    auto dSeed1   = make_unique_gpu_ptr(seeds[1].data(), seeds[1].size());
    auto dBeta =
        betaPtr == nullptr
            ? std::unique_ptr<void, void (*)(void *)>(nullptr,
                                                      FastFss::cuda::free_gpu)
            : make_unique_gpu_ptr(const_cast<void *>(betaPtr), betaDataSize);
    auto dShare0 =
        make_unique_gpu_ptr(data.elementNum * data.groupSize * sizeof(T));
    auto dShare1 =
        make_unique_gpu_ptr(data.elementNum * data.groupSize * sizeof(T));
    auto dCache = make_unique_gpu_ptr(cacheDataSize);

    EXPECT_EQ(FastFss_cuda_dcfKeyGen(
                  dKey.get(), keyDataSize, dAlpha.get(),
                  data.alpha.size() * sizeof(T), dBeta.get(), betaDataSize,
                  dSeed0.get(), seeds[0].size(), dSeed1.get(), seeds[1].size(),
                  data.bitWidthIn, data.bitWidthOut, data.groupSize, sizeof(T),
                  data.elementNum, nullptr),
              FAST_FSS_SUCCESS);
    EXPECT_EQ(FastFss_cuda_dcfEval(
                  dShare0.get(), data.elementNum * data.groupSize * sizeof(T),
                  dMaskedX.get(), data.maskedX.size() * sizeof(T), dKey.get(),
                  keyDataSize, dSeed0.get(), seeds[0].size(), 0,
                  data.bitWidthIn, data.bitWidthOut, data.groupSize, sizeof(T),
                  data.elementNum, dCache.get(), cacheDataSize, nullptr),
              FAST_FSS_SUCCESS);
    EXPECT_EQ(FastFss_cuda_dcfEval(
                  dShare1.get(), data.elementNum * data.groupSize * sizeof(T),
                  dMaskedX.get(), data.maskedX.size() * sizeof(T), dKey.get(),
                  keyDataSize, dSeed1.get(), seeds[1].size(), 1,
                  data.bitWidthIn, data.bitWidthOut, data.groupSize, sizeof(T),
                  data.elementNum, nullptr, 0, nullptr),
              FAST_FSS_SUCCESS);

    std::vector<T> share0(data.elementNum * data.groupSize);
    std::vector<T> share1(data.elementNum * data.groupSize);
    memcpy_gpu2cpu(share0.data(), dShare0.get(), share0.size() * sizeof(T));
    memcpy_gpu2cpu(share1.data(), dShare1.get(), share1.size() * sizeof(T));
    return reconstructShares(share0, share1, data.bitWidthOut);
}

template <typename T>
class DcfCudaTypedTest : public ::testing::Test
{
};

using DcfElementTypes = ::testing::
    Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, uint128_t>;

TYPED_TEST_SUITE(DcfCudaTypedTest, DcfElementTypes);

TYPED_TEST(DcfCudaTypedTest, VectorBetaUsesGroupSizeAsVectorLength)
{
    const auto data          = makeVectorBetaCase<TypeParam>();
    const auto reconstructed = evaluateAndReconstruct<TypeParam>(
        data, data.beta.data(), data.beta.size() * sizeof(TypeParam));
    EXPECT_EQ(reconstructed, data.expected);
}

TEST(DcfCudaTest, InvalidBitWidthReturnsError)
{
    std::size_t keyDataSize = 0;
    EXPECT_EQ(FastFss_cuda_dcfGetKeyDataSize(&keyDataSize, 0, 8, 1,
                                             sizeof(std::uint32_t), 1),
              FAST_FSS_INVALID_BITWIDTH_ERROR);
    EXPECT_EQ(FastFss_cuda_dcfGetCacheDataSize(&keyDataSize, 33, 8, 1,
                                               sizeof(std::uint32_t), 1),
              FAST_FSS_INVALID_BITWIDTH_ERROR);
}

TEST(DcfCudaTest, InvalidElementSizeReturnsError)
{
    std::size_t keyDataSize = 0;
    EXPECT_EQ(FastFss_cuda_dcfGetKeyDataSize(&keyDataSize, 8, 8, 1, 3, 1),
              FAST_FSS_INVALID_ELEMENT_SIZE_ERROR);
}

} // namespace
