// clang-format off
// nvcc -I include -I third_party/wideint/include -I third_party/googletest/googletest/include -I third_party/googletest/googletest src/cuda/config.cpp src/cuda/dpf.cu test/cuda/dpf.cu third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cuda_dpf.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/config.h>
#include <FastFss/cuda/dpf.h>
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
void makeDeviceKeyAndCache(const DpfFixtureData<T>                 &data,
                           const std::vector<std::uint8_t>         &seed0,
                           const std::vector<std::uint8_t>         &seed1,
                           std::unique_ptr<void, void (*)(void *)> &dKey,
                           std::unique_ptr<void, void (*)(void *)> &dCache)
{
    std::size_t keySize = 0;
    ASSERT_EQ(FastFss_cuda_dpfGetKeyDataSize(&keySize, data.bitWidthIn,
                                             data.bitWidthOut, data.groupSize,
                                             sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);
    std::size_t cacheSize = 0;
    ASSERT_EQ(FastFss_cuda_dpfGetCacheDataSize(&cacheSize, data.bitWidthIn,
                                               sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);

    dKey   = make_unique_gpu_ptr(keySize);
    dCache = make_unique_gpu_ptr(cacheSize);

    auto dAlpha = make_unique_gpu_ptr(const_cast<T *>(data.alpha.data()),
                                      data.alpha.size() * sizeof(T));
    auto dBeta  = make_unique_gpu_ptr(const_cast<T *>(data.beta.data()),
                                      data.beta.size() * sizeof(T));
    auto dSeed0 = make_unique_gpu_ptr(const_cast<std::uint8_t *>(seed0.data()),
                                      seed0.size());
    auto dSeed1 = make_unique_gpu_ptr(const_cast<std::uint8_t *>(seed1.data()),
                                      seed1.size());

    ASSERT_EQ(FastFss_cuda_dpfKeyGen(
                  dKey.get(), keySize, dAlpha.get(),
                  data.alpha.size() * sizeof(T), dBeta.get(),
                  data.beta.size() * sizeof(T), dSeed0.get(), seed0.size(),
                  dSeed1.get(), seed1.size(), data.bitWidthIn, data.bitWidthOut,
                  data.groupSize, sizeof(T), data.elementNum, nullptr),
              FAST_FSS_SUCCESS);
}

template <typename T>
class DpfCudaTypedTest : public ::testing::Test
{
};

using DpfElementTypes = ::testing::
    Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, uint128_t>;

TYPED_TEST_SUITE(DpfCudaTypedTest, DpfElementTypes);

TEST(DpfCudaTest, ReportsGridDim)
{
    EXPECT_GT(FastFss_cuda_getGridDim(), 0);
}

TYPED_TEST(DpfCudaTypedTest, EvalReturnsBetaOnlyAtZeroOffset)
{
    using T             = TypeParam;
    const auto  data    = makeFixtureData<T>();
    auto        seeds   = makeSeeds<T>(data.elementNum);
    std::size_t keySize = 0;
    ASSERT_EQ(FastFss_cuda_dpfGetKeyDataSize(&keySize, data.bitWidthIn,
                                             data.bitWidthOut, data.groupSize,
                                             sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);
    std::size_t cacheSize = 0;
    ASSERT_EQ(FastFss_cuda_dpfGetCacheDataSize(&cacheSize, data.bitWidthIn,
                                               sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);

    std::unique_ptr<void, void (*)(void *)> dKey(nullptr,
                                                 FastFss::cuda::free_gpu);
    std::unique_ptr<void, void (*)(void *)> dCache(nullptr,
                                                   FastFss::cuda::free_gpu);
    makeDeviceKeyAndCache(data, seeds[0], seeds[1], dKey, dCache);

    auto dMaskedX = make_unique_gpu_ptr(const_cast<T *>(data.maskedX.data()),
                                        data.maskedX.size() * sizeof(T));
    auto dSeed0   = make_unique_gpu_ptr(
        const_cast<std::uint8_t *>(seeds[0].data()), seeds[0].size());
    auto dSeed1 = make_unique_gpu_ptr(
        const_cast<std::uint8_t *>(seeds[1].data()), seeds[1].size());
    auto dShare0 =
        make_unique_gpu_ptr(data.elementNum * data.groupSize * sizeof(T));
    auto dShare1 =
        make_unique_gpu_ptr(data.elementNum * data.groupSize * sizeof(T));

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

    ASSERT_EQ(FastFss_cuda_dpfEval(
                  dShare0.get(), share0.size() * sizeof(T), dMaskedX.get(),
                  data.maskedX.size() * sizeof(T), dKey.get(), keySize,
                  dSeed0.get(), seeds[0].size(), 0, data.bitWidthIn,
                  data.bitWidthOut, data.groupSize, sizeof(T), data.elementNum,
                  dCache.get(), cacheSize, nullptr),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(
        FastFss_cuda_dpfEval(
            dShare1.get(), share1.size() * sizeof(T), dMaskedX.get(),
            data.maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed1.get(),
            seeds[1].size(), 1, data.bitWidthIn, data.bitWidthOut,
            data.groupSize, sizeof(T), data.elementNum, nullptr, 0, nullptr),
        FAST_FSS_SUCCESS);

    memcpy_gpu2cpu(share0.data(), dShare0.get(), share0.size() * sizeof(T));
    memcpy_gpu2cpu(share1.data(), dShare1.get(), share1.size() * sizeof(T));

    EXPECT_EQ(reconstructShares(share0, share1, data.bitWidthOut), expected);
}

TYPED_TEST(DpfCudaTypedTest, EvalAllPlacesBetaAtSelectedPoint)
{
    using T             = TypeParam;
    const auto  data    = makeFixtureData<T>();
    auto        seeds   = makeSeeds<T>(data.elementNum);
    std::size_t keySize = 0;
    ASSERT_EQ(FastFss_cuda_dpfGetKeyDataSize(&keySize, data.bitWidthIn,
                                             data.bitWidthOut, data.groupSize,
                                             sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);
    std::size_t cacheSize = 0;
    ASSERT_EQ(FastFss_cuda_dpfGetCacheDataSize(&cacheSize, data.bitWidthIn,
                                               sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);

    std::unique_ptr<void, void (*)(void *)> dKey(nullptr,
                                                 FastFss::cuda::free_gpu);
    std::unique_ptr<void, void (*)(void *)> dCache(nullptr,
                                                   FastFss::cuda::free_gpu);
    makeDeviceKeyAndCache(data, seeds[0], seeds[1], dKey, dCache);

    const std::size_t domainSize = std::size_t{1} << data.bitWidthIn;
    auto dMaskedX = make_unique_gpu_ptr(const_cast<T *>(data.maskedX.data()),
                                        data.maskedX.size() * sizeof(T));
    auto dSeed0   = make_unique_gpu_ptr(
        const_cast<std::uint8_t *>(seeds[0].data()), seeds[0].size());
    auto dSeed1 = make_unique_gpu_ptr(
        const_cast<std::uint8_t *>(seeds[1].data()), seeds[1].size());
    auto dShare0 = make_unique_gpu_ptr(data.elementNum * domainSize *
                                       data.groupSize * sizeof(T));
    auto dShare1 = make_unique_gpu_ptr(data.elementNum * domainSize *
                                       data.groupSize * sizeof(T));

    std::vector<T> share0(data.elementNum * domainSize * data.groupSize);
    std::vector<T> share1(data.elementNum * domainSize * data.groupSize);

    ASSERT_EQ(FastFss_cuda_dpfEvalAll(
                  dShare0.get(), share0.size() * sizeof(T), dMaskedX.get(),
                  data.maskedX.size() * sizeof(T), dKey.get(), keySize,
                  dSeed0.get(), seeds[0].size(), 0, data.bitWidthIn,
                  data.bitWidthOut, data.groupSize, sizeof(T), data.elementNum,
                  dCache.get(), cacheSize, nullptr),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(
        FastFss_cuda_dpfEvalAll(
            dShare1.get(), share1.size() * sizeof(T), dMaskedX.get(),
            data.maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed1.get(),
            seeds[1].size(), 1, data.bitWidthIn, data.bitWidthOut,
            data.groupSize, sizeof(T), data.elementNum, nullptr, 0, nullptr),
        FAST_FSS_SUCCESS);

    memcpy_gpu2cpu(share0.data(), dShare0.get(), share0.size() * sizeof(T));
    memcpy_gpu2cpu(share1.data(), dShare1.get(), share1.size() * sizeof(T));

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

TYPED_TEST(DpfCudaTypedTest, EvalMultiMatchesChosenPoints)
{
    using T             = TypeParam;
    const auto  data    = makeFixtureData<T>();
    auto        seeds   = makeSeeds<T>(data.elementNum);
    std::size_t keySize = 0;
    ASSERT_EQ(FastFss_cuda_dpfGetKeyDataSize(&keySize, data.bitWidthIn,
                                             data.bitWidthOut, data.groupSize,
                                             sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);
    std::size_t cacheSize = 0;
    ASSERT_EQ(FastFss_cuda_dpfGetCacheDataSize(&cacheSize, data.bitWidthIn,
                                               sizeof(T), data.elementNum),
              FAST_FSS_SUCCESS);

    std::unique_ptr<void, void (*)(void *)> dKey(nullptr,
                                                 FastFss::cuda::free_gpu);
    std::unique_ptr<void, void (*)(void *)> dCache(nullptr,
                                                   FastFss::cuda::free_gpu);
    makeDeviceKeyAndCache(data, seeds[0], seeds[1], dKey, dCache);

    const std::vector<T> points = {T(0), T(2), T(4), T(7)};
    auto dMaskedX = make_unique_gpu_ptr(const_cast<T *>(data.maskedX.data()),
                                        data.maskedX.size() * sizeof(T));
    auto dPoints  = make_unique_gpu_ptr(const_cast<T *>(points.data()),
                                        points.size() * sizeof(T));
    auto dSeed0   = make_unique_gpu_ptr(
        const_cast<std::uint8_t *>(seeds[0].data()), seeds[0].size());
    auto dSeed1 = make_unique_gpu_ptr(
        const_cast<std::uint8_t *>(seeds[1].data()), seeds[1].size());
    auto dShare0 = make_unique_gpu_ptr(data.elementNum * points.size() *
                                       data.groupSize * sizeof(T));
    auto dShare1 = make_unique_gpu_ptr(data.elementNum * points.size() *
                                       data.groupSize * sizeof(T));

    std::vector<T> share0(data.elementNum * points.size() * data.groupSize);
    std::vector<T> share1(data.elementNum * points.size() * data.groupSize);

    ASSERT_EQ(
        FastFss_cuda_dpfEvalMulti(
            dShare0.get(), share0.size() * sizeof(T), dMaskedX.get(),
            data.maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed0.get(),
            seeds[0].size(), 0, dPoints.get(), points.size() * sizeof(T),
            data.bitWidthIn, data.bitWidthOut, data.groupSize, sizeof(T),
            data.elementNum, dCache.get(), cacheSize, nullptr),
        FAST_FSS_SUCCESS);
    ASSERT_EQ(
        FastFss_cuda_dpfEvalMulti(
            dShare1.get(), share1.size() * sizeof(T), dMaskedX.get(),
            data.maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed1.get(),
            seeds[1].size(), 1, dPoints.get(), points.size() * sizeof(T),
            data.bitWidthIn, data.bitWidthOut, data.groupSize, sizeof(T),
            data.elementNum, nullptr, 0, nullptr),
        FAST_FSS_SUCCESS);

    memcpy_gpu2cpu(share0.data(), dShare0.get(), share0.size() * sizeof(T));
    memcpy_gpu2cpu(share1.data(), dShare1.get(), share1.size() * sizeof(T));

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

} // namespace
