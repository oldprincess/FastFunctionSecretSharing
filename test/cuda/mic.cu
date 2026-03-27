// clang-format off
// nvcc -I include -I third_party/wideint/include -I third_party/googletest/googletest/include -I third_party/googletest/googletest src/cuda/config.cpp src/cuda/mic.cu test/cuda/mic.cu third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cuda_mic.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/mic.h>
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

using FastFss::cuda::free_gpu;
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
        seeds[0][i] = static_cast<std::uint8_t>((29 * i + 7) & 0xff);
        seeds[1][i] = static_cast<std::uint8_t>((11 * i + 5) & 0xff);
    }
    return seeds;
}

template <typename T>
class MicCudaTypedTest : public ::testing::Test
{
};

using MicElementTypes = ::testing::
    Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, uint128_t>;

TYPED_TEST_SUITE(MicCudaTypedTest, MicElementTypes);

TYPED_TEST(MicCudaTypedTest, IntervalEvaluationMatchesExpectedMembership)
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

    std::size_t keySize = 0;
    ASSERT_EQ(FastFss_cuda_dcfMICGetKeyDataSize(
                  &keySize, bitWidthIn, bitWidthOut, sizeof(T), elementNum),
              FAST_FSS_SUCCESS);
    std::size_t cacheSize = 0;
    ASSERT_EQ(FastFss_cuda_dcfMICGetCacheDataSize(
                  &cacheSize, bitWidthIn, bitWidthOut, sizeof(T), elementNum),
              FAST_FSS_SUCCESS);
    std::vector<T> z(elementNum * leftEndpoints.size());

    auto dKey   = make_unique_gpu_ptr(keySize);
    auto dZ     = make_unique_gpu_ptr(z.size() * sizeof(T));
    auto dAlpha = make_unique_gpu_ptr(const_cast<T *>(alpha.data()),
                                      alpha.size() * sizeof(T));
    auto dSeed0 = make_unique_gpu_ptr(
        const_cast<std::uint8_t *>(seeds[0].data()), seeds[0].size());
    auto dSeed1 = make_unique_gpu_ptr(
        const_cast<std::uint8_t *>(seeds[1].data()), seeds[1].size());
    auto dLeft  = make_unique_gpu_ptr(const_cast<T *>(leftEndpoints.data()),
                                      leftEndpoints.size() * sizeof(T));
    auto dRight = make_unique_gpu_ptr(const_cast<T *>(rightEndpoints.data()),
                                      rightEndpoints.size() * sizeof(T));
    auto dCache = make_unique_gpu_ptr(cacheSize);

    ASSERT_EQ(FastFss_cuda_dcfMICKeyGen(
                  dKey.get(), keySize, dZ.get(), z.size() * sizeof(T),
                  dAlpha.get(), alpha.size() * sizeof(T), dSeed0.get(),
                  seeds[0].size(), dSeed1.get(), seeds[1].size(), dLeft.get(),
                  leftEndpoints.size() * sizeof(T), dRight.get(),
                  rightEndpoints.size() * sizeof(T), bitWidthIn, bitWidthOut,
                  sizeof(T), elementNum, nullptr),
              FAST_FSS_SUCCESS);
    memcpy_gpu2cpu(z.data(), dZ.get(), z.size() * sizeof(T));

    std::vector<T> sharedZ0(z.size());
    std::vector<T> sharedZ1(z.size());
    for (std::size_t i = 0; i < z.size(); ++i)
    {
        sharedZ0[i] = static_cast<T>(i + 1);
        sharedZ1[i] = z[i] - sharedZ0[i];
    }

    auto dMaskedX  = make_unique_gpu_ptr(const_cast<T *>(maskedX.data()),
                                         maskedX.size() * sizeof(T));
    auto dSharedZ0 = make_unique_gpu_ptr(const_cast<T *>(sharedZ0.data()),
                                         sharedZ0.size() * sizeof(T));
    auto dSharedZ1 = make_unique_gpu_ptr(const_cast<T *>(sharedZ1.data()),
                                         sharedZ1.size() * sizeof(T));
    auto dOut0     = make_unique_gpu_ptr(sharedZ0.size() * sizeof(T));
    auto dOut1     = make_unique_gpu_ptr(sharedZ1.size() * sizeof(T));

    std::vector<T> share0(sharedZ0.size());
    std::vector<T> share1(sharedZ1.size());

    ASSERT_EQ(
        FastFss_cuda_dcfMICEval(
            dOut0.get(), share0.size() * sizeof(T), dMaskedX.get(),
            maskedX.size() * sizeof(T), dKey.get(), keySize, dSharedZ0.get(),
            sharedZ0.size() * sizeof(T), dSeed0.get(), seeds[0].size(), 0,
            dLeft.get(), leftEndpoints.size() * sizeof(T), dRight.get(),
            rightEndpoints.size() * sizeof(T), bitWidthIn, bitWidthOut,
            sizeof(T), elementNum, dCache.get(), cacheSize, nullptr),
        FAST_FSS_SUCCESS);
    ASSERT_EQ(
        FastFss_cuda_dcfMICEval(
            dOut1.get(), share1.size() * sizeof(T), dMaskedX.get(),
            maskedX.size() * sizeof(T), dKey.get(), keySize, dSharedZ1.get(),
            sharedZ1.size() * sizeof(T), dSeed1.get(), seeds[1].size(), 1,
            dLeft.get(), leftEndpoints.size() * sizeof(T), dRight.get(),
            rightEndpoints.size() * sizeof(T), bitWidthIn, bitWidthOut,
            sizeof(T), elementNum, nullptr, 0, nullptr),
        FAST_FSS_SUCCESS);

    memcpy_gpu2cpu(share0.data(), dOut0.get(), share0.size() * sizeof(T));
    memcpy_gpu2cpu(share1.data(), dOut1.get(), share1.size() * sizeof(T));

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
