// clang-format off
// nvcc -I include -I third_party/googletest/googletest/include -I third_party/googletest/googletest src/cuda/config.cpp src/cuda/grotto.cu test/cuda/grotto.cu third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cuda_grotto.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/grotto.h>
#include <FastFss/errors.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "utils.cuh"

namespace {

using FastFss::cuda::make_unique_gpu_ptr;
using FastFss::cuda::memcpy_gpu2cpu;

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
std::vector<std::uint8_t> makeSeedBytes(std::size_t  elementNum,
                                        std::uint8_t mul,
                                        std::uint8_t add)
{
    std::vector<std::uint8_t> seed(16 * elementNum);
    for (std::size_t i = 0; i < seed.size(); ++i)
    {
        seed[i] = static_cast<std::uint8_t>((mul * i + add) & 0xff);
    }
    return seed;
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

TEST(GrottoCudaTest, EqEvalReconstructsZeroCase)
{
    using T                          = std::uint8_t;
    constexpr std::size_t bitWidthIn = 8;
    const std::vector<T>  alpha      = {3, 7, 11};
    const std::vector<T>  x          = {0, 5, 0};
    const std::size_t     n          = alpha.size();
    auto                  seed0      = makeSeedBytes<T>(n, 13, 1);
    auto                  seed1      = makeSeedBytes<T>(n, 17, 9);

    std::vector<T> maskedX(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        maskedX[i] = modBits<T>(alpha[i] + x[i], bitWidthIn);
    }

    std::size_t keySize = 0;
    ASSERT_EQ(
        FastFss_cuda_grottoGetKeyDataSize(&keySize, bitWidthIn, sizeof(T), n),
        FAST_FSS_SUCCESS);

    auto dKey     = make_unique_gpu_ptr(keySize);
    auto dAlpha   = make_unique_gpu_ptr(const_cast<T *>(alpha.data()),
                                        alpha.size() * sizeof(T));
    auto dSeed0   = make_unique_gpu_ptr(seed0.data(), seed0.size());
    auto dSeed1   = make_unique_gpu_ptr(seed1.data(), seed1.size());
    auto dMaskedX = make_unique_gpu_ptr(const_cast<T *>(maskedX.data()),
                                        maskedX.size() * sizeof(T));
    auto dOut0    = make_unique_gpu_ptr(maskedX.size() * sizeof(T));
    auto dOut1    = make_unique_gpu_ptr(maskedX.size() * sizeof(T));

    ASSERT_EQ(FastFss_cuda_grottoKeyGen(
                  dKey.get(), keySize, dAlpha.get(), alpha.size() * sizeof(T),
                  dSeed0.get(), seed0.size(), dSeed1.get(), seed1.size(),
                  bitWidthIn, sizeof(T), n, nullptr),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(
        FastFss_cuda_grottoEqEval(
            dOut0.get(), maskedX.size() * sizeof(T), dMaskedX.get(),
            maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed0.get(),
            seed0.size(), 0, bitWidthIn, sizeof(T), n, nullptr, 0, nullptr),
        FAST_FSS_SUCCESS);
    ASSERT_EQ(
        FastFss_cuda_grottoEqEval(
            dOut1.get(), maskedX.size() * sizeof(T), dMaskedX.get(),
            maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed1.get(),
            seed1.size(), 1, bitWidthIn, sizeof(T), n, nullptr, 0, nullptr),
        FAST_FSS_SUCCESS);

    std::vector<T> out0(n), out1(n);
    memcpy_gpu2cpu(out0.data(), dOut0.get(), out0.size() * sizeof(T));
    memcpy_gpu2cpu(out1.data(), dOut1.get(), out1.size() * sizeof(T));
    EXPECT_EQ(reconstructBooleanShares(out0, out1), (std::vector<T>{1, 0, 1}));
}

TEST(GrottoCudaTest, EqEvalMultiMatchesPoints)
{
    using T                          = std::uint16_t;
    constexpr std::size_t bitWidthIn = 12;
    const std::vector<T>  alpha      = {10, 20};
    const std::vector<T>  x          = {3, 7};
    const std::vector<T>  points     = {7, 3, 1};
    const std::size_t     n          = alpha.size();
    auto                  seed0      = makeSeedBytes<T>(n, 7, 3);
    auto                  seed1      = makeSeedBytes<T>(n, 9, 5);

    std::vector<T> maskedX(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        maskedX[i] = modBits<T>(alpha[i] + x[i], bitWidthIn);
    }

    std::size_t keySize = 0;
    ASSERT_EQ(
        FastFss_cuda_grottoGetKeyDataSize(&keySize, bitWidthIn, sizeof(T), n),
        FAST_FSS_SUCCESS);
    std::size_t cacheSize = 0;
    ASSERT_EQ(FastFss_cuda_grottoGetCacheDataSize(&cacheSize, bitWidthIn,
                                                  sizeof(T), n),
              FAST_FSS_SUCCESS);

    auto dKey     = make_unique_gpu_ptr(keySize);
    auto dAlpha   = make_unique_gpu_ptr(const_cast<T *>(alpha.data()),
                                        alpha.size() * sizeof(T));
    auto dSeed0   = make_unique_gpu_ptr(seed0.data(), seed0.size());
    auto dSeed1   = make_unique_gpu_ptr(seed1.data(), seed1.size());
    auto dMaskedX = make_unique_gpu_ptr(const_cast<T *>(maskedX.data()),
                                        maskedX.size() * sizeof(T));
    auto dPoints  = make_unique_gpu_ptr(const_cast<T *>(points.data()),
                                        points.size() * sizeof(T));
    auto dOut0    = make_unique_gpu_ptr(n * points.size() * sizeof(T));
    auto dOut1    = make_unique_gpu_ptr(n * points.size() * sizeof(T));
    auto dCache   = make_unique_gpu_ptr(cacheSize);

    ASSERT_EQ(FastFss_cuda_grottoKeyGen(
                  dKey.get(), keySize, dAlpha.get(), alpha.size() * sizeof(T),
                  dSeed0.get(), seed0.size(), dSeed1.get(), seed1.size(),
                  bitWidthIn, sizeof(T), n, nullptr),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cuda_grottoEqEvalMulti(
                  dOut0.get(), n * points.size() * sizeof(T), dMaskedX.get(),
                  maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed0.get(),
                  seed0.size(), 0, dPoints.get(), points.size() * sizeof(T),
                  bitWidthIn, sizeof(T), n, dCache.get(), cacheSize, nullptr),
              FAST_FSS_SUCCESS);
    ASSERT_EQ(FastFss_cuda_grottoEqEvalMulti(
                  dOut1.get(), n * points.size() * sizeof(T), dMaskedX.get(),
                  maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed1.get(),
                  seed1.size(), 1, dPoints.get(), points.size() * sizeof(T),
                  bitWidthIn, sizeof(T), n, nullptr, 0, nullptr),
              FAST_FSS_SUCCESS);

    std::vector<T> out0(n * points.size()), out1(n * points.size());
    memcpy_gpu2cpu(out0.data(), dOut0.get(), out0.size() * sizeof(T));
    memcpy_gpu2cpu(out1.data(), dOut1.get(), out1.size() * sizeof(T));
    const auto reconstructed = reconstructBooleanShares(out0, out1);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < points.size(); ++j)
        {
            EXPECT_EQ(reconstructed[i * points.size() + j],
                      x[i] == points[j] ? 1 : 0);
        }
    }
}

} // namespace
