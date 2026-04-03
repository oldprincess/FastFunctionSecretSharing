#include <FastFss/cuda/grotto.h>
#include <FastFss/errors.h>
#include <FastFss/grotto.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/cuda_test_utils.cuh"
#include "common/test_utils.hpp"

namespace FastFss::tests {
namespace {

TEST(CudaGrottoIntervalLutEvalTest, IntervalLutEvalReconstructsLookupValue)
{
    using T                             = std::uint8_t;
    constexpr std::size_t bitWidthIn    = 7;
    constexpr std::size_t bitWidthOut   = 8;
    constexpr std::size_t elementNum    = 17;
    constexpr std::size_t intervalCount = 3;
    constexpr std::size_t lutNum        = 2;

    for (const bool useStreams : {false, true})
    {
        ::FastFss::tests::cuda::StreamPair streams;
        auto                               seeds = make_seeds<T>(elementNum, 43, 1, 47, 9);
        auto                               alpha = random_vector<T>(elementNum, bitWidthIn);
        auto                               x     = random_vector<T>(elementNum, bitWidthIn);
        std::vector<T>                     maskedX(elementNum);
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

        auto dKey     = ::FastFss::tests::cuda::make_gpu_buffer(keySize);
        auto dAlpha   = ::FastFss::tests::cuda::make_gpu_buffer(alpha.data(), alpha.size() * sizeof(T));
        auto dMaskedX = ::FastFss::tests::cuda::make_gpu_buffer(maskedX.data(), maskedX.size() * sizeof(T));
        auto dSeed0   = ::FastFss::tests::cuda::make_gpu_buffer(seeds[0].data(), seeds[0].size());
        auto dSeed1   = ::FastFss::tests::cuda::make_gpu_buffer(seeds[1].data(), seeds[1].size());
        auto dLeft    = ::FastFss::tests::cuda::make_gpu_buffer(left.data(), left.size() * sizeof(T));
        auto dRight   = ::FastFss::tests::cuda::make_gpu_buffer(right.data(), right.size() * sizeof(T));
        auto dTable   = ::FastFss::tests::cuda::make_gpu_buffer(table.data(), table.size() * sizeof(T));
        auto dCache   = ::FastFss::tests::cuda::make_gpu_buffer(cacheSize);

        ASSERT_EQ(FastFss_cuda_grottoKeyGen(dKey.get(), keySize, dAlpha.get(), alpha.size() * sizeof(T), dSeed0.get(),
                                            seeds[0].size(), dSeed1.get(), seeds[1].size(), bitWidthIn, sizeof(T),
                                            elementNum, useStreams ? streams.party_stream(0) : nullptr),
                  FAST_FSS_SUCCESS);
        if (useStreams)
        {
            streams.synchronize(0);
        }

        auto dShareE0 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
        auto dShareE1 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
        auto dShareT0 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * lutNum * sizeof(T));
        auto dShareT1 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * lutNum * sizeof(T));
        ASSERT_EQ(FastFss_cuda_grottoIntervalLutEval(
                      dShareE0.get(), elementNum * sizeof(T), dShareT0.get(), elementNum * lutNum * sizeof(T),
                      dMaskedX.get(), maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed0.get(), seeds[0].size(), 0,
                      dLeft.get(), left.size() * sizeof(T), dRight.get(), right.size() * sizeof(T), dTable.get(),
                      table.size() * sizeof(T), bitWidthIn, bitWidthOut, sizeof(T), elementNum, dCache.get(), cacheSize,
                      useStreams ? streams.party_stream(0) : nullptr),
                  FAST_FSS_SUCCESS);
        ASSERT_EQ(FastFss_cuda_grottoIntervalLutEval(
                      dShareE1.get(), elementNum * sizeof(T), dShareT1.get(), elementNum * lutNum * sizeof(T),
                      dMaskedX.get(), maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed1.get(), seeds[1].size(), 1,
                      dLeft.get(), left.size() * sizeof(T), dRight.get(), right.size() * sizeof(T), dTable.get(),
                      table.size() * sizeof(T), bitWidthIn, bitWidthOut, sizeof(T), elementNum, nullptr, 0,
                      useStreams ? streams.party_stream(1) : nullptr),
                  FAST_FSS_SUCCESS);
        if (useStreams)
        {
            streams.synchronize_all();
        }

        std::vector<T> shareE0(elementNum), shareE1(elementNum);
        std::vector<T> shareT0(elementNum * lutNum), shareT1(elementNum * lutNum);
        ::FastFss::tests::cuda::memcpy_gpu_to_cpu(shareE0.data(), dShareE0.get(), shareE0.size() * sizeof(T));
        ::FastFss::tests::cuda::memcpy_gpu_to_cpu(shareE1.data(), dShareE1.get(), shareE1.size() * sizeof(T));
        ::FastFss::tests::cuda::memcpy_gpu_to_cpu(shareT0.data(), dShareT0.get(), shareT0.size() * sizeof(T));
        ::FastFss::tests::cuda::memcpy_gpu_to_cpu(shareT1.data(), dShareT1.get(), shareT1.size() * sizeof(T));

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
}

} // namespace
} // namespace FastFss::tests
