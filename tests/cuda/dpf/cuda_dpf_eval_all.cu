#include <FastFss/cuda/dpf.h>
#include <FastFss/dpf.h>
#include <FastFss/errors.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/cuda_test_utils.cuh"
#include "common/test_types.hpp"
#include "common/test_utils.hpp"

namespace FastFss::tests {
namespace {

template <typename T>
class CudaDpfEvalAllTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CudaDpfEvalAllTest, EvalElementTypes);

TYPED_TEST(CudaDpfEvalAllTest, EvalAllPlacesBetaAtSelectedPoint)
{
    using T                         = TypeParam;
    constexpr std::size_t groupSize = 3;

    for (const bool useStreams : {false, true})
    {
        ::FastFss::tests::cuda::StreamPair streams;
        for (const auto &widths : eval_width_configs<T>())
        {
            if (widths.bitWidthIn > 10)
            {
                continue;
            }
            for (const auto elementNum : element_counts())
            {
                auto           seeds = make_seeds<T>(elementNum, 13, 1, 23, 7);
                auto           alpha = random_vector<T>(elementNum, widths.bitWidthIn);
                auto           x     = random_vector<T>(elementNum, widths.bitWidthIn);
                auto           beta  = random_vector<T>(elementNum * groupSize, widths.bitWidthOut);
                std::vector<T> maskedX(elementNum);
                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    maskedX[i] = mod_bits<T>(alpha[i] + x[i], widths.bitWidthIn);
                }

                std::size_t keySize = 0;
                ASSERT_EQ(FastFss_dpfGetKeyDataSize(&keySize, widths.bitWidthIn, widths.bitWidthOut, groupSize,
                                                    sizeof(T), elementNum),
                          FAST_FSS_SUCCESS);
                std::size_t cacheSize = 0;
                ASSERT_EQ(FastFss_dpfGetCacheDataSize(&cacheSize, widths.bitWidthIn, sizeof(T), elementNum),
                          FAST_FSS_SUCCESS);

                auto dKey = ::FastFss::tests::cuda::make_gpu_buffer(keySize);
                auto dAlpha =
                    ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(alpha.data()), alpha.size() * sizeof(T));
                auto dMaskedX = ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(maskedX.data()),
                                                                        maskedX.size() * sizeof(T));
                auto dBeta =
                    ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(beta.data()), beta.size() * sizeof(T));
                auto              dSeed0 = ::FastFss::tests::cuda::make_gpu_buffer(seeds[0].data(), seeds[0].size());
                auto              dSeed1 = ::FastFss::tests::cuda::make_gpu_buffer(seeds[1].data(), seeds[1].size());
                const std::size_t domainSize = std::size_t{1} << widths.bitWidthIn;
                auto dShare0 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * domainSize * groupSize * sizeof(T));
                auto dShare1 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * domainSize * groupSize * sizeof(T));
                auto dCache  = ::FastFss::tests::cuda::make_gpu_buffer(cacheSize);

                ASSERT_EQ(
                    FastFss_cuda_dpfKeyGen(dKey.get(), keySize, dAlpha.get(), alpha.size() * sizeof(T), dBeta.get(),
                                           beta.size() * sizeof(T), dSeed0.get(), seeds[0].size(), dSeed1.get(),
                                           seeds[1].size(), widths.bitWidthIn, widths.bitWidthOut, groupSize, sizeof(T),
                                           elementNum, useStreams ? streams.party_stream(0) : nullptr),
                    FAST_FSS_SUCCESS);
                if (useStreams)
                {
                    streams.synchronize(0);
                }
                ASSERT_EQ(FastFss_cuda_dpfEvalAll(dShare0.get(), elementNum * domainSize * groupSize * sizeof(T),
                                                  dMaskedX.get(), maskedX.size() * sizeof(T), dKey.get(), keySize,
                                                  dSeed0.get(), seeds[0].size(), 0, widths.bitWidthIn,
                                                  widths.bitWidthOut, groupSize, sizeof(T), elementNum, dCache.get(),
                                                  cacheSize, useStreams ? streams.party_stream(0) : nullptr),
                          FAST_FSS_SUCCESS);
                ASSERT_EQ(FastFss_cuda_dpfEvalAll(dShare1.get(), elementNum * domainSize * groupSize * sizeof(T),
                                                  dMaskedX.get(), maskedX.size() * sizeof(T), dKey.get(), keySize,
                                                  dSeed1.get(), seeds[1].size(), 1, widths.bitWidthIn,
                                                  widths.bitWidthOut, groupSize, sizeof(T), elementNum, nullptr, 0,
                                                  useStreams ? streams.party_stream(1) : nullptr),
                          FAST_FSS_SUCCESS);
                if (useStreams)
                {
                    streams.synchronize_all();
                }

                std::vector<T> share0(elementNum * domainSize * groupSize);
                std::vector<T> share1(elementNum * domainSize * groupSize);
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(share0.data(), dShare0.get(), share0.size() * sizeof(T));
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(share1.data(), dShare1.get(), share1.size() * sizeof(T));
                const auto reconstructed = reconstruct_arithmetic(share0, share1, widths.bitWidthOut);
                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    for (std::size_t point = 0; point < domainSize; ++point)
                    {
                        for (std::size_t j = 0; j < groupSize; ++j)
                        {
                            const std::size_t index    = i * domainSize * groupSize + point * groupSize + j;
                            const T           expected = point == static_cast<std::size_t>(x[i])
                                                             ? mod_bits<T>(beta[i * groupSize + j], widths.bitWidthOut)
                                                             : T(0);
                            if (reconstructed[index] != expected)
                            {
                                FAIL() << "useStreams=" << useStreams << ", bitWidthIn=" << widths.bitWidthIn
                                       << ", bitWidthOut=" << widths.bitWidthOut << ", elementNum=" << elementNum
                                       << ", elementBits=" << (sizeof(T) * 8) << ", i=" << i << ", point=" << point
                                       << ", groupIndex=" << j;
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace
} // namespace FastFss::tests
