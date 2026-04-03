#include <FastFss/cuda/grotto.h>
#include <FastFss/errors.h>
#include <FastFss/grotto.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/cuda_test_utils.cuh"
#include "common/test_types.hpp"
#include "common/test_utils.hpp"

namespace FastFss::tests {
namespace {

template <typename T>
std::vector<WidthConfig> grotto_width_configs()
{
    auto                     configs = eval_width_configs<T>();
    std::vector<WidthConfig> filtered;
    for (const auto &config : configs)
    {
        if (config.bitWidthIn >= 6)
        {
            filtered.push_back(config);
        }
    }
    return filtered;
}

template <typename T>
class CudaGrottoEqEvalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CudaGrottoEqEvalTest, EvalElementTypes);

TYPED_TEST(CudaGrottoEqEvalTest, EqEvalMatchesZeroTest)
{
    using T = TypeParam;

    for (const bool useStreams : {false, true})
    {
        ::FastFss::tests::cuda::StreamPair streams;
        for (const auto &widths : grotto_width_configs<T>())
        {
            for (const auto elementNum : element_counts())
            {
                auto           seeds = make_seeds<T>(elementNum, 11, 5, 19, 9);
                auto           alpha = random_vector<T>(elementNum, widths.bitWidthIn);
                auto           x     = random_vector<T>(elementNum, widths.bitWidthIn);
                std::vector<T> maskedX(elementNum);
                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    if (i % 3 == 0)
                    {
                        x[i] = T(0);
                    }
                    maskedX[i] = mod_bits<T>(alpha[i] + x[i], widths.bitWidthIn);
                }

                std::size_t keySize = 0;
                ASSERT_EQ(FastFss_grottoGetKeyDataSize(&keySize, widths.bitWidthIn, sizeof(T), elementNum),
                          FAST_FSS_SUCCESS);
                std::size_t cacheSize = 0;
                ASSERT_EQ(FastFss_grottoGetCacheDataSize(&cacheSize, widths.bitWidthIn, sizeof(T), elementNum),
                          FAST_FSS_SUCCESS);

                auto dKey = ::FastFss::tests::cuda::make_gpu_buffer(keySize);
                auto dAlpha =
                    ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(alpha.data()), alpha.size() * sizeof(T));
                auto dMaskedX = ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(maskedX.data()),
                                                                        maskedX.size() * sizeof(T));
                auto dSeed0   = ::FastFss::tests::cuda::make_gpu_buffer(seeds[0].data(), seeds[0].size());
                auto dSeed1   = ::FastFss::tests::cuda::make_gpu_buffer(seeds[1].data(), seeds[1].size());
                auto dCache   = ::FastFss::tests::cuda::make_gpu_buffer(cacheSize);

                ASSERT_EQ(
                    FastFss_cuda_grottoKeyGen(dKey.get(), keySize, dAlpha.get(), alpha.size() * sizeof(T), dSeed0.get(),
                                              seeds[0].size(), dSeed1.get(), seeds[1].size(), widths.bitWidthIn,
                                              sizeof(T), elementNum, useStreams ? streams.party_stream(0) : nullptr),
                    FAST_FSS_SUCCESS);
                if (useStreams)
                {
                    streams.synchronize(0);
                }

                auto dEq0 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
                auto dEq1 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
                ASSERT_EQ(FastFss_cuda_grottoEqEval(
                              dEq0.get(), elementNum * sizeof(T), dMaskedX.get(), maskedX.size() * sizeof(T),
                              dKey.get(), keySize, dSeed0.get(), seeds[0].size(), 0, widths.bitWidthIn, sizeof(T),
                              elementNum, dCache.get(), cacheSize, useStreams ? streams.party_stream(0) : nullptr),
                          FAST_FSS_SUCCESS);
                ASSERT_EQ(FastFss_cuda_grottoEqEval(dEq1.get(), elementNum * sizeof(T), dMaskedX.get(),
                                                    maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed1.get(),
                                                    seeds[1].size(), 1, widths.bitWidthIn, sizeof(T), elementNum,
                                                    nullptr, 0, useStreams ? streams.party_stream(1) : nullptr),
                          FAST_FSS_SUCCESS);
                if (useStreams)
                {
                    streams.synchronize_all();
                }

                std::vector<T> eq0(elementNum), eq1(elementNum);
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(eq0.data(), dEq0.get(), eq0.size() * sizeof(T));
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(eq1.data(), dEq1.get(), eq1.size() * sizeof(T));
                const auto eqOut = reconstruct_boolean(eq0, eq1);
                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    if (eqOut[i] != (x[i] == T(0) ? T(1) : T(0)))
                    {
                        FAIL() << "useStreams=" << useStreams << ", bitWidthIn=" << widths.bitWidthIn
                               << ", elementNum=" << elementNum << ", elementBits=" << (sizeof(T) * 8) << ", i=" << i;
                    }
                }
            }
        }
    }
}

} // namespace
} // namespace FastFss::tests
