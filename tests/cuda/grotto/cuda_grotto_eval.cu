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
class CudaGrottoEvalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CudaGrottoEvalTest, EvalElementTypes);

TYPED_TEST(CudaGrottoEvalTest, EvalMatchesStrictAndClosedBounds)
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

                auto dLt0 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
                auto dLt1 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
                auto dLe0 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
                auto dLe1 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
                ASSERT_EQ(
                    FastFss_cuda_grottoEval(dLt0.get(), elementNum * sizeof(T), dMaskedX.get(),
                                            maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed0.get(),
                                            seeds[0].size(), false, 0, widths.bitWidthIn, sizeof(T), elementNum,
                                            dCache.get(), cacheSize, useStreams ? streams.party_stream(0) : nullptr),
                    FAST_FSS_SUCCESS);
                ASSERT_EQ(FastFss_cuda_grottoEval(dLt1.get(), elementNum * sizeof(T), dMaskedX.get(),
                                                  maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed1.get(),
                                                  seeds[1].size(), false, 1, widths.bitWidthIn, sizeof(T), elementNum,
                                                  nullptr, 0, useStreams ? streams.party_stream(1) : nullptr),
                          FAST_FSS_SUCCESS);
                ASSERT_EQ(FastFss_cuda_grottoEval(dLe0.get(), elementNum * sizeof(T), dMaskedX.get(),
                                                  maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed0.get(),
                                                  seeds[0].size(), true, 0, widths.bitWidthIn, sizeof(T), elementNum,
                                                  nullptr, 0, useStreams ? streams.party_stream(0) : nullptr),
                          FAST_FSS_SUCCESS);
                ASSERT_EQ(FastFss_cuda_grottoEval(dLe1.get(), elementNum * sizeof(T), dMaskedX.get(),
                                                  maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed1.get(),
                                                  seeds[1].size(), true, 1, widths.bitWidthIn, sizeof(T), elementNum,
                                                  nullptr, 0, useStreams ? streams.party_stream(1) : nullptr),
                          FAST_FSS_SUCCESS);
                if (useStreams)
                {
                    streams.synchronize_all();
                }

                std::vector<T> lt0(elementNum), lt1(elementNum), le0(elementNum), le1(elementNum);
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(lt0.data(), dLt0.get(), lt0.size() * sizeof(T));
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(lt1.data(), dLt1.get(), lt1.size() * sizeof(T));
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(le0.data(), dLe0.get(), le0.size() * sizeof(T));
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(le1.data(), dLe1.get(), le1.size() * sizeof(T));
                const auto ltOut = reconstruct_boolean(lt0, lt1);
                const auto leOut = reconstruct_boolean(le0, le1);
                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    EXPECT_EQ(ltOut[i], maskedX[i] < alpha[i] ? T(1) : T(0));
                    EXPECT_EQ(leOut[i], maskedX[i] <= alpha[i] ? T(1) : T(0));
                }
            }
        }
    }
}

} // namespace
} // namespace FastFss::tests
