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
class CudaGrottoMicEvalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CudaGrottoMicEvalTest, EvalElementTypes);

TYPED_TEST(CudaGrottoMicEvalTest, MICEvalMatchesIntervals)
{
    using T = TypeParam;

    for (const bool useStreams : {false, true})
    {
        ::FastFss::tests::cuda::StreamPair streams;
        for (const auto &widths : grotto_width_configs<T>())
        {
            for (const auto elementNum : element_counts())
            {
                auto           seeds = make_seeds<T>(elementNum, 37, 3, 41, 9);
                auto           alpha = random_vector<T>(elementNum, widths.bitWidthIn);
                auto           x     = random_vector<T>(elementNum, widths.bitWidthIn);
                std::vector<T> maskedX(elementNum);
                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    maskedX[i] = mod_bits<T>(alpha[i] + x[i], widths.bitWidthIn);
                }

                const std::size_t intervalCount = 3;
                auto              left          = random_vector<T>(intervalCount, widths.bitWidthIn);
                auto              right         = random_vector<T>(intervalCount, widths.bitWidthIn);
                for (std::size_t i = 0; i < intervalCount; ++i)
                {
                    if (right[i] < left[i])
                    {
                        std::swap(left[i], right[i]);
                    }
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
                auto dLeft    = ::FastFss::tests::cuda::make_gpu_buffer(left.data(), left.size() * sizeof(T));
                auto dRight   = ::FastFss::tests::cuda::make_gpu_buffer(right.data(), right.size() * sizeof(T));
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

                auto dShare0 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * intervalCount * sizeof(T));
                auto dShare1 = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * intervalCount * sizeof(T));
                ASSERT_EQ(
                    FastFss_cuda_grottoMICEval(dShare0.get(), elementNum * intervalCount * sizeof(T), dMaskedX.get(),
                                               maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed0.get(),
                                               seeds[0].size(), 0, dLeft.get(), left.size() * sizeof(T), dRight.get(),
                                               right.size() * sizeof(T), widths.bitWidthIn, sizeof(T), elementNum,
                                               dCache.get(), cacheSize, useStreams ? streams.party_stream(0) : nullptr),
                    FAST_FSS_SUCCESS);
                ASSERT_EQ(
                    FastFss_cuda_grottoMICEval(dShare1.get(), elementNum * intervalCount * sizeof(T), dMaskedX.get(),
                                               maskedX.size() * sizeof(T), dKey.get(), keySize, dSeed1.get(),
                                               seeds[1].size(), 1, dLeft.get(), left.size() * sizeof(T), dRight.get(),
                                               right.size() * sizeof(T), widths.bitWidthIn, sizeof(T), elementNum,
                                               nullptr, 0, useStreams ? streams.party_stream(1) : nullptr),
                    FAST_FSS_SUCCESS);
                if (useStreams)
                {
                    streams.synchronize_all();
                }

                std::vector<T> share0(elementNum * intervalCount), share1(elementNum * intervalCount);
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(share0.data(), dShare0.get(), share0.size() * sizeof(T));
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(share1.data(), dShare1.get(), share1.size() * sizeof(T));

                const auto out = reconstruct_boolean(share0, share1);
                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    for (std::size_t j = 0; j < intervalCount; ++j)
                    {
                        const bool inRange = left[j] <= x[i] && x[i] <= right[j];
                        EXPECT_EQ(out[i * intervalCount + j], inRange ? T(1) : T(0));
                    }
                }
            }
        }
    }
}

} // namespace
} // namespace FastFss::tests
