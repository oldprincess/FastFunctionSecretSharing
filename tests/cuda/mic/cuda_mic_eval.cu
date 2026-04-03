#include <FastFss/cuda/mic.h>
#include <FastFss/errors.h>
#include <FastFss/mic.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/cuda_test_utils.cuh"
#include "common/test_types.hpp"
#include "common/test_utils.hpp"

namespace FastFss::tests {
namespace {

template <typename T>
class CudaMicEvalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CudaMicEvalTest, EvalElementTypes);

TYPED_TEST(CudaMicEvalTest, EvalMatchesIntervalMembership)
{
    using T = TypeParam;

    for (const bool useStreams : {false, true})
    {
        ::FastFss::tests::cuda::StreamPair streams;
        for (const auto &widths : eval_width_configs<T>())
        {
            for (const auto elementNum : element_counts())
            {
                constexpr std::size_t intervalCount = 3;
                auto                  seeds         = make_seeds<T>(elementNum, 29, 7, 11, 5);
                auto                  alpha         = random_vector<T>(elementNum, widths.bitWidthIn);
                auto                  x             = random_vector<T>(elementNum, widths.bitWidthIn);
                std::vector<T>        maskedX(elementNum);
                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    maskedX[i] = mod_bits<T>(alpha[i] + x[i], widths.bitWidthIn);
                }
                auto left  = random_vector<T>(intervalCount, widths.bitWidthIn);
                auto right = random_vector<T>(intervalCount, widths.bitWidthIn);
                for (std::size_t i = 0; i < intervalCount; ++i)
                {
                    if (right[i] < left[i])
                    {
                        std::swap(left[i], right[i]);
                    }
                }

                std::size_t keySize = 0;
                ASSERT_EQ(FastFss_dcfMICGetKeyDataSize(&keySize, widths.bitWidthIn, widths.bitWidthOut, sizeof(T),
                                                       elementNum),
                          FAST_FSS_SUCCESS);
                std::size_t cacheSize = 0;
                ASSERT_EQ(FastFss_dcfMICGetCacheDataSize(&cacheSize, widths.bitWidthIn, widths.bitWidthOut, sizeof(T),
                                                         elementNum),
                          FAST_FSS_SUCCESS);

                auto dKey = ::FastFss::tests::cuda::make_gpu_buffer(keySize);
                auto dZ   = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * intervalCount * sizeof(T));
                auto dAlpha =
                    ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(alpha.data()), alpha.size() * sizeof(T));
                auto dMaskedX = ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(maskedX.data()),
                                                                        maskedX.size() * sizeof(T));
                auto dSeed0   = ::FastFss::tests::cuda::make_gpu_buffer(seeds[0].data(), seeds[0].size());
                auto dSeed1   = ::FastFss::tests::cuda::make_gpu_buffer(seeds[1].data(), seeds[1].size());
                auto dLeft =
                    ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(left.data()), left.size() * sizeof(T));
                auto dRight =
                    ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(right.data()), right.size() * sizeof(T));
                auto dCache = ::FastFss::tests::cuda::make_gpu_buffer(cacheSize);

                ASSERT_EQ(
                    FastFss_cuda_dcfMICKeyGen(
                        dKey.get(), keySize, dZ.get(), elementNum * intervalCount * sizeof(T), dAlpha.get(),
                        alpha.size() * sizeof(T), dSeed0.get(), seeds[0].size(), dSeed1.get(), seeds[1].size(),
                        dLeft.get(), left.size() * sizeof(T), dRight.get(), right.size() * sizeof(T), widths.bitWidthIn,
                        widths.bitWidthOut, sizeof(T), elementNum, useStreams ? streams.party_stream(0) : nullptr),
                    FAST_FSS_SUCCESS);

                std::vector<T> z(elementNum * intervalCount);
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(z.data(), dZ.get(), z.size() * sizeof(T));
                std::vector<T> sharedZ0(z.size());
                std::vector<T> sharedZ1(z.size());
                for (std::size_t i = 0; i < z.size(); ++i)
                {
                    sharedZ0[i] = mod_bits<T>(random_value<T>(widths.bitWidthOut), widths.bitWidthOut);
                    sharedZ1[i] = mod_bits<T>(z[i] - sharedZ0[i], widths.bitWidthOut);
                }

                auto dSharedZ0 = ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(sharedZ0.data()),
                                                                         sharedZ0.size() * sizeof(T));
                auto dSharedZ1 = ::FastFss::tests::cuda::make_gpu_buffer(const_cast<T *>(sharedZ1.data()),
                                                                         sharedZ1.size() * sizeof(T));
                auto dOut0     = ::FastFss::tests::cuda::make_gpu_buffer(z.size() * sizeof(T));
                auto dOut1     = ::FastFss::tests::cuda::make_gpu_buffer(z.size() * sizeof(T));

                ASSERT_EQ(FastFss_cuda_dcfMICEval(
                              dOut0.get(), z.size() * sizeof(T), dMaskedX.get(), maskedX.size() * sizeof(T), dKey.get(),
                              keySize, dSharedZ0.get(), sharedZ0.size() * sizeof(T), dSeed0.get(), seeds[0].size(), 0,
                              dLeft.get(), left.size() * sizeof(T), dRight.get(), right.size() * sizeof(T),
                              widths.bitWidthIn, widths.bitWidthOut, sizeof(T), elementNum, dCache.get(), cacheSize,
                              useStreams ? streams.party_stream(0) : nullptr),
                          FAST_FSS_SUCCESS);
                ASSERT_EQ(FastFss_cuda_dcfMICEval(
                              dOut1.get(), z.size() * sizeof(T), dMaskedX.get(), maskedX.size() * sizeof(T), dKey.get(),
                              keySize, dSharedZ1.get(), sharedZ1.size() * sizeof(T), dSeed1.get(), seeds[1].size(), 1,
                              dLeft.get(), left.size() * sizeof(T), dRight.get(), right.size() * sizeof(T),
                              widths.bitWidthIn, widths.bitWidthOut, sizeof(T), elementNum, nullptr, 0,
                              useStreams ? streams.party_stream(1) : nullptr),
                          FAST_FSS_SUCCESS);

                std::vector<T> share0(z.size());
                std::vector<T> share1(z.size());
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(share0.data(), dOut0.get(), share0.size() * sizeof(T));
                ::FastFss::tests::cuda::memcpy_gpu_to_cpu(share1.data(), dOut1.get(), share1.size() * sizeof(T));

                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    for (std::size_t j = 0; j < intervalCount; ++j)
                    {
                        const auto idx     = i * intervalCount + j;
                        const auto out     = mod_bits<T>(share0[idx] + share1[idx], widths.bitWidthOut);
                        const bool inRange = left[j] <= x[i] && x[i] <= right[j];
                        EXPECT_EQ(out, inRange ? T(1) : T(0));
                    }
                }
            }
        }
    }
}

} // namespace
} // namespace FastFss::tests
