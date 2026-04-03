#include <FastFss/cpu/mic.h>
#include <FastFss/errors.h>
#include <FastFss/mic.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/test_types.hpp"
#include "common/test_utils.hpp"

namespace FastFss::tests {
namespace {

template <typename T>
class CpuMicEvalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuMicEvalTest, EvalElementTypes);

TYPED_TEST(CpuMicEvalTest, EvalMatchesIntervalMembership)
{
    using T = TypeParam;

    for (const auto &widths : eval_width_configs<T>())
    {
        for (const auto elementNum : element_counts())
        {
            auto           seeds = make_seeds<T>(elementNum, 31, 3, 17, 9);
            auto           alpha = random_vector<T>(elementNum, widths.bitWidthIn);
            auto           x     = random_vector<T>(elementNum, widths.bitWidthIn);
            std::vector<T> maskedX(elementNum);
            for (std::size_t i = 0; i < elementNum; ++i)
            {
                maskedX[i] = mod_bits<T>(alpha[i] + x[i], widths.bitWidthIn);
            }

            const std::size_t intervalCount  = 3;
            auto              leftEndpoints  = random_vector<T>(intervalCount, widths.bitWidthIn);
            auto              rightEndpoints = random_vector<T>(intervalCount, widths.bitWidthIn);
            for (std::size_t i = 0; i < intervalCount; ++i)
            {
                if (rightEndpoints[i] < leftEndpoints[i])
                {
                    std::swap(leftEndpoints[i], rightEndpoints[i]);
                }
            }

            std::size_t keySize = 0;
            ASSERT_EQ(
                FastFss_dcfMICGetKeyDataSize(&keySize, widths.bitWidthIn, widths.bitWidthOut, sizeof(T), elementNum),
                FAST_FSS_SUCCESS);
            std::size_t cacheSize = 0;
            ASSERT_EQ(FastFss_dcfMICGetCacheDataSize(&cacheSize, widths.bitWidthIn, widths.bitWidthOut, sizeof(T),
                                                     elementNum),
                      FAST_FSS_SUCCESS);

            std::vector<std::uint8_t> key(keySize);
            std::vector<T>            z(elementNum * intervalCount);
            ASSERT_EQ(
                FastFss_cpu_dcfMICKeyGen(key.data(), key.size(), z.data(), z.size() * sizeof(T), alpha.data(),
                                         alpha.size() * sizeof(T), seeds[0].data(), seeds[0].size(), seeds[1].data(),
                                         seeds[1].size(), leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
                                         rightEndpoints.data(), rightEndpoints.size() * sizeof(T), widths.bitWidthIn,
                                         widths.bitWidthOut, sizeof(T), elementNum),
                FAST_FSS_SUCCESS);

            std::vector<T> sharedZ0(z.size());
            std::vector<T> sharedZ1(z.size());
            for (std::size_t i = 0; i < z.size(); ++i)
            {
                sharedZ0[i] = mod_bits<T>(random_value<T>(widths.bitWidthOut), widths.bitWidthOut);
                sharedZ1[i] = mod_bits<T>(z[i] - sharedZ0[i], widths.bitWidthOut);
            }

            std::vector<T>            share0(z.size());
            std::vector<T>            share1(z.size());
            std::vector<std::uint8_t> cache(cacheSize);

            ASSERT_EQ(FastFss_cpu_dcfMICEval(
                          share0.data(), share0.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                          key.data(), key.size(), sharedZ0.data(), sharedZ0.size() * sizeof(T), seeds[0].data(),
                          seeds[0].size(), 0, leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
                          rightEndpoints.data(), rightEndpoints.size() * sizeof(T), widths.bitWidthIn,
                          widths.bitWidthOut, sizeof(T), elementNum, cache.data(), cache.size()),
                      FAST_FSS_SUCCESS);
            ASSERT_EQ(FastFss_cpu_dcfMICEval(share1.data(), share1.size() * sizeof(T), maskedX.data(),
                                             maskedX.size() * sizeof(T), key.data(), key.size(), sharedZ1.data(),
                                             sharedZ1.size() * sizeof(T), seeds[1].data(), seeds[1].size(), 1,
                                             leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
                                             rightEndpoints.data(), rightEndpoints.size() * sizeof(T),
                                             widths.bitWidthIn, widths.bitWidthOut, sizeof(T), elementNum, nullptr, 0),
                      FAST_FSS_SUCCESS);

            for (std::size_t i = 0; i < elementNum; ++i)
            {
                for (std::size_t j = 0; j < intervalCount; ++j)
                {
                    const auto idx     = i * intervalCount + j;
                    const auto out     = mod_bits<T>(share0[idx] + share1[idx], widths.bitWidthOut);
                    const bool inRange = leftEndpoints[j] <= x[i] && x[i] <= rightEndpoints[j];
                    EXPECT_EQ(out, inRange ? T(1) : T(0));
                }
            }
        }
    }
}

} // namespace
} // namespace FastFss::tests
