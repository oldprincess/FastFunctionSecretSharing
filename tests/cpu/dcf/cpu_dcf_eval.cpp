#include <FastFss/cpu/dcf.h>
#include <FastFss/dcf.h>
#include <FastFss/errors.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/test_types.hpp"
#include "common/test_utils.hpp"

namespace FastFss::tests {
namespace {

template <typename T>
class CpuDcfEvalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuDcfEvalTest, EvalElementTypes);

TYPED_TEST(CpuDcfEvalTest, EvalMatchesExpectedRandomizedCases)
{
    using T                          = TypeParam;
    constexpr std::size_t kGroupSize = 3;

    for (const auto &widths : eval_width_configs<T>())
    {
        for (const auto elementNum : element_counts())
        {
            auto seeds = make_seeds<T>(elementNum, 17, 3, 29, 7);

            auto           alpha   = random_vector<T>(elementNum, widths.bitWidthIn);
            auto           maskedX = random_vector<T>(elementNum, widths.bitWidthIn);
            auto           beta    = random_vector<T>(elementNum * kGroupSize, widths.bitWidthOut);
            std::vector<T> expected(elementNum * kGroupSize, T(0));

            for (std::size_t i = 0; i < elementNum; ++i)
            {
                if (maskedX[i] < alpha[i])
                {
                    for (std::size_t j = 0; j < kGroupSize; ++j)
                    {
                        expected[i * kGroupSize + j] = mod_bits<T>(beta[i * kGroupSize + j], widths.bitWidthOut);
                    }
                }
            }

            std::size_t keySize = 0;
            ASSERT_EQ(FastFss_dcfGetKeyDataSize(&keySize, widths.bitWidthIn, widths.bitWidthOut, kGroupSize, sizeof(T),
                                                elementNum),
                      FAST_FSS_SUCCESS);
            std::size_t cacheSize = 0;
            ASSERT_EQ(FastFss_dcfGetCacheDataSize(&cacheSize, widths.bitWidthIn, widths.bitWidthOut, kGroupSize,
                                                  sizeof(T), elementNum),
                      FAST_FSS_SUCCESS);

            std::vector<std::uint8_t> key(keySize);
            ASSERT_EQ(FastFss_cpu_dcfKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), beta.data(),
                                            beta.size() * sizeof(T), seeds[0].data(), seeds[0].size(), seeds[1].data(),
                                            seeds[1].size(), widths.bitWidthIn, widths.bitWidthOut, kGroupSize,
                                            sizeof(T), elementNum),
                      FAST_FSS_SUCCESS);

            std::vector<T>            share0(elementNum * kGroupSize);
            std::vector<T>            share1(elementNum * kGroupSize);
            std::vector<std::uint8_t> cache(cacheSize);

            ASSERT_EQ(FastFss_cpu_dcfEval(share0.data(), share0.size() * sizeof(T), maskedX.data(),
                                          maskedX.size() * sizeof(T), key.data(), key.size(), seeds[0].data(),
                                          seeds[0].size(), 0, widths.bitWidthIn, widths.bitWidthOut, kGroupSize,
                                          sizeof(T), elementNum, cache.data(), cache.size()),
                      FAST_FSS_SUCCESS);
            ASSERT_EQ(FastFss_cpu_dcfEval(share1.data(), share1.size() * sizeof(T), maskedX.data(),
                                          maskedX.size() * sizeof(T), key.data(), key.size(), seeds[1].data(),
                                          seeds[1].size(), 1, widths.bitWidthIn, widths.bitWidthOut, kGroupSize,
                                          sizeof(T), elementNum, nullptr, 0),
                      FAST_FSS_SUCCESS);

            EXPECT_EQ(reconstruct_arithmetic(share0, share1, widths.bitWidthOut), expected);
        }
    }
}

} // namespace
} // namespace FastFss::tests
