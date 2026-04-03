#include <FastFss/cpu/grotto.h>
#include <FastFss/errors.h>
#include <FastFss/grotto.h>
#include <gtest/gtest.h>

#include <vector>

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
class CpuGrottoMicEvalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuGrottoMicEvalTest, EvalElementTypes);

TYPED_TEST(CpuGrottoMicEvalTest, MICEvalMatchesIntervals)
{
    using T = TypeParam;

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
            std::vector<std::uint8_t> key(keySize);
            ASSERT_EQ(FastFss_cpu_grottoKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T),
                                               seeds[0].data(), seeds[0].size(), seeds[1].data(), seeds[1].size(),
                                               widths.bitWidthIn, sizeof(T), elementNum),
                      FAST_FSS_SUCCESS);

            std::vector<T>            share0(elementNum * intervalCount);
            std::vector<T>            share1(elementNum * intervalCount);
            std::vector<std::uint8_t> cache(cacheSize);
            ASSERT_EQ(FastFss_cpu_grottoMICEval(share0.data(), share0.size() * sizeof(T), maskedX.data(),
                                                maskedX.size() * sizeof(T), key.data(), key.size(), seeds[0].data(),
                                                seeds[0].size(), 0, left.data(), left.size() * sizeof(T), right.data(),
                                                right.size() * sizeof(T), widths.bitWidthIn, sizeof(T), elementNum,
                                                cache.data(), cache.size()),
                      FAST_FSS_SUCCESS);
            ASSERT_EQ(FastFss_cpu_grottoMICEval(share1.data(), share1.size() * sizeof(T), maskedX.data(),
                                                maskedX.size() * sizeof(T), key.data(), key.size(), seeds[1].data(),
                                                seeds[1].size(), 1, left.data(), left.size() * sizeof(T), right.data(),
                                                right.size() * sizeof(T), widths.bitWidthIn, sizeof(T), elementNum,
                                                nullptr, 0),
                      FAST_FSS_SUCCESS);

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

} // namespace
} // namespace FastFss::tests
