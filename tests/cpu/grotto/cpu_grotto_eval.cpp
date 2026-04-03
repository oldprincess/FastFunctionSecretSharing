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
class CpuGrottoEvalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuGrottoEvalTest, EvalElementTypes);

TYPED_TEST(CpuGrottoEvalTest, EvalMatchesStrictAndClosedBounds)
{
    using T = TypeParam;

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

            std::vector<std::uint8_t> key(keySize);
            ASSERT_EQ(FastFss_cpu_grottoKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T),
                                               seeds[0].data(), seeds[0].size(), seeds[1].data(), seeds[1].size(),
                                               widths.bitWidthIn, sizeof(T), elementNum),
                      FAST_FSS_SUCCESS);

            std::vector<T>            lt0(elementNum), lt1(elementNum);
            std::vector<T>            le0(elementNum), le1(elementNum);
            std::vector<std::uint8_t> cache(cacheSize);

            ASSERT_EQ(
                FastFss_cpu_grottoEval(lt0.data(), lt0.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                                       key.data(), key.size(), seeds[0].data(), seeds[0].size(), false, 0,
                                       widths.bitWidthIn, sizeof(T), elementNum, cache.data(), cache.size()),
                FAST_FSS_SUCCESS);
            ASSERT_EQ(
                FastFss_cpu_grottoEval(lt1.data(), lt1.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                                       key.data(), key.size(), seeds[1].data(), seeds[1].size(), false, 1,
                                       widths.bitWidthIn, sizeof(T), elementNum, nullptr, 0),
                FAST_FSS_SUCCESS);
            ASSERT_EQ(
                FastFss_cpu_grottoEval(le0.data(), le0.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                                       key.data(), key.size(), seeds[0].data(), seeds[0].size(), true, 0,
                                       widths.bitWidthIn, sizeof(T), elementNum, nullptr, 0),
                FAST_FSS_SUCCESS);
            ASSERT_EQ(
                FastFss_cpu_grottoEval(le1.data(), le1.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                                       key.data(), key.size(), seeds[1].data(), seeds[1].size(), true, 1,
                                       widths.bitWidthIn, sizeof(T), elementNum, nullptr, 0),
                FAST_FSS_SUCCESS);

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

} // namespace
} // namespace FastFss::tests
