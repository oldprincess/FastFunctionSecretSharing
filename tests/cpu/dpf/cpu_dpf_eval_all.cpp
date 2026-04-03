#include <FastFss/cpu/dpf.h>
#include <FastFss/dpf.h>
#include <FastFss/errors.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/test_types.hpp"
#include "common/test_utils.hpp"

namespace FastFss::tests {
namespace {

template <typename T>
class CpuDpfEvalAllTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuDpfEvalAllTest, EvalElementTypes);

TYPED_TEST(CpuDpfEvalAllTest, EvalAllPlacesBetaAtSelectedPoint)
{
    using T                          = TypeParam;
    constexpr std::size_t kGroupSize = 3;

    for (const auto &widths : eval_width_configs<T>())
    {
        if (widths.bitWidthIn > 10)
        {
            continue;
        }
        for (const auto elementNum : element_counts())
        {
            auto seeds = make_seeds<T>(elementNum, 13, 1, 23, 7);

            auto           alpha = random_vector<T>(elementNum, widths.bitWidthIn);
            auto           x     = random_vector<T>(elementNum, widths.bitWidthIn);
            auto           beta  = random_vector<T>(elementNum * kGroupSize, widths.bitWidthOut);
            std::vector<T> maskedX(elementNum);
            for (std::size_t i = 0; i < elementNum; ++i)
            {
                maskedX[i] = mod_bits<T>(alpha[i] + x[i], widths.bitWidthIn);
            }

            std::size_t keySize = 0;
            ASSERT_EQ(FastFss_dpfGetKeyDataSize(&keySize, widths.bitWidthIn, widths.bitWidthOut, kGroupSize, sizeof(T),
                                                elementNum),
                      FAST_FSS_SUCCESS);
            std::size_t cacheSize = 0;
            ASSERT_EQ(FastFss_dpfGetCacheDataSize(&cacheSize, widths.bitWidthIn, sizeof(T), elementNum),
                      FAST_FSS_SUCCESS);

            std::vector<std::uint8_t> key(keySize);
            ASSERT_EQ(FastFss_cpu_dpfKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), beta.data(),
                                            beta.size() * sizeof(T), seeds[0].data(), seeds[0].size(), seeds[1].data(),
                                            seeds[1].size(), widths.bitWidthIn, widths.bitWidthOut, kGroupSize,
                                            sizeof(T), elementNum),
                      FAST_FSS_SUCCESS);

            const std::size_t         domainSize = std::size_t{1} << widths.bitWidthIn;
            std::vector<T>            share0(elementNum * domainSize * kGroupSize);
            std::vector<T>            share1(elementNum * domainSize * kGroupSize);
            std::vector<std::uint8_t> cache(cacheSize);

            ASSERT_EQ(FastFss_cpu_dpfEvalAll(share0.data(), share0.size() * sizeof(T), maskedX.data(),
                                             maskedX.size() * sizeof(T), key.data(), key.size(), seeds[0].data(),
                                             seeds[0].size(), 0, widths.bitWidthIn, widths.bitWidthOut, kGroupSize,
                                             sizeof(T), elementNum, cache.data(), cache.size()),
                      FAST_FSS_SUCCESS);
            ASSERT_EQ(FastFss_cpu_dpfEvalAll(share1.data(), share1.size() * sizeof(T), maskedX.data(),
                                             maskedX.size() * sizeof(T), key.data(), key.size(), seeds[1].data(),
                                             seeds[1].size(), 1, widths.bitWidthIn, widths.bitWidthOut, kGroupSize,
                                             sizeof(T), elementNum, nullptr, 0),
                      FAST_FSS_SUCCESS);

            const auto reconstructed = reconstruct_arithmetic(share0, share1, widths.bitWidthOut);
            for (std::size_t i = 0; i < elementNum; ++i)
            {
                for (std::size_t point = 0; point < domainSize; ++point)
                {
                    for (std::size_t j = 0; j < kGroupSize; ++j)
                    {
                        const std::size_t index    = i * domainSize * kGroupSize + point * kGroupSize + j;
                        const T           expected = point == static_cast<std::size_t>(x[i])
                                                         ? mod_bits<T>(beta[i * kGroupSize + j], widths.bitWidthOut)
                                                         : T(0);
                        EXPECT_EQ(reconstructed[index], expected);
                    }
                }
            }
        }
    }
}

} // namespace
} // namespace FastFss::tests
