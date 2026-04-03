#include <FastFss/cpu/ottt.h>
#include <FastFss/errors.h>
#include <FastFss/ottt.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/test_types.hpp"
#include "common/test_utils.hpp"

namespace FastFss::tests {
namespace {

template <typename T>
class CpuOtttLutEvalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuOtttLutEvalTest, EvalElementTypes);

TYPED_TEST(CpuOtttLutEvalTest, LutEvalReconstructsLookupValue)
{
    using T                           = TypeParam;
    constexpr std::size_t kBitWidthIn = 7;

    for (const auto elementNum : element_counts())
    {
        std::size_t keySize = 0;
        ASSERT_EQ(FastFss_otttGetKeyDataSize(&keySize, kBitWidthIn, elementNum), FAST_FSS_SUCCESS);

        auto           alpha = random_vector<T>(elementNum, kBitWidthIn);
        auto           x     = random_vector<T>(elementNum, kBitWidthIn);
        std::vector<T> maskedX(elementNum);
        for (std::size_t i = 0; i < elementNum; ++i)
        {
            maskedX[i] = mod_bits<T>(alpha[i] + x[i], kBitWidthIn);
        }

        std::vector<T> lut(1ULL << kBitWidthIn);
        for (std::size_t i = 0; i < lut.size(); ++i)
        {
            lut[i] = mod_bits<T>(T(i) * T(3) + T(1), bit_size_v<T>);
        }

        std::vector<std::uint8_t> key0(keySize);
        std::vector<std::uint8_t> key1(keySize, 0);
        for (std::size_t i = 0; i < key0.size(); ++i)
        {
            key0[i] = static_cast<std::uint8_t>((7 * i + 3) & 0xffU);
            key1[i] = key0[i];
        }
        ASSERT_EQ(FastFss_cpu_otttKeyGen(key1.data(), key1.size(), alpha.data(), alpha.size() * sizeof(T), kBitWidthIn,
                                         sizeof(T), elementNum),
                  FAST_FSS_SUCCESS);

        std::vector<T> shareE0(elementNum), shareE1(elementNum);
        std::vector<T> shareT0(elementNum), shareT1(elementNum);
        ASSERT_EQ(FastFss_cpu_otttLutEval(shareE0.data(), shareE0.size() * sizeof(T), shareT0.data(),
                                          shareT0.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                                          key0.data(), key0.size(), 0, lut.data(), lut.size() * sizeof(T), kBitWidthIn,
                                          sizeof(T), elementNum),
                  FAST_FSS_SUCCESS);
        ASSERT_EQ(FastFss_cpu_otttLutEval(shareE1.data(), shareE1.size() * sizeof(T), shareT1.data(),
                                          shareT1.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                                          key1.data(), key1.size(), 1, lut.data(), lut.size() * sizeof(T), kBitWidthIn,
                                          sizeof(T), elementNum),
                  FAST_FSS_SUCCESS);

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            const T e   = mod_bits<T>(shareE0[i] + shareE1[i], 1);
            const T t   = mod_bits<T>(shareT0[i] + shareT1[i], bit_size_v<T>);
            const T out = mod_bits<T>(e == 0 ? t : (T(0) - t), bit_size_v<T>);
            EXPECT_EQ(out, lut[static_cast<std::size_t>(x[i])]);
        }
    }
}

} // namespace
} // namespace FastFss::tests
