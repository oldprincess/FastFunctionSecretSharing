// clang-format off
// g++ -I include -I third_party/googletest/googletest/include -I third_party/googletest/googletest src/cpu/ottt.cpp src/cpu/config.cpp test/cpu/ottt.cpp third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cpu_ottt.exe -std=c++17 -fopenmp
// clang-format on
#include <FastFss/cpu/ottt.h>
#include <FastFss/errors.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "wideint/wideint.hpp"

namespace {

using uint128_t = wideint::uint<2>;

template <typename T>
constexpr T modBits(T x, std::size_t bitWidth) noexcept
{
    if (bitWidth == sizeof(T) * 8)
    {
        return x;
    }
    return x & ((T(1) << static_cast<unsigned int>(bitWidth)) - T(1));
}

template <typename T>
class OtttCpuTypedTest : public ::testing::Test
{
};

using OtttElementTypes = ::testing::
    Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, uint128_t>;

TYPED_TEST_SUITE(OtttCpuTypedTest, OtttElementTypes);

TYPED_TEST(OtttCpuTypedTest, LutEvalReconstructsLookupValue)
{
    using T = TypeParam;
    const std::size_t bitWidthIn =
        sizeof(T) == 1 ? 7 : (sizeof(T) == 2 ? 9 : 9);
    const std::size_t bitWidthOut = sizeof(T) == 16 ? 127 : sizeof(T) * 8;
    const std::size_t elementNum  = 8;

    std::size_t keyDataSize = 0;
    ASSERT_EQ(
        FastFss_cpu_otttGetKeyDataSize(&keyDataSize, bitWidthIn, elementNum),
        FAST_FSS_SUCCESS);

    std::vector<T> alpha = {T(3), T(5), T(7), T(9), T(11), T(13), T(15), T(17)};
    std::vector<T> x     = {T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)};
    std::vector<T> maskedX(elementNum);
    for (std::size_t i = 0; i < elementNum; ++i)
    {
        alpha[i]   = modBits<T>(alpha[i], bitWidthIn);
        x[i]       = modBits<T>(x[i], bitWidthIn);
        maskedX[i] = modBits<T>(x[i] + alpha[i], bitWidthIn);
    }

    std::vector<T> lut(1ULL << bitWidthIn);
    for (std::size_t i = 0; i < lut.size(); ++i)
    {
        lut[i] = static_cast<T>(i);
    }

    std::vector<std::uint8_t> key0(keyDataSize);
    std::vector<std::uint8_t> key1(keyDataSize);
    for (std::size_t i = 0; i < key0.size(); ++i)
    {
        key0[i] = static_cast<std::uint8_t>((7 * i + 3) & 0xff);
        key1[i] = key0[i];
    }
    ASSERT_EQ(FastFss_cpu_otttKeyGen(key1.data(), key1.size(), alpha.data(),
                                     alpha.size() * sizeof(T), bitWidthIn,
                                     sizeof(T), elementNum),
              FAST_FSS_SUCCESS);

    std::vector<T> shareE0(elementNum), shareE1(elementNum);
    std::vector<T> shareT0(elementNum), shareT1(elementNum);
    ASSERT_EQ(
        FastFss_cpu_otttLutEval(
            shareE0.data(), shareE0.size() * sizeof(T), shareT0.data(),
            shareT0.size() * sizeof(T), maskedX.data(),
            maskedX.size() * sizeof(T), key0.data(), key0.size(), 0, lut.data(),
            lut.size() * sizeof(T), bitWidthIn, sizeof(T), elementNum),
        FAST_FSS_SUCCESS);
    ASSERT_EQ(
        FastFss_cpu_otttLutEval(
            shareE1.data(), shareE1.size() * sizeof(T), shareT1.data(),
            shareT1.size() * sizeof(T), maskedX.data(),
            maskedX.size() * sizeof(T), key1.data(), key1.size(), 1, lut.data(),
            lut.size() * sizeof(T), bitWidthIn, sizeof(T), elementNum),
        FAST_FSS_SUCCESS);

    for (std::size_t i = 0; i < elementNum; ++i)
    {
        const T e   = modBits<T>(shareE0[i] + shareE1[i], 1);
        const T t   = modBits<T>(shareT0[i] + shareT1[i], bitWidthOut);
        const T out = modBits<T>(e == 0 ? t : (T(0) - t), bitWidthOut);
        EXPECT_EQ(out, lut[static_cast<std::size_t>(x[i])]);
    }
}

} // namespace
