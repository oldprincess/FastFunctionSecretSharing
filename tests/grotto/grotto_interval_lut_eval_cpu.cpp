#include <FastFssPP/grotto.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "grotto_interval_lut_eval_test_case.h"

namespace {

template <typename T>
T bit_mask(std::size_t bitWidth)
{
    return (bitWidth == sizeof(T) * 8) ? ~T(0) : (T(1) << bitWidth) - T(1);
}

} // namespace

namespace FastFss::tests::grotto {

struct CpuGrottoIntervalLutEvalTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
    std::size_t intervalCount;
    std::size_t lutNum;
};

template <typename T>
class CpuGrottoIntervalLutEvalTestBase : public ::testing::Test,
                                         public ::testing::WithParamInterface<CpuGrottoIntervalLutEvalTestParams>
{
public:
    using Type = T;
};

class CpuGrottoIntervalLutEvalTestU8 : public CpuGrottoIntervalLutEvalTestBase<std::uint8_t>
{
};
class CpuGrottoIntervalLutEvalTestU16 : public CpuGrottoIntervalLutEvalTestBase<std::uint16_t>
{
};
class CpuGrottoIntervalLutEvalTestU32 : public CpuGrottoIntervalLutEvalTestBase<std::uint32_t>
{
};
class CpuGrottoIntervalLutEvalTestU64 : public CpuGrottoIntervalLutEvalTestBase<std::uint64_t>
{
};
class CpuGrottoIntervalLutEvalTestU128 : public CpuGrottoIntervalLutEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCpuGrottoIntervalLutEvalTestBody(Fixture* self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 17};

    using T = typename Fixture::Type;

    CpuGrottoIntervalLutEvalTestParams params = self->GetParam();

    const std::size_t bitWidthIn    = params.bitWidthIn;
    const std::size_t bitWidthOut   = params.bitWidthOut;
    const std::size_t intervalCount = params.intervalCount;
    const std::size_t lutNum        = params.lutNum;
    const std::size_t elementSize   = sizeof(T);
    const T           maskOut       = bit_mask<T>(bitWidthOut);

    for (std::size_t elementNum : kElementNumConfigs)
    {
        GrottoIntervalLutEvalTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, intervalCount, lutNum);

        std::size_t keySize   = FastFss::grotto::grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum);
        std::size_t cacheSize = FastFss::grotto::grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);

        std::vector<std::uint8_t> key(keySize);
        FastFss::grotto::cpu::grottoKeyGen<T>(
            std::span<std::uint8_t>(key.data(), key.size()),
            std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
            std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
            std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), bitWidthIn);

        std::vector<T>            shareE0(elementNum), shareE1(elementNum);
        std::vector<T>            shareT0(elementNum * lutNum), shareT1(elementNum * lutNum);
        std::vector<std::uint8_t> evalCache(cacheSize);
        FastFss::grotto::cpu::grottoIntervalLutEval<T>(
            std::span<T>(shareE0.data(), shareE0.size()), std::span<T>(shareT0.data(), shareT0.size()),
            std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
            std::span<const std::uint8_t>(key.data(), key.size()),
            std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), 0,
            std::span<const T>(testCase.left.data(), testCase.left.size()),
            std::span<const T>(testCase.right.data(), testCase.right.size()),
            std::span<const T>(testCase.table.data(), testCase.table.size()), bitWidthIn, bitWidthOut,
            std::span<std::uint8_t>(evalCache.data(), evalCache.size()));
        FastFss::grotto::cpu::grottoIntervalLutEval<T>(
            std::span<T>(shareE1.data(), shareE1.size()), std::span<T>(shareT1.data(), shareT1.size()),
            std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
            std::span<const std::uint8_t>(key.data(), key.size()),
            std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), 1,
            std::span<const T>(testCase.left.data(), testCase.left.size()),
            std::span<const T>(testCase.right.data(), testCase.right.size()),
            std::span<const T>(testCase.table.data(), testCase.table.size()), bitWidthIn, bitWidthOut,
            std::span<std::uint8_t>{});

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            for (std::size_t j = 0; j < lutNum; ++j)
            {
                T e   = shareE0[i] + shareE1[i];
                T t   = (shareT0[i * lutNum + j] + shareT1[i * lutNum + j]) & maskOut;
                T out = (e * t) & maskOut;
                ASSERT_EQ(out, testCase.expectedT[i * lutNum + j]);
            }
        }
    }
}

TEST_P(CpuGrottoIntervalLutEvalTestU8, RunTestBody)
{
    RunCpuGrottoIntervalLutEvalTestBody(this);
}
TEST_P(CpuGrottoIntervalLutEvalTestU16, RunTestBody)
{
    RunCpuGrottoIntervalLutEvalTestBody(this);
}
TEST_P(CpuGrottoIntervalLutEvalTestU32, RunTestBody)
{
    RunCpuGrottoIntervalLutEvalTestBody(this);
}
TEST_P(CpuGrottoIntervalLutEvalTestU64, RunTestBody)
{
    RunCpuGrottoIntervalLutEvalTestBody(this);
}
TEST_P(CpuGrottoIntervalLutEvalTestU128, RunTestBody)
{
    RunCpuGrottoIntervalLutEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoIntervalLutEvalTestU8,
                         ::testing::Values(CpuGrottoIntervalLutEvalTestParams{6, 8, 3, 2},
                                           CpuGrottoIntervalLutEvalTestParams{7, 8, 3, 2},
                                           CpuGrottoIntervalLutEvalTestParams{8, 8, 3, 2}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoIntervalLutEvalTestU16,
                         ::testing::Values(CpuGrottoIntervalLutEvalTestParams{6, 16, 3, 2},
                                           CpuGrottoIntervalLutEvalTestParams{15, 16, 3, 2},
                                           CpuGrottoIntervalLutEvalTestParams{16, 16, 3, 2}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoIntervalLutEvalTestU32,
                         ::testing::Values(CpuGrottoIntervalLutEvalTestParams{6, 32, 3, 2},
                                           CpuGrottoIntervalLutEvalTestParams{31, 32, 3, 2},
                                           CpuGrottoIntervalLutEvalTestParams{32, 32, 3, 2}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoIntervalLutEvalTestU64,
                         ::testing::Values(CpuGrottoIntervalLutEvalTestParams{6, 64, 3, 2},
                                           CpuGrottoIntervalLutEvalTestParams{63, 64, 3, 2},
                                           CpuGrottoIntervalLutEvalTestParams{64, 64, 3, 2}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoIntervalLutEvalTestU128,
                         ::testing::Values(CpuGrottoIntervalLutEvalTestParams{6, 128, 3, 2},
                                           CpuGrottoIntervalLutEvalTestParams{127, 128, 3, 2},
                                           CpuGrottoIntervalLutEvalTestParams{128, 128, 3, 2}));

} // namespace FastFss::tests::grotto
