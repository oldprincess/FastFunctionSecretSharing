#include <FastFssPP/grotto.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "grotto_eval_test_case.h"

namespace FastFss::tests::grotto {

struct CpuGrottoEvalTestParams
{
    std::size_t bitWidthIn;
};

template <typename T>
class CpuGrottoEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CpuGrottoEvalTestParams>
{
public:
    using Type = T;
};

class CpuGrottoEvalTestU8 : public CpuGrottoEvalTestBase<std::uint8_t>
{
};
class CpuGrottoEvalTestU16 : public CpuGrottoEvalTestBase<std::uint16_t>
{
};
class CpuGrottoEvalTestU32 : public CpuGrottoEvalTestBase<std::uint32_t>
{
};
class CpuGrottoEvalTestU64 : public CpuGrottoEvalTestBase<std::uint64_t>
{
};
class CpuGrottoEvalTestU128 : public CpuGrottoEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCpuGrottoEvalTestBody(Fixture* self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};

    using T = typename Fixture::Type;

    CpuGrottoEvalTestParams params = self->GetParam();

    const std::size_t bitWidthIn  = params.bitWidthIn;
    const std::size_t elementSize = sizeof(T);

    for (std::size_t elementNum : kElementNumConfigs)
    {
        GrottoEvalTestCase<T> testCase(bitWidthIn, elementNum);

        std::size_t keySize   = FastFss::grotto::grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum);
        std::size_t cacheSize = FastFss::grotto::grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);

        std::vector<std::uint8_t> key(keySize);
        FastFss::grotto::cpu::grottoKeyGen<T>(
            std::span<std::uint8_t>(key.data(), key.size()), std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
            std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
            std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), bitWidthIn);

        std::vector<T>            lt0(elementNum), lt1(elementNum);
        std::vector<T>            le0(elementNum), le1(elementNum);
        std::vector<std::uint8_t> evalCache(cacheSize);
        FastFss::grotto::cpu::grottoEval<T>(
            std::span<T>(lt0.data(), lt0.size()),
            std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
            std::span<const std::uint8_t>(key.data(), key.size()),
            std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), false, 0, bitWidthIn,
            std::span<std::uint8_t>(evalCache.data(), evalCache.size()));
        FastFss::grotto::cpu::grottoEval<T>(
            std::span<T>(lt1.data(), lt1.size()),
            std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
            std::span<const std::uint8_t>(key.data(), key.size()),
            std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), false, 1, bitWidthIn,
            std::span<std::uint8_t>{});
        FastFss::grotto::cpu::grottoEval<T>(
            std::span<T>(le0.data(), le0.size()),
            std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
            std::span<const std::uint8_t>(key.data(), key.size()),
            std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), true, 0, bitWidthIn,
            std::span<std::uint8_t>{});
        FastFss::grotto::cpu::grottoEval<T>(
            std::span<T>(le1.data(), le1.size()),
            std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
            std::span<const std::uint8_t>(key.data(), key.size()),
            std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), true, 1, bitWidthIn,
            std::span<std::uint8_t>{});

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            const T ltOut = (lt0[i] + lt1[i]) & T(1);
            const T leOut = (le0[i] + le1[i]) & T(1);
            ASSERT_EQ(ltOut, testCase.expectedLt[i]);
            ASSERT_EQ(leOut, testCase.expectedLe[i]);
        }
    }
}

TEST_P(CpuGrottoEvalTestU8, RunTestBody)
{
    RunCpuGrottoEvalTestBody(this);
}
TEST_P(CpuGrottoEvalTestU16, RunTestBody)
{
    RunCpuGrottoEvalTestBody(this);
}
TEST_P(CpuGrottoEvalTestU32, RunTestBody)
{
    RunCpuGrottoEvalTestBody(this);
}
TEST_P(CpuGrottoEvalTestU64, RunTestBody)
{
    RunCpuGrottoEvalTestBody(this);
}
TEST_P(CpuGrottoEvalTestU128, RunTestBody)
{
    RunCpuGrottoEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoEvalTestU8,
                         ::testing::Values(CpuGrottoEvalTestParams{6},
                                           CpuGrottoEvalTestParams{7},
                                           CpuGrottoEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoEvalTestU16,
                         ::testing::Values(CpuGrottoEvalTestParams{6},
                                           CpuGrottoEvalTestParams{15},
                                           CpuGrottoEvalTestParams{16}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoEvalTestU32,
                         ::testing::Values(CpuGrottoEvalTestParams{6},
                                           CpuGrottoEvalTestParams{31},
                                           CpuGrottoEvalTestParams{32}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoEvalTestU64,
                         ::testing::Values(CpuGrottoEvalTestParams{6},
                                           CpuGrottoEvalTestParams{63},
                                           CpuGrottoEvalTestParams{64}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoEvalTestU128,
                         ::testing::Values(CpuGrottoEvalTestParams{6},
                                           CpuGrottoEvalTestParams{127},
                                           CpuGrottoEvalTestParams{128}));

} // namespace FastFss::tests::grotto
