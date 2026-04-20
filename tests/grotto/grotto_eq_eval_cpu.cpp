#include <FastFssPP/grotto.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "grotto_eq_eval_test_case.h"

namespace FastFss::tests::grotto {

struct CpuGrottoEqEvalTestParams
{
    std::size_t bitWidthIn;
};

template <typename T>
class CpuGrottoEqEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CpuGrottoEqEvalTestParams>
{
public:
    using Type = T;
};

class CpuGrottoEqEvalTestU8 : public CpuGrottoEqEvalTestBase<std::uint8_t>
{
};
class CpuGrottoEqEvalTestU16 : public CpuGrottoEqEvalTestBase<std::uint16_t>
{
};
class CpuGrottoEqEvalTestU32 : public CpuGrottoEqEvalTestBase<std::uint32_t>
{
};
class CpuGrottoEqEvalTestU64 : public CpuGrottoEqEvalTestBase<std::uint64_t>
{
};
class CpuGrottoEqEvalTestU128 : public CpuGrottoEqEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCpuGrottoEqEvalTestBody(Fixture* self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};

    using T = typename Fixture::Type;

    CpuGrottoEqEvalTestParams params = self->GetParam();

    const std::size_t bitWidthIn  = params.bitWidthIn;
    const std::size_t elementSize = sizeof(T);

    for (std::size_t elementNum : kElementNumConfigs)
    {
        GrottoEqEvalTestCase<T> testCase(bitWidthIn, elementNum);

        std::size_t keySize   = FastFss::grotto::grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum);
        std::size_t cacheSize = FastFss::grotto::grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);

        std::vector<std::uint8_t> key(keySize);
        FastFss::grotto::cpu::grottoKeyGen<T>(
            std::span<std::uint8_t>(key.data(), key.size()), std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
            std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
            std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), bitWidthIn);

        std::vector<T>            eq0(elementNum), eq1(elementNum);
        std::vector<std::uint8_t> evalCache(cacheSize);
        FastFss::grotto::cpu::grottoEqEval<T>(
            std::span<T>(eq0.data(), eq0.size()),
            std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
            std::span<const std::uint8_t>(key.data(), key.size()),
            std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), 0, bitWidthIn,
            std::span<std::uint8_t>(evalCache.data(), evalCache.size()));
        FastFss::grotto::cpu::grottoEqEval<T>(
            std::span<T>(eq1.data(), eq1.size()),
            std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
            std::span<const std::uint8_t>(key.data(), key.size()),
            std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), 1, bitWidthIn,
            std::span<std::uint8_t>{});

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            const T eqOut = (eq0[i] + eq1[i]) & T(1);
            ASSERT_EQ(eqOut, testCase.expected[i]);
        }
    }
}

TEST_P(CpuGrottoEqEvalTestU8, RunTestBody)
{
    RunCpuGrottoEqEvalTestBody(this);
}
TEST_P(CpuGrottoEqEvalTestU16, RunTestBody)
{
    RunCpuGrottoEqEvalTestBody(this);
}
TEST_P(CpuGrottoEqEvalTestU32, RunTestBody)
{
    RunCpuGrottoEqEvalTestBody(this);
}
TEST_P(CpuGrottoEqEvalTestU64, RunTestBody)
{
    RunCpuGrottoEqEvalTestBody(this);
}
TEST_P(CpuGrottoEqEvalTestU128, RunTestBody)
{
    RunCpuGrottoEqEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoEqEvalTestU8,
                         ::testing::Values(CpuGrottoEqEvalTestParams{6},
                                           CpuGrottoEqEvalTestParams{7},
                                           CpuGrottoEqEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoEqEvalTestU16,
                         ::testing::Values(CpuGrottoEqEvalTestParams{6},
                                           CpuGrottoEqEvalTestParams{15},
                                           CpuGrottoEqEvalTestParams{16}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoEqEvalTestU32,
                         ::testing::Values(CpuGrottoEqEvalTestParams{6},
                                           CpuGrottoEqEvalTestParams{31},
                                           CpuGrottoEqEvalTestParams{32}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoEqEvalTestU64,
                         ::testing::Values(CpuGrottoEqEvalTestParams{6},
                                           CpuGrottoEqEvalTestParams{63},
                                           CpuGrottoEqEvalTestParams{64}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoEqEvalTestU128,
                         ::testing::Values(CpuGrottoEqEvalTestParams{6},
                                           CpuGrottoEqEvalTestParams{127},
                                           CpuGrottoEqEvalTestParams{128}));

} // namespace FastFss::tests::grotto
