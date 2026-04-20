#include <FastFssPP/grotto.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "grotto_mic_eval_test_case.h"

namespace FastFss::tests::grotto {

struct CpuGrottoMicEvalTestParams
{
    std::size_t bitWidthIn;
};

template <typename T>
class CpuGrottoMicEvalTestBase : public ::testing::Test,
                                 public ::testing::WithParamInterface<CpuGrottoMicEvalTestParams>
{
public:
    using Type = T;
};

class CpuGrottoMicEvalTestU8 : public CpuGrottoMicEvalTestBase<std::uint8_t>
{
};
class CpuGrottoMicEvalTestU16 : public CpuGrottoMicEvalTestBase<std::uint16_t>
{
};
class CpuGrottoMicEvalTestU32 : public CpuGrottoMicEvalTestBase<std::uint32_t>
{
};
class CpuGrottoMicEvalTestU64 : public CpuGrottoMicEvalTestBase<std::uint64_t>
{
};
class CpuGrottoMicEvalTestU128 : public CpuGrottoMicEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCpuGrottoMicEvalTestBody(Fixture* self)
{
    static std::vector<std::size_t> kElementNumConfigs    = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kIntervalCountConfigs = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CpuGrottoMicEvalTestParams params = self->GetParam();

    const std::size_t bitWidthIn  = params.bitWidthIn;
    const std::size_t elementSize = sizeof(T);

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t intervalCount : kIntervalCountConfigs)
        {
            GrottoMicEvalTestCase<T> testCase(bitWidthIn, elementNum, intervalCount);

            std::size_t keySize   = FastFss::grotto::grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum);
            std::size_t cacheSize = FastFss::grotto::grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);

            std::vector<std::uint8_t> key(keySize);
            FastFss::grotto::cpu::grottoKeyGen<T>(
                std::span<std::uint8_t>(key.data(), key.size()),
                std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
                std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
                std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), bitWidthIn);

            std::vector<T>            share0(elementNum * intervalCount);
            std::vector<T>            share1(elementNum * intervalCount);
            std::vector<std::uint8_t> evalCache(cacheSize);
            FastFss::grotto::cpu::grottoMICEval<T>(
                std::span<T>(share0.data(), share0.size()),
                std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                std::span<const std::uint8_t>(key.data(), key.size()),
                std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), 0,
                std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), bitWidthIn,
                std::span<std::uint8_t>(evalCache.data(), evalCache.size()));
            FastFss::grotto::cpu::grottoMICEval<T>(
                std::span<T>(share1.data(), share1.size()),
                std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                std::span<const std::uint8_t>(key.data(), key.size()),
                std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), 1,
                std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), bitWidthIn,
                std::span<std::uint8_t>{});

            for (std::size_t i = 0; i < share0.size(); ++i)
            {
                const T out = (share0[i] + share1[i]) & T(1);
                ASSERT_EQ(out, testCase.expected[i]);
            }
        }
    }
}

TEST_P(CpuGrottoMicEvalTestU8, RunTestBody)
{
    RunCpuGrottoMicEvalTestBody(this);
}
TEST_P(CpuGrottoMicEvalTestU16, RunTestBody)
{
    RunCpuGrottoMicEvalTestBody(this);
}
TEST_P(CpuGrottoMicEvalTestU32, RunTestBody)
{
    RunCpuGrottoMicEvalTestBody(this);
}
TEST_P(CpuGrottoMicEvalTestU64, RunTestBody)
{
    RunCpuGrottoMicEvalTestBody(this);
}
TEST_P(CpuGrottoMicEvalTestU128, RunTestBody)
{
    RunCpuGrottoMicEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoMicEvalTestU8,
                         ::testing::Values(CpuGrottoMicEvalTestParams{6},
                                           CpuGrottoMicEvalTestParams{7},
                                           CpuGrottoMicEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoMicEvalTestU16,
                         ::testing::Values(CpuGrottoMicEvalTestParams{6},
                                           CpuGrottoMicEvalTestParams{15},
                                           CpuGrottoMicEvalTestParams{16}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoMicEvalTestU32,
                         ::testing::Values(CpuGrottoMicEvalTestParams{6},
                                           CpuGrottoMicEvalTestParams{31},
                                           CpuGrottoMicEvalTestParams{32}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoMicEvalTestU64,
                         ::testing::Values(CpuGrottoMicEvalTestParams{6},
                                           CpuGrottoMicEvalTestParams{63},
                                           CpuGrottoMicEvalTestParams{64}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuGrottoMicEvalTestU128,
                         ::testing::Values(CpuGrottoMicEvalTestParams{6},
                                           CpuGrottoMicEvalTestParams{127},
                                           CpuGrottoMicEvalTestParams{128}));

} // namespace FastFss::tests::grotto
