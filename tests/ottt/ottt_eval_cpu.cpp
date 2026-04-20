#include <FastFssPP/ottt.h>
#include <FastFssPP/prng.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "ottt_lut_eval_case.h"

namespace FastFss::tests::ottt {

struct CpuOtttEvalTestParams
{
    std::size_t bitWidthIn;
};

template <typename T>
class CpuOtttLutEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CpuOtttEvalTestParams>
{
public:
    using Type = T;
};

class CpuOtttLutEvalTestU8 : public CpuOtttLutEvalTestBase<std::uint8_t>
{
};
class CpuOtttLutEvalTestU16 : public CpuOtttLutEvalTestBase<std::uint16_t>
{
};
class CpuOtttLutEvalTestU32 : public CpuOtttLutEvalTestBase<std::uint32_t>
{
};
class CpuOtttLutEvalTestU64 : public CpuOtttLutEvalTestBase<std::uint64_t>
{
};
class CpuOtttLutEvalTestU128 : public CpuOtttLutEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCpuOtttLutEvalTestBody(Fixture* self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kLutsNumConfigs    = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    std::size_t bitWidthIn = self->GetParam().bitWidthIn;

    FastFss::prng::cpu::Prng prng;

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t lutsNum : kLutsNumConfigs)
        {
            OtttLutEvalTestCase<T> testCase(bitWidthIn, lutsNum, elementNum);

            std::size_t keySize = FastFss::ottt::otttGetKeyDataSize(bitWidthIn, elementNum);

            std::vector<std::uint8_t> key0(keySize);
            std::vector<std::uint8_t> key1(keySize);
            prng.generate(key0.data(), 8, key0.size());
            key1.assign(key0.begin(), key0.end());
            FastFss::ottt::cpu::otttKeyGen<T>(std::span<std::uint8_t>(key1.data(), key1.size()),
                                              std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
                                              bitWidthIn);

            std::vector<T> shareE0(elementNum), shareE1(elementNum);
            std::vector<T> shareT0(elementNum * lutsNum), shareT1(elementNum * lutsNum);
            FastFss::ottt::cpu::otttLutEval<T>(
                std::span<T>(shareE0.data(), shareE0.size()), std::span<T>(shareT0.data(), shareT0.size()),
                std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                std::span<const std::uint8_t>(key0.data(), key0.size()), 0,
                std::span<const T>(testCase.lut.data(), testCase.lut.size()), bitWidthIn);
            FastFss::ottt::cpu::otttLutEval<T>(
                std::span<T>(shareE1.data(), shareE1.size()), std::span<T>(shareT1.data(), shareT1.size()),
                std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                std::span<const std::uint8_t>(key1.data(), key1.size()), 1,
                std::span<const T>(testCase.lut.data(), testCase.lut.size()), bitWidthIn);

            for (std::size_t i = 0; i < elementNum; ++i)
            {
                T e = shareE0[i] + shareE1[i];
                for (std::size_t j = 0; j < lutsNum; ++j)
                {
                    T t   = (shareT0[i * lutsNum + j] + shareT1[i * lutsNum + j]) & testCase.maskOut;
                    T out = (e * t) & testCase.maskOut;
                    ASSERT_EQ(out, testCase.expected[i * lutsNum + j]);
                }
            }
        }
    }
}

TEST_P(CpuOtttLutEvalTestU8, RunTestBody)
{
    RunCpuOtttLutEvalTestBody(this);
}
TEST_P(CpuOtttLutEvalTestU16, RunTestBody)
{
    RunCpuOtttLutEvalTestBody(this);
}
TEST_P(CpuOtttLutEvalTestU32, RunTestBody)
{
    RunCpuOtttLutEvalTestBody(this);
}
TEST_P(CpuOtttLutEvalTestU64, RunTestBody)
{
    RunCpuOtttLutEvalTestBody(this);
}
TEST_P(CpuOtttLutEvalTestU128, RunTestBody)
{
    RunCpuOtttLutEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CpuOtttLutEvalTestU8,
                         ::testing::Values(CpuOtttEvalTestParams{3},
                                           CpuOtttEvalTestParams{4},
                                           CpuOtttEvalTestParams{5},
                                           CpuOtttEvalTestParams{6},
                                           CpuOtttEvalTestParams{7},
                                           CpuOtttEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuOtttLutEvalTestU16,
                         ::testing::Values(CpuOtttEvalTestParams{3},
                                           CpuOtttEvalTestParams{4},
                                           CpuOtttEvalTestParams{5},
                                           CpuOtttEvalTestParams{6},
                                           CpuOtttEvalTestParams{7},
                                           CpuOtttEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuOtttLutEvalTestU32,
                         ::testing::Values(CpuOtttEvalTestParams{3},
                                           CpuOtttEvalTestParams{4},
                                           CpuOtttEvalTestParams{5},
                                           CpuOtttEvalTestParams{6},
                                           CpuOtttEvalTestParams{7},
                                           CpuOtttEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuOtttLutEvalTestU64,
                         ::testing::Values(CpuOtttEvalTestParams{3},
                                           CpuOtttEvalTestParams{4},
                                           CpuOtttEvalTestParams{5},
                                           CpuOtttEvalTestParams{6},
                                           CpuOtttEvalTestParams{7},
                                           CpuOtttEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuOtttLutEvalTestU128,
                         ::testing::Values(CpuOtttEvalTestParams{3},
                                           CpuOtttEvalTestParams{4},
                                           CpuOtttEvalTestParams{5},
                                           CpuOtttEvalTestParams{6},
                                           CpuOtttEvalTestParams{7},
                                           CpuOtttEvalTestParams{8}));

} // namespace FastFss::tests::ottt
