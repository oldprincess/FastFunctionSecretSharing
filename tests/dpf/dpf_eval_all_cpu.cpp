#include <FastFssPP/dpf.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "dpf_eval_all_case.h"

namespace FastFss::tests::dpf {

struct CpuDpfEvalAllTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

template <typename T>
class CpuDpfEvalAllTestBase : public ::testing::Test, public ::testing::WithParamInterface<CpuDpfEvalAllTestParams>
{
public:
    using Type = T;
};

class CpuDpfEvalAllTestU8 : public CpuDpfEvalAllTestBase<std::uint8_t>
{
};
class CpuDpfEvalAllTestU16 : public CpuDpfEvalAllTestBase<std::uint16_t>
{
};
class CpuDpfEvalAllTestU32 : public CpuDpfEvalAllTestBase<std::uint32_t>
{
};
class CpuDpfEvalAllTestU64 : public CpuDpfEvalAllTestBase<std::uint64_t>
{
};
class CpuDpfEvalAllTestU128 : public CpuDpfEvalAllTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCpuDpfEvalAllTestBody(Fixture* self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kGroupSizeConfigs  = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CpuDpfEvalAllTestParams params = self->GetParam();

    std::size_t bitWidthIn  = params.bitWidthIn;
    std::size_t bitWidthOut = params.bitWidthOut;
    std::size_t elementSize = sizeof(T);
    T           maskOut     = (bitWidthOut == elementSize * 8) ? ~T(0) : (T(1) << bitWidthOut) - 1;

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t groupSize : kGroupSizeConfigs)
        {
            DpfEvalAllTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, groupSize);
            std::size_t           domainSize = testCase.domainSize;
            std::size_t           keyDataSize =
                FastFss::dpf::dpfGetKeyDataSize(bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
            std::size_t cacheDataSize = FastFss::dpf::dpfGetCacheDataSize(bitWidthIn, elementSize, elementNum);
            std::vector<std::uint8_t> key(keyDataSize);

            FastFss::dpf::cpu::dpfKeyGen<T>(std::span<std::uint8_t>(key.data(), key.size()),
                                            std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
                                            std::span<const T>(testCase.beta.data(), testCase.beta.size()),
                                            std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
                                            std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()),
                                            bitWidthIn, bitWidthOut, groupSize);

            std::vector<T>            y0(elementNum * domainSize * groupSize);
            std::vector<T>            y1(elementNum * domainSize * groupSize);
            std::vector<std::uint8_t> evalCache(cacheDataSize);
            FastFss::dpf::cpu::dpfEvalAll<T>(
                std::span<T>(y0.data(), y0.size()),
                std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                std::span<const std::uint8_t>(key.data(), key.size()),
                std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), 0, bitWidthIn, bitWidthOut,
                groupSize, std::span<std::uint8_t>(evalCache.data(), evalCache.size()));
            FastFss::dpf::cpu::dpfEvalAll<T>(
                std::span<T>(y1.data(), y1.size()),
                std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                std::span<const std::uint8_t>(key.data(), key.size()),
                std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), 1, bitWidthIn, bitWidthOut,
                groupSize, std::span<std::uint8_t>{});

            for (std::size_t i = 0; i < y0.size(); ++i)
            {
                ASSERT_EQ((y0[i] + y1[i]) & maskOut, testCase.expected[i]);
            }
        }
    }
}

TEST_P(CpuDpfEvalAllTestU8, RunTestBody)
{
    RunCpuDpfEvalAllTestBody(this);
}
TEST_P(CpuDpfEvalAllTestU16, RunTestBody)
{
    RunCpuDpfEvalAllTestBody(this);
}
TEST_P(CpuDpfEvalAllTestU32, RunTestBody)
{
    RunCpuDpfEvalAllTestBody(this);
}
TEST_P(CpuDpfEvalAllTestU64, RunTestBody)
{
    RunCpuDpfEvalAllTestBody(this);
}
TEST_P(CpuDpfEvalAllTestU128, RunTestBody)
{
    RunCpuDpfEvalAllTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalAllTestU8,
                         ::testing::Values(CpuDpfEvalAllTestParams{1, 1},
                                           CpuDpfEvalAllTestParams{1, 7},
                                           CpuDpfEvalAllTestParams{1, 8},
                                           CpuDpfEvalAllTestParams{7, 1},
                                           CpuDpfEvalAllTestParams{7, 7},
                                           CpuDpfEvalAllTestParams{7, 8},
                                           CpuDpfEvalAllTestParams{8, 1},
                                           CpuDpfEvalAllTestParams{8, 7},
                                           CpuDpfEvalAllTestParams{8, 8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalAllTestU16,
                         ::testing::Values(CpuDpfEvalAllTestParams{1, 1},
                                           CpuDpfEvalAllTestParams{1, 9},
                                           CpuDpfEvalAllTestParams{1, 10},
                                           CpuDpfEvalAllTestParams{9, 1},
                                           CpuDpfEvalAllTestParams{9, 9},
                                           CpuDpfEvalAllTestParams{9, 10},
                                           CpuDpfEvalAllTestParams{10, 1},
                                           CpuDpfEvalAllTestParams{10, 9},
                                           CpuDpfEvalAllTestParams{10, 10}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalAllTestU32,
                         ::testing::Values(CpuDpfEvalAllTestParams{1, 1},
                                           CpuDpfEvalAllTestParams{1, 9},
                                           CpuDpfEvalAllTestParams{1, 10},
                                           CpuDpfEvalAllTestParams{9, 1},
                                           CpuDpfEvalAllTestParams{9, 9},
                                           CpuDpfEvalAllTestParams{9, 10},
                                           CpuDpfEvalAllTestParams{10, 1},
                                           CpuDpfEvalAllTestParams{10, 9},
                                           CpuDpfEvalAllTestParams{10, 10}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalAllTestU64,
                         ::testing::Values(CpuDpfEvalAllTestParams{1, 1},
                                           CpuDpfEvalAllTestParams{1, 9},
                                           CpuDpfEvalAllTestParams{1, 10},
                                           CpuDpfEvalAllTestParams{9, 1},
                                           CpuDpfEvalAllTestParams{9, 9},
                                           CpuDpfEvalAllTestParams{9, 10},
                                           CpuDpfEvalAllTestParams{10, 1},
                                           CpuDpfEvalAllTestParams{10, 9},
                                           CpuDpfEvalAllTestParams{10, 10}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalAllTestU128,
                         ::testing::Values(CpuDpfEvalAllTestParams{1, 1},
                                           CpuDpfEvalAllTestParams{1, 9},
                                           CpuDpfEvalAllTestParams{1, 10},
                                           CpuDpfEvalAllTestParams{9, 1},
                                           CpuDpfEvalAllTestParams{9, 9},
                                           CpuDpfEvalAllTestParams{9, 10},
                                           CpuDpfEvalAllTestParams{10, 1},
                                           CpuDpfEvalAllTestParams{10, 9},
                                           CpuDpfEvalAllTestParams{10, 10}));

} // namespace FastFss::tests::dpf
