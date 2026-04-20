#include <FastFssPP/dpf.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "dpf_eval_case.h"

namespace FastFss::tests::dpf {

struct CpuDpfEvalTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

template <typename T>
class CpuDpfEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CpuDpfEvalTestParams>
{
public:
    using Type = T;
};

class CpuDpfEvalTestU8 : public CpuDpfEvalTestBase<std::uint8_t>
{
};
class CpuDpfEvalTestU16 : public CpuDpfEvalTestBase<std::uint16_t>
{
};
class CpuDpfEvalTestU32 : public CpuDpfEvalTestBase<std::uint32_t>
{
};
class CpuDpfEvalTestU64 : public CpuDpfEvalTestBase<std::uint64_t>
{
};
class CpuDpfEvalTestU128 : public CpuDpfEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCpuDpfEvalTestBody(Fixture* self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kGroupSizeConfigs  = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CpuDpfEvalTestParams params = self->GetParam();

    std::size_t bitWidthIn  = params.bitWidthIn;
    std::size_t bitWidthOut = params.bitWidthOut;
    std::size_t elementSize = sizeof(T);
    T           maskOut     = (bitWidthOut == elementSize * 8) ? ~T(0) : (T(1) << bitWidthOut) - 1;

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t groupSize : kGroupSizeConfigs)
        {
            DpfEvalTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, groupSize);

            std::size_t keyDataSize =
                FastFss::dpf::dpfGetKeyDataSize(bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
            std::size_t cacheDataSize = FastFss::dpf::dpfGetCacheDataSize(bitWidthIn, elementSize, elementNum);

            std::vector<std::uint8_t> key(keyDataSize);
            FastFss::dpf::cpu::dpfKeyGen<T>(std::span<std::uint8_t>(key.data(), key.size()),
                                            std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
                                            std::span<const T>(testCase.beta.data(), testCase.beta.size()),
                                            std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
                                            std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()),
                                            bitWidthIn, bitWidthOut, groupSize);

            std::vector<T>            y0(elementNum * groupSize);
            std::vector<T>            y1(elementNum * groupSize);
            std::vector<std::uint8_t> evalCache(cacheDataSize);

            FastFss::dpf::cpu::dpfEval<T>(std::span<T>(y0.data(), y0.size()),
                                          std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                                          std::span<const std::uint8_t>(key.data(), key.size()),
                                          std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
                                          0, bitWidthIn, bitWidthOut, groupSize,
                                          std::span<std::uint8_t>(evalCache.data(), evalCache.size()));
            FastFss::dpf::cpu::dpfEval<T>(std::span<T>(y1.data(), y1.size()),
                                          std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                                          std::span<const std::uint8_t>(key.data(), key.size()),
                                          std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()),
                                          1, bitWidthIn, bitWidthOut, groupSize, std::span<std::uint8_t>{});

            for (std::size_t i = 0; i < y0.size(); ++i)
            {
                ASSERT_EQ((y0[i] + y1[i]) & maskOut, testCase.expected[i]);
            }
        }
    }
}

TEST_P(CpuDpfEvalTestU8, RunTestBody)
{
    RunCpuDpfEvalTestBody(this);
}
TEST_P(CpuDpfEvalTestU16, RunTestBody)
{
    RunCpuDpfEvalTestBody(this);
}
TEST_P(CpuDpfEvalTestU32, RunTestBody)
{
    RunCpuDpfEvalTestBody(this);
}
TEST_P(CpuDpfEvalTestU64, RunTestBody)
{
    RunCpuDpfEvalTestBody(this);
}
TEST_P(CpuDpfEvalTestU128, RunTestBody)
{
    RunCpuDpfEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalTestU8,
                         ::testing::Values(CpuDpfEvalTestParams{1, 1},
                                           CpuDpfEvalTestParams{1, 7},
                                           CpuDpfEvalTestParams{1, 8},
                                           CpuDpfEvalTestParams{7, 1},
                                           CpuDpfEvalTestParams{7, 7},
                                           CpuDpfEvalTestParams{7, 8},
                                           CpuDpfEvalTestParams{8, 1},
                                           CpuDpfEvalTestParams{8, 7},
                                           CpuDpfEvalTestParams{8, 8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalTestU16,
                         ::testing::Values(CpuDpfEvalTestParams{1, 1},
                                           CpuDpfEvalTestParams{1, 15},
                                           CpuDpfEvalTestParams{1, 16},
                                           CpuDpfEvalTestParams{15, 1},
                                           CpuDpfEvalTestParams{15, 15},
                                           CpuDpfEvalTestParams{15, 16},
                                           CpuDpfEvalTestParams{16, 1},
                                           CpuDpfEvalTestParams{16, 15},
                                           CpuDpfEvalTestParams{16, 16}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalTestU32,
                         ::testing::Values(CpuDpfEvalTestParams{1, 1},
                                           CpuDpfEvalTestParams{1, 31},
                                           CpuDpfEvalTestParams{1, 32},
                                           CpuDpfEvalTestParams{31, 1},
                                           CpuDpfEvalTestParams{31, 31},
                                           CpuDpfEvalTestParams{31, 32},
                                           CpuDpfEvalTestParams{32, 1},
                                           CpuDpfEvalTestParams{32, 31},
                                           CpuDpfEvalTestParams{32, 32}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalTestU64,
                         ::testing::Values(CpuDpfEvalTestParams{1, 1},
                                           CpuDpfEvalTestParams{1, 63},
                                           CpuDpfEvalTestParams{1, 64},
                                           CpuDpfEvalTestParams{63, 1},
                                           CpuDpfEvalTestParams{63, 63},
                                           CpuDpfEvalTestParams{63, 64},
                                           CpuDpfEvalTestParams{64, 1},
                                           CpuDpfEvalTestParams{64, 63},
                                           CpuDpfEvalTestParams{64, 64}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalTestU128,
                         ::testing::Values(CpuDpfEvalTestParams{1, 1},
                                           CpuDpfEvalTestParams{1, 127},
                                           CpuDpfEvalTestParams{1, 128},
                                           CpuDpfEvalTestParams{127, 1},
                                           CpuDpfEvalTestParams{127, 127},
                                           CpuDpfEvalTestParams{127, 128},
                                           CpuDpfEvalTestParams{128, 1},
                                           CpuDpfEvalTestParams{128, 127},
                                           CpuDpfEvalTestParams{128, 128}));

} // namespace FastFss::tests::dpf
