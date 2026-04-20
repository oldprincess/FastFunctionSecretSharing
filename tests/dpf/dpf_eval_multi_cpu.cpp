#include <FastFssPP/dpf.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "dpf_eval_multi_case.h"

namespace FastFss::tests::dpf {

struct CpuDpfEvalMultiTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
    std::size_t pointNum;
};

template <typename T>
class CpuDpfEvalMultiTestBase : public ::testing::Test, public ::testing::WithParamInterface<CpuDpfEvalMultiTestParams>
{
public:
    using Type = T;
};

class CpuDpfEvalMultiTestU8 : public CpuDpfEvalMultiTestBase<std::uint8_t>
{
};
class CpuDpfEvalMultiTestU16 : public CpuDpfEvalMultiTestBase<std::uint16_t>
{
};
class CpuDpfEvalMultiTestU32 : public CpuDpfEvalMultiTestBase<std::uint32_t>
{
};
class CpuDpfEvalMultiTestU64 : public CpuDpfEvalMultiTestBase<std::uint64_t>
{
};
class CpuDpfEvalMultiTestU128 : public CpuDpfEvalMultiTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCpuDpfEvalMultiTestBody(Fixture* self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kGroupSizeConfigs  = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CpuDpfEvalMultiTestParams params = self->GetParam();

    std::size_t bitWidthIn  = params.bitWidthIn;
    std::size_t bitWidthOut = params.bitWidthOut;
    std::size_t pointNum    = params.pointNum;
    std::size_t elementSize = sizeof(T);
    T           maskOut     = (bitWidthOut == elementSize * 8) ? ~T(0) : (T(1) << bitWidthOut) - 1;

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t groupSize : kGroupSizeConfigs)
        {
            DpfEvalMultiTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, groupSize, pointNum);

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

            std::vector<T>            y0(elementNum * testCase.points.size() * groupSize);
            std::vector<T>            y1(elementNum * testCase.points.size() * groupSize);
            std::vector<std::uint8_t> evalCache(cacheDataSize);
            FastFss::dpf::cpu::dpfEvalMulti<T>(
                std::span<T>(y0.data(), y0.size()),
                std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                std::span<const std::uint8_t>(key.data(), key.size()),
                std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), 0,
                std::span<const T>(testCase.points.data(), testCase.points.size()), bitWidthIn, bitWidthOut, groupSize,
                std::span<std::uint8_t>(evalCache.data(), evalCache.size()));
            FastFss::dpf::cpu::dpfEvalMulti<T>(
                std::span<T>(y1.data(), y1.size()),
                std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                std::span<const std::uint8_t>(key.data(), key.size()),
                std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), 1,
                std::span<const T>(testCase.points.data(), testCase.points.size()), bitWidthIn, bitWidthOut, groupSize,
                std::span<std::uint8_t>{});

            for (std::size_t i = 0; i < y0.size(); ++i)
            {
                ASSERT_EQ((y0[i] + y1[i]) & maskOut, testCase.expected[i]);
            }
        }
    }
}

TEST_P(CpuDpfEvalMultiTestU8, RunTestBody)
{
    RunCpuDpfEvalMultiTestBody(this);
}
TEST_P(CpuDpfEvalMultiTestU16, RunTestBody)
{
    RunCpuDpfEvalMultiTestBody(this);
}
TEST_P(CpuDpfEvalMultiTestU32, RunTestBody)
{
    RunCpuDpfEvalMultiTestBody(this);
}
TEST_P(CpuDpfEvalMultiTestU64, RunTestBody)
{
    RunCpuDpfEvalMultiTestBody(this);
}
TEST_P(CpuDpfEvalMultiTestU128, RunTestBody)
{
    RunCpuDpfEvalMultiTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalMultiTestU8,
                         ::testing::Values(CpuDpfEvalMultiTestParams{1, 1, 1},
                                           CpuDpfEvalMultiTestParams{1, 7, 1},
                                           CpuDpfEvalMultiTestParams{1, 8, 1},
                                           CpuDpfEvalMultiTestParams{7, 1, 3},
                                           CpuDpfEvalMultiTestParams{7, 7, 7},
                                           CpuDpfEvalMultiTestParams{7, 8, 15},
                                           CpuDpfEvalMultiTestParams{8, 1, 31},
                                           CpuDpfEvalMultiTestParams{8, 7, 63},
                                           CpuDpfEvalMultiTestParams{8, 8, 127}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalMultiTestU16,
                         ::testing::Values(CpuDpfEvalMultiTestParams{1, 1, 1},
                                           CpuDpfEvalMultiTestParams{1, 15, 1},
                                           CpuDpfEvalMultiTestParams{1, 16, 1},
                                           CpuDpfEvalMultiTestParams{15, 1, 3},
                                           CpuDpfEvalMultiTestParams{15, 15, 7},
                                           CpuDpfEvalMultiTestParams{15, 16, 15},
                                           CpuDpfEvalMultiTestParams{16, 1, 31},
                                           CpuDpfEvalMultiTestParams{16, 15, 63},
                                           CpuDpfEvalMultiTestParams{16, 16, 127}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalMultiTestU32,
                         ::testing::Values(CpuDpfEvalMultiTestParams{1, 1, 1},
                                           CpuDpfEvalMultiTestParams{1, 31, 1},
                                           CpuDpfEvalMultiTestParams{1, 32, 1},
                                           CpuDpfEvalMultiTestParams{31, 1, 3},
                                           CpuDpfEvalMultiTestParams{31, 31, 7},
                                           CpuDpfEvalMultiTestParams{31, 32, 15},
                                           CpuDpfEvalMultiTestParams{32, 1, 31},
                                           CpuDpfEvalMultiTestParams{32, 31, 63},
                                           CpuDpfEvalMultiTestParams{32, 32, 127}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalMultiTestU64,
                         ::testing::Values(CpuDpfEvalMultiTestParams{1, 1, 1},
                                           CpuDpfEvalMultiTestParams{1, 63, 1},
                                           CpuDpfEvalMultiTestParams{1, 64, 1},
                                           CpuDpfEvalMultiTestParams{63, 1, 3},
                                           CpuDpfEvalMultiTestParams{63, 63, 7},
                                           CpuDpfEvalMultiTestParams{63, 64, 15},
                                           CpuDpfEvalMultiTestParams{64, 1, 31},
                                           CpuDpfEvalMultiTestParams{64, 63, 63},
                                           CpuDpfEvalMultiTestParams{64, 64, 127}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDpfEvalMultiTestU128,
                         ::testing::Values(CpuDpfEvalMultiTestParams{1, 1, 1},
                                           CpuDpfEvalMultiTestParams{1, 127, 1},
                                           CpuDpfEvalMultiTestParams{1, 128, 1},
                                           CpuDpfEvalMultiTestParams{127, 1, 3},
                                           CpuDpfEvalMultiTestParams{127, 127, 7},
                                           CpuDpfEvalMultiTestParams{127, 128, 15},
                                           CpuDpfEvalMultiTestParams{128, 1, 31},
                                           CpuDpfEvalMultiTestParams{128, 127, 63},
                                           CpuDpfEvalMultiTestParams{128, 128, 127}));

} // namespace FastFss::tests::dpf
