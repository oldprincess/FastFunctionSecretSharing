#include <FastFssPP/dcf.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "dcf_eval_test_case.h"

namespace FastFss::tests::dcf {

struct CpuDcfEvalTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

template <typename T>
class CpuDcfEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CpuDcfEvalTestParams>
{
public:
    using Type = T;
};

class CpuDcfEvalTestU8 : public CpuDcfEvalTestBase<std::uint8_t>
{
};

class CpuDcfEvalTestU16 : public CpuDcfEvalTestBase<std::uint16_t>
{
};

class CpuDcfEvalTestU32 : public CpuDcfEvalTestBase<std::uint32_t>
{
};

class CpuDcfEvalTestU64 : public CpuDcfEvalTestBase<std::uint64_t>
{
};

class CpuDcfEvalTestU128 : public CpuDcfEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCpuDcfEvalTestBody(Fixture* self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kGroupSizeConfigs  = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CpuDcfEvalTestParams params = self->GetParam();

    std::size_t bitWidthIn  = params.bitWidthIn;
    std::size_t bitWidthOut = params.bitWidthOut;
    std::size_t elementSize = sizeof(T);

    T maskOut = (bitWidthOut == elementSize * 8) ? ~T(0) : (T(1) << bitWidthOut) - 1;

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t groupSize : kGroupSizeConfigs)
        {
            DcfEvalTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, groupSize);

            // generate DCF key
            std::size_t keyDataSize =
                FastFss::dcf::dcfGetKeyDataSize(bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);

            std::vector<std::uint8_t> key(keyDataSize);

            FastFss::dcf::cpu::dcfKeyGen<T>(std::span<std::uint8_t>(key.data(), key.size()),
                                            std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
                                            std::span<const T>(testCase.beta.data(), testCase.beta.size()),
                                            std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
                                            std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()),
                                            bitWidthIn, bitWidthOut, groupSize);

            // generate DCF cache
            std::size_t cacheDataSize =
                FastFss::dcf::dcfGetCacheDataSize(bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
            std::vector<std::uint8_t> evalCache(cacheDataSize);

            // eval
            std::vector<T> y0(elementNum * groupSize);
            std::vector<T> y1(elementNum * groupSize);

            FastFss::dcf::cpu::dcfEval<T>(std::span<T>(y0.data(), y0.size()),
                                          std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                                          std::span<const std::uint8_t>(key.data(), key.size()),
                                          std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
                                          0, bitWidthIn, bitWidthOut, groupSize, std::span<std::uint8_t>{});
            FastFss::dcf::cpu::dcfEval<T>(std::span<T>(y1.data(), y1.size()),
                                          std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                                          std::span<const std::uint8_t>(key.data(), key.size()),
                                          std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()),
                                          1, bitWidthIn, bitWidthOut, groupSize,
                                          std::span<std::uint8_t>(evalCache.data(), evalCache.size()));

            // check
            for (std::size_t i = 0; i < y0.size(); ++i)
            {
                T y = (y0[i] + y1[i]) & maskOut;
                ASSERT_EQ(y, testCase.y[i]);
            }
        }
    }
}

TEST_P(CpuDcfEvalTestU8, RunTestBody)
{
    RunCpuDcfEvalTestBody(this);
}

TEST_P(CpuDcfEvalTestU16, RunTestBody)
{
    RunCpuDcfEvalTestBody(this);
}

TEST_P(CpuDcfEvalTestU32, RunTestBody)
{
    RunCpuDcfEvalTestBody(this);
}

TEST_P(CpuDcfEvalTestU64, RunTestBody)
{
    RunCpuDcfEvalTestBody(this);
}

TEST_P(CpuDcfEvalTestU128, RunTestBody)
{
    RunCpuDcfEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CpuDcfEvalTestU8,
                         ::testing::Values(CpuDcfEvalTestParams{1, 1},
                                           CpuDcfEvalTestParams{1, 7},
                                           CpuDcfEvalTestParams{1, 8},
                                           CpuDcfEvalTestParams{7, 1},
                                           CpuDcfEvalTestParams{7, 7},
                                           CpuDcfEvalTestParams{7, 8},
                                           CpuDcfEvalTestParams{8, 1},
                                           CpuDcfEvalTestParams{8, 7},
                                           CpuDcfEvalTestParams{8, 8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDcfEvalTestU16,
                         ::testing::Values(CpuDcfEvalTestParams{1, 1},
                                           CpuDcfEvalTestParams{1, 15},
                                           CpuDcfEvalTestParams{1, 16},
                                           CpuDcfEvalTestParams{15, 1},
                                           CpuDcfEvalTestParams{15, 15},
                                           CpuDcfEvalTestParams{15, 16},
                                           CpuDcfEvalTestParams{16, 1},
                                           CpuDcfEvalTestParams{16, 15},
                                           CpuDcfEvalTestParams{16, 16}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuDcfEvalTestU32,
                         ::testing::Values(CpuDcfEvalTestParams{1, 1},
                                           CpuDcfEvalTestParams{1, 31},
                                           CpuDcfEvalTestParams{1, 32},
                                           CpuDcfEvalTestParams{31, 1},
                                           CpuDcfEvalTestParams{31, 31},
                                           CpuDcfEvalTestParams{31, 32},
                                           CpuDcfEvalTestParams{32, 1},
                                           CpuDcfEvalTestParams{32, 31},
                                           CpuDcfEvalTestParams{32, 32}));

INSTANTIATE_TEST_SUITE_P(,
                         CpuDcfEvalTestU64,
                         ::testing::Values(CpuDcfEvalTestParams{1, 1},
                                           CpuDcfEvalTestParams{1, 63},
                                           CpuDcfEvalTestParams{1, 64},
                                           CpuDcfEvalTestParams{63, 1},
                                           CpuDcfEvalTestParams{63, 63},
                                           CpuDcfEvalTestParams{63, 64},
                                           CpuDcfEvalTestParams{64, 1},
                                           CpuDcfEvalTestParams{64, 63},
                                           CpuDcfEvalTestParams{64, 64}));

INSTANTIATE_TEST_SUITE_P(,
                         CpuDcfEvalTestU128,
                         ::testing::Values(CpuDcfEvalTestParams{1, 1},
                                           CpuDcfEvalTestParams{1, 127},
                                           CpuDcfEvalTestParams{1, 128},
                                           CpuDcfEvalTestParams{127, 1},
                                           CpuDcfEvalTestParams{127, 127},
                                           CpuDcfEvalTestParams{127, 128},
                                           CpuDcfEvalTestParams{128, 1},
                                           CpuDcfEvalTestParams{128, 127},
                                           CpuDcfEvalTestParams{128, 128}));

} // namespace FastFss::tests::dcf
