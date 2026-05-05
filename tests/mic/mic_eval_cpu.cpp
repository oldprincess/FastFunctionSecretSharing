#include <FastFssPP/config.h>
#include <FastFssPP/mic.h>
#include <FastFssPP/prng.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>
#include <wideint/wideint.hpp>

#include "../../src/kernel/mic.h"
#include "../../src/kernel/parallel_execute.h"
#include "mic_eval_test_case.h"

namespace FastFss::tests::mic {

static_assert(kernel::detail::has_strided_support_v<kernel::DcfMICEvalTask<std::uint32_t>>,
              "DcfMICEvalTask should expose fine-grained parallel interfaces");

struct CpuMicEvalTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

template <typename T>
class CpuMicEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CpuMicEvalTestParams>
{
public:
    using Type = T;
};

class CpuMicEvalTestU8 : public CpuMicEvalTestBase<std::uint8_t>
{
};
class CpuMicEvalTestU16 : public CpuMicEvalTestBase<std::uint16_t>
{
};
class CpuMicEvalTestU32 : public CpuMicEvalTestBase<std::uint32_t>
{
};
class CpuMicEvalTestU64 : public CpuMicEvalTestBase<std::uint64_t>
{
};
class CpuMicEvalTestU128 : public CpuMicEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCpuMicEvalTestBody(Fixture* self)
{
    static std::vector<std::size_t> kElementNumConfigs    = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kIntervalCountConfigs = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CpuMicEvalTestParams params = self->GetParam();

    const std::size_t bitWidthIn  = params.bitWidthIn;
    const std::size_t bitWidthOut = params.bitWidthOut;
    const std::size_t elementSize = sizeof(T);
    const std::size_t valueBits   = sizeof(T) * 8;
    const T           maskOut     = (bitWidthOut >= valueBits) ? ~T(0) : ((T(1) << bitWidthOut) - T(1));

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t intervalCount : kIntervalCountConfigs)
        {
            MicEvalTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, intervalCount);

            std::size_t keySize = FastFss::mic::dcfMICGetKeyDataSize(bitWidthIn, bitWidthOut, elementSize, elementNum);
            std::size_t cacheSize =
                FastFss::mic::dcfMICGetCacheDataSize(bitWidthIn, bitWidthOut, elementSize, elementNum);

            std::vector<std::uint8_t> key(keySize);
            std::vector<T>            z(elementNum * intervalCount);
            FastFss::mic::cpu::dcfMICKeyGen<T>(
                std::span<std::uint8_t>(key.data(), key.size()), std::span<T>(z.data(), z.size()),
                std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
                std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
                std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()),
                std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), bitWidthIn,
                bitWidthOut);

            std::vector<T> sharedZ0(z.size());
            std::vector<T> sharedZ1(z.size());
            {
                FastFss::prng::cpu::Prng zPrng;
                ASSERT_NO_THROW(zPrng.generate(sharedZ0.data(), bitWidthOut, sharedZ0.size()));
                for (std::size_t i = 0; i < z.size(); ++i)
                {
                    sharedZ0[i] &= maskOut;
                    sharedZ1[i] = (z[i] - sharedZ0[i]) & maskOut;
                }
            }

            std::vector<T>            share0(z.size());
            std::vector<T>            share1(z.size());
            std::vector<std::uint8_t> evalCache(cacheSize);
            FastFss::mic::cpu::dcfMICEval<T>(
                std::span<T>(share0.data(), share0.size()),
                std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                std::span<const std::uint8_t>(key.data(), key.size()),
                std::span<const T>(sharedZ0.data(), sharedZ0.size()),
                std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), 0,
                std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), bitWidthIn,
                bitWidthOut, std::span<std::uint8_t>(evalCache.data(), evalCache.size()));
            FastFss::mic::cpu::dcfMICEval<T>(
                std::span<T>(share1.data(), share1.size()),
                std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                std::span<const std::uint8_t>(key.data(), key.size()),
                std::span<const T>(sharedZ1.data(), sharedZ1.size()),
                std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), 1,
                std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), bitWidthIn,
                bitWidthOut, std::span<std::uint8_t>{});

            for (std::size_t i = 0; i < share0.size(); ++i)
            {
                const T out = (share0[i] + share1[i]) & maskOut;
                ASSERT_EQ(out, testCase.expected[i]);
            }
        }
    }
}

TEST(CpuMicEvalFineGrainedPath, ProducesExpectedOutput)
{
    using T = std::uint32_t;

    const int previousNumThreads = FastFss::config::cpu::getNumThreads();
    struct ThreadCountGuard
    {
        int previous;
        ~ThreadCountGuard()
        {
            FastFss::config::cpu::setNumThreads(previous);
        }
    } guard{previousNumThreads};
    FastFss::config::cpu::setNumThreads(8);

    const std::size_t bitWidthIn    = 32;
    const std::size_t bitWidthOut   = 32;
    const std::size_t elementNum    = 1;
    const std::size_t intervalCount = 64;
    const std::size_t elementSize   = sizeof(T);

    MicEvalTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, intervalCount);
    const std::size_t  keySize = FastFss::mic::dcfMICGetKeyDataSize(bitWidthIn, bitWidthOut, elementSize, elementNum);

    std::vector<std::uint8_t> key(keySize);
    std::vector<T>            z(elementNum * intervalCount);
    FastFss::mic::cpu::dcfMICKeyGen<T>(
        std::span<std::uint8_t>(key.data(), key.size()), std::span<T>(z.data(), z.size()),
        std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
        std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
        std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()),
        std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
        std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), bitWidthIn, bitWidthOut);

    std::vector<T> sharedOut0(z.size());
    std::vector<T> sharedOut1(z.size());
    std::vector<T> sharedZ0(z.size(), 0);
    std::vector<T> sharedZ1 = z;

    ASSERT_NO_THROW(FastFss::mic::cpu::dcfMICEval<T>(
        std::span<T>(sharedOut0.data(), sharedOut0.size()),
        std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
        std::span<const std::uint8_t>(key.data(), key.size()), std::span<const T>(sharedZ0.data(), sharedZ0.size()),
        std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), 0,
        std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
        std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), bitWidthIn, bitWidthOut,
        std::span<std::uint8_t>{}));
    ASSERT_NO_THROW(FastFss::mic::cpu::dcfMICEval<T>(
        std::span<T>(sharedOut1.data(), sharedOut1.size()),
        std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
        std::span<const std::uint8_t>(key.data(), key.size()), std::span<const T>(sharedZ1.data(), sharedZ1.size()),
        std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), 1,
        std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
        std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), bitWidthIn, bitWidthOut,
        std::span<std::uint8_t>{}));

    for (std::size_t i = 0; i < z.size(); ++i)
    {
        ASSERT_EQ(sharedOut0[i] + sharedOut1[i], testCase.expected[i]);
    }
}

TEST_P(CpuMicEvalTestU8, RunTestBody)
{
    RunCpuMicEvalTestBody(this);
}
TEST_P(CpuMicEvalTestU16, RunTestBody)
{
    RunCpuMicEvalTestBody(this);
}
TEST_P(CpuMicEvalTestU32, RunTestBody)
{
    RunCpuMicEvalTestBody(this);
}
TEST_P(CpuMicEvalTestU64, RunTestBody)
{
    RunCpuMicEvalTestBody(this);
}
TEST_P(CpuMicEvalTestU128, RunTestBody)
{
    RunCpuMicEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CpuMicEvalTestU8,
                         ::testing::Values(CpuMicEvalTestParams{1, 1},
                                           CpuMicEvalTestParams{1, 7},
                                           CpuMicEvalTestParams{1, 8},
                                           CpuMicEvalTestParams{7, 1},
                                           CpuMicEvalTestParams{7, 7},
                                           CpuMicEvalTestParams{7, 8},
                                           CpuMicEvalTestParams{8, 1},
                                           CpuMicEvalTestParams{8, 7},
                                           CpuMicEvalTestParams{8, 8}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuMicEvalTestU16,
                         ::testing::Values(CpuMicEvalTestParams{1, 1},
                                           CpuMicEvalTestParams{1, 15},
                                           CpuMicEvalTestParams{1, 16},
                                           CpuMicEvalTestParams{15, 1},
                                           CpuMicEvalTestParams{15, 15},
                                           CpuMicEvalTestParams{15, 16},
                                           CpuMicEvalTestParams{16, 1},
                                           CpuMicEvalTestParams{16, 15},
                                           CpuMicEvalTestParams{16, 16}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuMicEvalTestU32,
                         ::testing::Values(CpuMicEvalTestParams{1, 1},
                                           CpuMicEvalTestParams{1, 31},
                                           CpuMicEvalTestParams{1, 32},
                                           CpuMicEvalTestParams{31, 1},
                                           CpuMicEvalTestParams{31, 31},
                                           CpuMicEvalTestParams{31, 32},
                                           CpuMicEvalTestParams{32, 1},
                                           CpuMicEvalTestParams{32, 31},
                                           CpuMicEvalTestParams{32, 32}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuMicEvalTestU64,
                         ::testing::Values(CpuMicEvalTestParams{1, 1},
                                           CpuMicEvalTestParams{1, 63},
                                           CpuMicEvalTestParams{1, 64},
                                           CpuMicEvalTestParams{63, 1},
                                           CpuMicEvalTestParams{63, 63},
                                           CpuMicEvalTestParams{63, 64},
                                           CpuMicEvalTestParams{64, 1},
                                           CpuMicEvalTestParams{64, 63},
                                           CpuMicEvalTestParams{64, 64}));
INSTANTIATE_TEST_SUITE_P(,
                         CpuMicEvalTestU128,
                         ::testing::Values(CpuMicEvalTestParams{1, 1},
                                           CpuMicEvalTestParams{1, 127},
                                           CpuMicEvalTestParams{1, 128},
                                           CpuMicEvalTestParams{127, 1},
                                           CpuMicEvalTestParams{127, 127},
                                           CpuMicEvalTestParams{127, 128},
                                           CpuMicEvalTestParams{128, 1},
                                           CpuMicEvalTestParams{128, 127},
                                           CpuMicEvalTestParams{128, 128}));

} // namespace FastFss::tests::mic
