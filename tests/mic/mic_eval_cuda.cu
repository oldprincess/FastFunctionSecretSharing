#include <FastFssPP/mic.h>
#include <FastFssPP/prng.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "mic_eval_test_case.h"

namespace FastFss::tests::mic {

struct CudaMicEvalTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

template <typename T>
class CudaMicEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CudaMicEvalTestParams>
{
public:
    using Type = T;
};

class CudaMicEvalTestU8 : public CudaMicEvalTestBase<std::uint8_t>
{
};
class CudaMicEvalTestU16 : public CudaMicEvalTestBase<std::uint16_t>
{
};
class CudaMicEvalTestU32 : public CudaMicEvalTestBase<std::uint32_t>
{
};
class CudaMicEvalTestU64 : public CudaMicEvalTestBase<std::uint64_t>
{
};
class CudaMicEvalTestU128 : public CudaMicEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCudaMicEvalTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs    = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kIntervalCountConfigs = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CudaMicEvalTestParams params = self->GetParam();

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

            thrust::device_vector<T>            alpha_d(testCase.alpha.size());
            thrust::device_vector<T>            maskedX_d(testCase.maskedX.size());
            thrust::device_vector<std::uint8_t> seed0_d(testCase.seed0.size());
            thrust::device_vector<std::uint8_t> seed1_d(testCase.seed1.size());
            thrust::device_vector<T>            left_d(testCase.leftEndpoints.size());
            thrust::device_vector<T>            right_d(testCase.rightEndpoints.size());
            thrust::copy(testCase.alpha.begin(), testCase.alpha.end(), alpha_d.begin());
            thrust::copy(testCase.maskedX.begin(), testCase.maskedX.end(), maskedX_d.begin());
            thrust::copy(testCase.seed0.begin(), testCase.seed0.end(), seed0_d.begin());
            thrust::copy(testCase.seed1.begin(), testCase.seed1.end(), seed1_d.begin());
            thrust::copy(testCase.leftEndpoints.begin(), testCase.leftEndpoints.end(), left_d.begin());
            thrust::copy(testCase.rightEndpoints.begin(), testCase.rightEndpoints.end(), right_d.begin());

            thrust::device_vector<std::uint8_t> key_d(keySize);
            thrust::device_vector<T>            z_d(elementNum * intervalCount);

            FastFss::mic::cuda::dcfMICKeyGen<T>(
                std::span<std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<T>(thrust::raw_pointer_cast(z_d.data()), z_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(alpha_d.data()), alpha_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(left_d.data()), left_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(right_d.data()), right_d.size()), bitWidthIn, bitWidthOut,
                nullptr);

            std::vector<T> z(elementNum * intervalCount);
            thrust::copy(z_d.begin(), z_d.end(), z.begin());

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

            thrust::device_vector<T> sharedZ0_d(sharedZ0.begin(), sharedZ0.end());
            thrust::device_vector<T> sharedZ1_d(sharedZ1.begin(), sharedZ1.end());

            thrust::device_vector<T>            share0_d(z.size());
            thrust::device_vector<T>            share1_d(z.size());
            thrust::device_vector<std::uint8_t> cache_d(cacheSize);

            FastFss::mic::cuda::dcfMICEval<T>(
                std::span<T>(thrust::raw_pointer_cast(share0_d.data()), share0_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(sharedZ0_d.data()), sharedZ0_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), 0,
                std::span<const T>(thrust::raw_pointer_cast(left_d.data()), left_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(right_d.data()), right_d.size()), bitWidthIn, bitWidthOut,
                std::span<std::uint8_t>(thrust::raw_pointer_cast(cache_d.data()), cache_d.size()), nullptr);
            FastFss::mic::cuda::dcfMICEval<T>(
                std::span<T>(thrust::raw_pointer_cast(share1_d.data()), share1_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(sharedZ1_d.data()), sharedZ1_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), 1,
                std::span<const T>(thrust::raw_pointer_cast(left_d.data()), left_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(right_d.data()), right_d.size()), bitWidthIn, bitWidthOut,
                std::span<std::uint8_t>{}, nullptr);

            std::vector<T> share0(z.size());
            std::vector<T> share1(z.size());
            thrust::copy(share0_d.begin(), share0_d.end(), share0.begin());
            thrust::copy(share1_d.begin(), share1_d.end(), share1.begin());

            for (std::size_t i = 0; i < share0.size(); ++i)
            {
                const T out = (share0[i] + share1[i]) & maskOut;
                ASSERT_EQ(out, testCase.expected[i]);
            }
        }
    }
}

TEST_P(CudaMicEvalTestU8, RunTestBody)
{
    RunCudaMicEvalTestBody(this);
}
TEST_P(CudaMicEvalTestU16, RunTestBody)
{
    RunCudaMicEvalTestBody(this);
}
TEST_P(CudaMicEvalTestU32, RunTestBody)
{
    RunCudaMicEvalTestBody(this);
}
TEST_P(CudaMicEvalTestU64, RunTestBody)
{
    RunCudaMicEvalTestBody(this);
}
TEST_P(CudaMicEvalTestU128, RunTestBody)
{
    RunCudaMicEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CudaMicEvalTestU8,
                         ::testing::Values(CudaMicEvalTestParams{1, 1},
                                           CudaMicEvalTestParams{1, 7},
                                           CudaMicEvalTestParams{1, 8},
                                           CudaMicEvalTestParams{7, 1},
                                           CudaMicEvalTestParams{7, 7},
                                           CudaMicEvalTestParams{7, 8},
                                           CudaMicEvalTestParams{8, 1},
                                           CudaMicEvalTestParams{8, 7},
                                           CudaMicEvalTestParams{8, 8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaMicEvalTestU16,
                         ::testing::Values(CudaMicEvalTestParams{1, 1},
                                           CudaMicEvalTestParams{1, 15},
                                           CudaMicEvalTestParams{1, 16},
                                           CudaMicEvalTestParams{15, 1},
                                           CudaMicEvalTestParams{15, 15},
                                           CudaMicEvalTestParams{15, 16},
                                           CudaMicEvalTestParams{16, 1},
                                           CudaMicEvalTestParams{16, 15},
                                           CudaMicEvalTestParams{16, 16}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaMicEvalTestU32,
                         ::testing::Values(CudaMicEvalTestParams{1, 1},
                                           CudaMicEvalTestParams{1, 31},
                                           CudaMicEvalTestParams{1, 32},
                                           CudaMicEvalTestParams{31, 1},
                                           CudaMicEvalTestParams{31, 31},
                                           CudaMicEvalTestParams{31, 32},
                                           CudaMicEvalTestParams{32, 1},
                                           CudaMicEvalTestParams{32, 31},
                                           CudaMicEvalTestParams{32, 32}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaMicEvalTestU64,
                         ::testing::Values(CudaMicEvalTestParams{1, 1},
                                           CudaMicEvalTestParams{1, 63},
                                           CudaMicEvalTestParams{1, 64},
                                           CudaMicEvalTestParams{63, 1},
                                           CudaMicEvalTestParams{63, 63},
                                           CudaMicEvalTestParams{63, 64},
                                           CudaMicEvalTestParams{64, 1},
                                           CudaMicEvalTestParams{64, 63},
                                           CudaMicEvalTestParams{64, 64}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaMicEvalTestU128,
                         ::testing::Values(CudaMicEvalTestParams{1, 1},
                                           CudaMicEvalTestParams{1, 127},
                                           CudaMicEvalTestParams{1, 128},
                                           CudaMicEvalTestParams{127, 1},
                                           CudaMicEvalTestParams{127, 127},
                                           CudaMicEvalTestParams{127, 128},
                                           CudaMicEvalTestParams{128, 1},
                                           CudaMicEvalTestParams{128, 127},
                                           CudaMicEvalTestParams{128, 128}));

} // namespace FastFss::tests::mic
