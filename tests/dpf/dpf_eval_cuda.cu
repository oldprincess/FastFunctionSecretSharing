#include <FastFssPP/dpf.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "dpf_eval_case.h"

namespace FastFss::tests::dpf {

struct CudaDpfEvalTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

template <typename T>
class CudaDpfEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CudaDpfEvalTestParams>
{
public:
    using Type = T;
};

class CudaDpfEvalTestU8 : public CudaDpfEvalTestBase<std::uint8_t>
{
};
class CudaDpfEvalTestU16 : public CudaDpfEvalTestBase<std::uint16_t>
{
};
class CudaDpfEvalTestU32 : public CudaDpfEvalTestBase<std::uint32_t>
{
};
class CudaDpfEvalTestU64 : public CudaDpfEvalTestBase<std::uint64_t>
{
};
class CudaDpfEvalTestU128 : public CudaDpfEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCudaDpfEvalTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kGroupSizeConfigs  = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CudaDpfEvalTestParams params = self->GetParam();

    std::size_t bitWidthIn  = params.bitWidthIn;
    std::size_t bitWidthOut = params.bitWidthOut;
    std::size_t elementSize = sizeof(T);
    std::size_t valueBits   = sizeof(T) * 8;
    T           maskOut     = (bitWidthOut >= valueBits) ? ~T(0) : ((T(1) << bitWidthOut) - T(1));

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t groupSize : kGroupSizeConfigs)
        {
            DpfEvalTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, groupSize);

            std::size_t keyDataSize =
                FastFss::dpf::dpfGetKeyDataSize(bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
            std::size_t cacheDataSize = FastFss::dpf::dpfGetCacheDataSize(bitWidthIn, elementSize, elementNum);

            thrust::device_vector<T>            alpha_d(testCase.alpha.size());
            thrust::device_vector<T>            beta_d(testCase.beta.size());
            thrust::device_vector<T>            maskedX_d(testCase.maskedX.size());
            thrust::device_vector<std::uint8_t> seed0_d(testCase.seed0.size());
            thrust::device_vector<std::uint8_t> seed1_d(testCase.seed1.size());
            thrust::copy(testCase.alpha.begin(), testCase.alpha.end(), alpha_d.begin());
            thrust::copy(testCase.beta.begin(), testCase.beta.end(), beta_d.begin());
            thrust::copy(testCase.maskedX.begin(), testCase.maskedX.end(), maskedX_d.begin());
            thrust::copy(testCase.seed0.begin(), testCase.seed0.end(), seed0_d.begin());
            thrust::copy(testCase.seed1.begin(), testCase.seed1.end(), seed1_d.begin());

            thrust::device_vector<std::uint8_t> key_d(keyDataSize);
            thrust::device_vector<std::uint8_t> evalCache_d(cacheDataSize);
            FastFss::dpf::cuda::dpfKeyGen<T>(
                std::span<std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(alpha_d.data()), alpha_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(beta_d.data()), beta_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), bitWidthIn,
                bitWidthOut, groupSize, nullptr);

            std::vector<T>           y0(elementNum * groupSize);
            std::vector<T>           y1(elementNum * groupSize);
            thrust::device_vector<T> y0_d(y0.size());
            thrust::device_vector<T> y1_d(y1.size());

            FastFss::dpf::cuda::dpfEval<T>(
                std::span<T>(thrust::raw_pointer_cast(y0_d.data()), y0_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), 0, bitWidthIn,
                bitWidthOut, groupSize,
                std::span<std::uint8_t>(thrust::raw_pointer_cast(evalCache_d.data()), evalCache_d.size()), nullptr);
            FastFss::dpf::cuda::dpfEval<T>(
                std::span<T>(thrust::raw_pointer_cast(y1_d.data()), y1_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), 1, bitWidthIn,
                bitWidthOut, groupSize, std::span<std::uint8_t>{}, nullptr);

            thrust::copy(y0_d.begin(), y0_d.end(), y0.begin());
            thrust::copy(y1_d.begin(), y1_d.end(), y1.begin());
            for (std::size_t i = 0; i < y0.size(); ++i)
            {
                ASSERT_EQ((y0[i] + y1[i]) & maskOut, testCase.expected[i]);
            }
        }
    }
}

TEST_P(CudaDpfEvalTestU8, RunTestBody)
{
    RunCudaDpfEvalTestBody(this);
}
TEST_P(CudaDpfEvalTestU16, RunTestBody)
{
    RunCudaDpfEvalTestBody(this);
}
TEST_P(CudaDpfEvalTestU32, RunTestBody)
{
    RunCudaDpfEvalTestBody(this);
}
TEST_P(CudaDpfEvalTestU64, RunTestBody)
{
    RunCudaDpfEvalTestBody(this);
}
TEST_P(CudaDpfEvalTestU128, RunTestBody)
{
    RunCudaDpfEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalTestU8,
                         ::testing::Values(CudaDpfEvalTestParams{1, 1},
                                           CudaDpfEvalTestParams{1, 7},
                                           CudaDpfEvalTestParams{1, 8},
                                           CudaDpfEvalTestParams{7, 1},
                                           CudaDpfEvalTestParams{7, 7},
                                           CudaDpfEvalTestParams{7, 8},
                                           CudaDpfEvalTestParams{8, 1},
                                           CudaDpfEvalTestParams{8, 7},
                                           CudaDpfEvalTestParams{8, 8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalTestU16,
                         ::testing::Values(CudaDpfEvalTestParams{1, 1},
                                           CudaDpfEvalTestParams{1, 15},
                                           CudaDpfEvalTestParams{1, 16},
                                           CudaDpfEvalTestParams{15, 1},
                                           CudaDpfEvalTestParams{15, 15},
                                           CudaDpfEvalTestParams{15, 16},
                                           CudaDpfEvalTestParams{16, 1},
                                           CudaDpfEvalTestParams{16, 15},
                                           CudaDpfEvalTestParams{16, 16}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalTestU32,
                         ::testing::Values(CudaDpfEvalTestParams{1, 1},
                                           CudaDpfEvalTestParams{1, 31},
                                           CudaDpfEvalTestParams{1, 32},
                                           CudaDpfEvalTestParams{31, 1},
                                           CudaDpfEvalTestParams{31, 31},
                                           CudaDpfEvalTestParams{31, 32},
                                           CudaDpfEvalTestParams{32, 1},
                                           CudaDpfEvalTestParams{32, 31},
                                           CudaDpfEvalTestParams{32, 32}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalTestU64,
                         ::testing::Values(CudaDpfEvalTestParams{1, 1},
                                           CudaDpfEvalTestParams{1, 63},
                                           CudaDpfEvalTestParams{1, 64},
                                           CudaDpfEvalTestParams{63, 1},
                                           CudaDpfEvalTestParams{63, 63},
                                           CudaDpfEvalTestParams{63, 64},
                                           CudaDpfEvalTestParams{64, 1},
                                           CudaDpfEvalTestParams{64, 63},
                                           CudaDpfEvalTestParams{64, 64}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalTestU128,
                         ::testing::Values(CudaDpfEvalTestParams{1, 1},
                                           CudaDpfEvalTestParams{1, 127},
                                           CudaDpfEvalTestParams{1, 128},
                                           CudaDpfEvalTestParams{127, 1},
                                           CudaDpfEvalTestParams{127, 127},
                                           CudaDpfEvalTestParams{127, 128},
                                           CudaDpfEvalTestParams{128, 1},
                                           CudaDpfEvalTestParams{128, 127},
                                           CudaDpfEvalTestParams{128, 128}));

} // namespace FastFss::tests::dpf
