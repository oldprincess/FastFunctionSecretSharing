#include <FastFssPP/dpf.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "dpf_eval_all_case.h"

namespace FastFss::tests::dpf {

struct CudaDpfEvalAllTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

template <typename T>
class CudaDpfEvalAllTestBase : public ::testing::Test, public ::testing::WithParamInterface<CudaDpfEvalAllTestParams>
{
public:
    using Type = T;
};

class CudaDpfEvalAllTestU8 : public CudaDpfEvalAllTestBase<std::uint8_t>
{
};
class CudaDpfEvalAllTestU16 : public CudaDpfEvalAllTestBase<std::uint16_t>
{
};
class CudaDpfEvalAllTestU32 : public CudaDpfEvalAllTestBase<std::uint32_t>
{
};
class CudaDpfEvalAllTestU64 : public CudaDpfEvalAllTestBase<std::uint64_t>
{
};
class CudaDpfEvalAllTestU128 : public CudaDpfEvalAllTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCudaDpfEvalAllTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kGroupSizeConfigs  = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CudaDpfEvalAllTestParams params = self->GetParam();

    std::size_t bitWidthIn  = params.bitWidthIn;
    std::size_t bitWidthOut = params.bitWidthOut;
    std::size_t elementSize = sizeof(T);
    T           maskOut     = (bitWidthOut == elementSize * 8) ? ~T(0) : (T(1) << bitWidthOut) - 1;
    std::size_t domainSize  = std::size_t{1} << bitWidthIn;

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t groupSize : kGroupSizeConfigs)
        {
            DpfEvalAllTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, groupSize);

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

            std::vector<T>           y0(elementNum * domainSize * groupSize);
            std::vector<T>           y1(elementNum * domainSize * groupSize);
            thrust::device_vector<T> y0_d(y0.size());
            thrust::device_vector<T> y1_d(y1.size());

            FastFss::dpf::cuda::dpfEvalAll<T>(
                std::span<T>(thrust::raw_pointer_cast(y0_d.data()), y0_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), 0, bitWidthIn,
                bitWidthOut, groupSize,
                std::span<std::uint8_t>(thrust::raw_pointer_cast(evalCache_d.data()), evalCache_d.size()), nullptr);
            FastFss::dpf::cuda::dpfEvalAll<T>(
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

TEST_P(CudaDpfEvalAllTestU8, RunTestBody)
{
    RunCudaDpfEvalAllTestBody(this);
}
TEST_P(CudaDpfEvalAllTestU16, RunTestBody)
{
    RunCudaDpfEvalAllTestBody(this);
}
TEST_P(CudaDpfEvalAllTestU32, RunTestBody)
{
    RunCudaDpfEvalAllTestBody(this);
}
TEST_P(CudaDpfEvalAllTestU64, RunTestBody)
{
    RunCudaDpfEvalAllTestBody(this);
}
TEST_P(CudaDpfEvalAllTestU128, RunTestBody)
{
    RunCudaDpfEvalAllTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalAllTestU8,
                         ::testing::Values(CudaDpfEvalAllTestParams{1, 1},
                                           CudaDpfEvalAllTestParams{1, 7},
                                           CudaDpfEvalAllTestParams{1, 8},
                                           CudaDpfEvalAllTestParams{7, 1},
                                           CudaDpfEvalAllTestParams{7, 7},
                                           CudaDpfEvalAllTestParams{7, 8},
                                           CudaDpfEvalAllTestParams{8, 1},
                                           CudaDpfEvalAllTestParams{8, 7},
                                           CudaDpfEvalAllTestParams{8, 8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalAllTestU16,
                         ::testing::Values(CudaDpfEvalAllTestParams{1, 1},
                                           CudaDpfEvalAllTestParams{1, 9},
                                           CudaDpfEvalAllTestParams{1, 10},
                                           CudaDpfEvalAllTestParams{9, 1},
                                           CudaDpfEvalAllTestParams{9, 9},
                                           CudaDpfEvalAllTestParams{9, 10},
                                           CudaDpfEvalAllTestParams{10, 1},
                                           CudaDpfEvalAllTestParams{10, 9},
                                           CudaDpfEvalAllTestParams{10, 10}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalAllTestU32,
                         ::testing::Values(CudaDpfEvalAllTestParams{1, 1},
                                           CudaDpfEvalAllTestParams{1, 9},
                                           CudaDpfEvalAllTestParams{1, 10},
                                           CudaDpfEvalAllTestParams{9, 1},
                                           CudaDpfEvalAllTestParams{9, 9},
                                           CudaDpfEvalAllTestParams{9, 10},
                                           CudaDpfEvalAllTestParams{10, 1},
                                           CudaDpfEvalAllTestParams{10, 9},
                                           CudaDpfEvalAllTestParams{10, 10}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalAllTestU64,
                         ::testing::Values(CudaDpfEvalAllTestParams{1, 1},
                                           CudaDpfEvalAllTestParams{1, 9},
                                           CudaDpfEvalAllTestParams{1, 10},
                                           CudaDpfEvalAllTestParams{9, 1},
                                           CudaDpfEvalAllTestParams{9, 9},
                                           CudaDpfEvalAllTestParams{9, 10},
                                           CudaDpfEvalAllTestParams{10, 1},
                                           CudaDpfEvalAllTestParams{10, 9},
                                           CudaDpfEvalAllTestParams{10, 10}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalAllTestU128,
                         ::testing::Values(CudaDpfEvalAllTestParams{1, 1},
                                           CudaDpfEvalAllTestParams{1, 9},
                                           CudaDpfEvalAllTestParams{1, 10},
                                           CudaDpfEvalAllTestParams{9, 1},
                                           CudaDpfEvalAllTestParams{9, 9},
                                           CudaDpfEvalAllTestParams{9, 10},
                                           CudaDpfEvalAllTestParams{10, 1},
                                           CudaDpfEvalAllTestParams{10, 9},
                                           CudaDpfEvalAllTestParams{10, 10}));

} // namespace FastFss::tests::dpf
