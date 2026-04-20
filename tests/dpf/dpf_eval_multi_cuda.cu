#include <FastFssPP/dpf.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "dpf_eval_multi_case.h"

namespace FastFss::tests::dpf {

struct CudaDpfEvalMultiTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
    std::size_t pointNum;
};

template <typename T>
class CudaDpfEvalMultiTestBase : public ::testing::Test,
                                 public ::testing::WithParamInterface<CudaDpfEvalMultiTestParams>
{
public:
    using Type = T;
};

class CudaDpfEvalMultiTestU8 : public CudaDpfEvalMultiTestBase<std::uint8_t>
{
};
class CudaDpfEvalMultiTestU16 : public CudaDpfEvalMultiTestBase<std::uint16_t>
{
};
class CudaDpfEvalMultiTestU32 : public CudaDpfEvalMultiTestBase<std::uint32_t>
{
};
class CudaDpfEvalMultiTestU64 : public CudaDpfEvalMultiTestBase<std::uint64_t>
{
};
class CudaDpfEvalMultiTestU128 : public CudaDpfEvalMultiTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCudaDpfEvalMultiTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kGroupSizeConfigs  = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CudaDpfEvalMultiTestParams params      = self->GetParam();
    const std::size_t          bitWidthIn  = params.bitWidthIn;
    const std::size_t          bitWidthOut = params.bitWidthOut;
    const std::size_t          pointNum    = params.pointNum;
    const std::size_t          elementSize = sizeof(T);
    const std::size_t          valueBits   = sizeof(T) * 8;
    const T                    maskOut     = (bitWidthOut >= valueBits) ? ~T(0) : ((T(1) << bitWidthOut) - T(1));

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t groupSize : kGroupSizeConfigs)
        {
            DpfEvalMultiTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, groupSize, pointNum);

            std::size_t keyDataSize =
                FastFss::dpf::dpfGetKeyDataSize(bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
            std::size_t cacheDataSize = FastFss::dpf::dpfGetCacheDataSize(bitWidthIn, elementSize, elementNum);

            thrust::device_vector<T>            alpha_d(testCase.alpha.size());
            thrust::device_vector<T>            beta_d(testCase.beta.size());
            thrust::device_vector<T>            maskedX_d(testCase.maskedX.size());
            thrust::device_vector<T>            points_d(testCase.points.size());
            thrust::device_vector<std::uint8_t> seed0_d(testCase.seed0.size());
            thrust::device_vector<std::uint8_t> seed1_d(testCase.seed1.size());
            thrust::copy(testCase.alpha.begin(), testCase.alpha.end(), alpha_d.begin());
            thrust::copy(testCase.beta.begin(), testCase.beta.end(), beta_d.begin());
            thrust::copy(testCase.maskedX.begin(), testCase.maskedX.end(), maskedX_d.begin());
            thrust::copy(testCase.points.begin(), testCase.points.end(), points_d.begin());
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

            std::vector<T>           y0(elementNum * testCase.points.size() * groupSize);
            std::vector<T>           y1(elementNum * testCase.points.size() * groupSize);
            thrust::device_vector<T> y0_d(y0.size());
            thrust::device_vector<T> y1_d(y1.size());
            FastFss::dpf::cuda::dpfEvalMulti<T>(
                std::span<T>(thrust::raw_pointer_cast(y0_d.data()), y0_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), 0,
                std::span<const T>(thrust::raw_pointer_cast(points_d.data()), points_d.size()), bitWidthIn, bitWidthOut,
                groupSize,
                std::span<std::uint8_t>(thrust::raw_pointer_cast(evalCache_d.data()), evalCache_d.size()), nullptr);
            FastFss::dpf::cuda::dpfEvalMulti<T>(
                std::span<T>(thrust::raw_pointer_cast(y1_d.data()), y1_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), 1,
                std::span<const T>(thrust::raw_pointer_cast(points_d.data()), points_d.size()), bitWidthIn, bitWidthOut,
                groupSize, std::span<std::uint8_t>{}, nullptr);

            thrust::copy(y0_d.begin(), y0_d.end(), y0.begin());
            thrust::copy(y1_d.begin(), y1_d.end(), y1.begin());
            for (std::size_t i = 0; i < y0.size(); ++i)
            {
                ASSERT_EQ((y0[i] + y1[i]) & maskOut, testCase.expected[i]);
            }
        }
    }
}

TEST_P(CudaDpfEvalMultiTestU8, RunTestBody)
{
    RunCudaDpfEvalMultiTestBody(this);
}
TEST_P(CudaDpfEvalMultiTestU16, RunTestBody)
{
    RunCudaDpfEvalMultiTestBody(this);
}
TEST_P(CudaDpfEvalMultiTestU32, RunTestBody)
{
    RunCudaDpfEvalMultiTestBody(this);
}
TEST_P(CudaDpfEvalMultiTestU64, RunTestBody)
{
    RunCudaDpfEvalMultiTestBody(this);
}
TEST_P(CudaDpfEvalMultiTestU128, RunTestBody)
{
    RunCudaDpfEvalMultiTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalMultiTestU8,
                         ::testing::Values(CudaDpfEvalMultiTestParams{1, 1, 1},
                                           CudaDpfEvalMultiTestParams{1, 7, 1},
                                           CudaDpfEvalMultiTestParams{1, 8, 1},
                                           CudaDpfEvalMultiTestParams{7, 1, 3},
                                           CudaDpfEvalMultiTestParams{7, 7, 7},
                                           CudaDpfEvalMultiTestParams{7, 8, 15},
                                           CudaDpfEvalMultiTestParams{8, 1, 31},
                                           CudaDpfEvalMultiTestParams{8, 7, 63},
                                           CudaDpfEvalMultiTestParams{8, 8, 127}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalMultiTestU16,
                         ::testing::Values(CudaDpfEvalMultiTestParams{1, 1, 1},
                                           CudaDpfEvalMultiTestParams{1, 15, 1},
                                           CudaDpfEvalMultiTestParams{1, 16, 1},
                                           CudaDpfEvalMultiTestParams{15, 1, 3},
                                           CudaDpfEvalMultiTestParams{15, 15, 7},
                                           CudaDpfEvalMultiTestParams{15, 16, 15},
                                           CudaDpfEvalMultiTestParams{16, 1, 31},
                                           CudaDpfEvalMultiTestParams{16, 15, 63},
                                           CudaDpfEvalMultiTestParams{16, 16, 127}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalMultiTestU32,
                         ::testing::Values(CudaDpfEvalMultiTestParams{1, 1, 1},
                                           CudaDpfEvalMultiTestParams{1, 31, 1},
                                           CudaDpfEvalMultiTestParams{1, 32, 1},
                                           CudaDpfEvalMultiTestParams{31, 1, 3},
                                           CudaDpfEvalMultiTestParams{31, 31, 7},
                                           CudaDpfEvalMultiTestParams{31, 32, 15},
                                           CudaDpfEvalMultiTestParams{32, 1, 31},
                                           CudaDpfEvalMultiTestParams{32, 31, 63},
                                           CudaDpfEvalMultiTestParams{32, 32, 127}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalMultiTestU64,
                         ::testing::Values(CudaDpfEvalMultiTestParams{1, 1, 1},
                                           CudaDpfEvalMultiTestParams{1, 63, 1},
                                           CudaDpfEvalMultiTestParams{1, 64, 1},
                                           CudaDpfEvalMultiTestParams{63, 1, 3},
                                           CudaDpfEvalMultiTestParams{63, 63, 7},
                                           CudaDpfEvalMultiTestParams{63, 64, 15},
                                           CudaDpfEvalMultiTestParams{64, 1, 31},
                                           CudaDpfEvalMultiTestParams{64, 63, 63},
                                           CudaDpfEvalMultiTestParams{64, 64, 127}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDpfEvalMultiTestU128,
                         ::testing::Values(CudaDpfEvalMultiTestParams{1, 1, 1},
                                           CudaDpfEvalMultiTestParams{1, 127, 1},
                                           CudaDpfEvalMultiTestParams{1, 128, 1},
                                           CudaDpfEvalMultiTestParams{127, 1, 3},
                                           CudaDpfEvalMultiTestParams{127, 127, 7},
                                           CudaDpfEvalMultiTestParams{127, 128, 15},
                                           CudaDpfEvalMultiTestParams{128, 1, 31},
                                           CudaDpfEvalMultiTestParams{128, 127, 63},
                                           CudaDpfEvalMultiTestParams{128, 128, 127}));

} // namespace FastFss::tests::dpf
