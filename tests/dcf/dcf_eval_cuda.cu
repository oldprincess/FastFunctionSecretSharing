#include <FastFssPP/dcf.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "dcf_eval_test_case.h"

namespace FastFss::tests::dcf {

struct CudaDcfEvalTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

template <typename T>
class CudaDcfEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CudaDcfEvalTestParams>
{
public:
    using Type = T;
};

class CudaDcfEvalTestU8 : public CudaDcfEvalTestBase<std::uint8_t>
{
};

class CudaDcfEvalTestU16 : public CudaDcfEvalTestBase<std::uint16_t>
{
};

class CudaDcfEvalTestU32 : public CudaDcfEvalTestBase<std::uint32_t>
{
};

class CudaDcfEvalTestU64 : public CudaDcfEvalTestBase<std::uint64_t>
{
};

class CudaDcfEvalTestU128 : public CudaDcfEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCudaDcfEvalTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kGroupSizeConfigs  = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CudaDcfEvalTestParams params = self->GetParam();

    std::size_t bitWidthIn  = params.bitWidthIn;
    std::size_t bitWidthOut = params.bitWidthOut;
    std::size_t elementSize = sizeof(T);

    T maskOut = (bitWidthOut == elementSize * 8) ? ~T(0) : (T(1) << bitWidthOut) - 1;

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t groupSize : kGroupSizeConfigs)
        {
            DcfEvalTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, groupSize);

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

            // generate DCF key
            std::size_t keyDataSize =
                FastFss::dcf::dcfGetKeyDataSize(bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);

            thrust::device_vector<std::uint8_t> key_d(keyDataSize);
            FastFss::dcf::cuda::dcfKeyGen<T>(
                std::span<std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(alpha_d.data()), alpha_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(beta_d.data()), beta_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), bitWidthIn,
                bitWidthOut, groupSize, nullptr);

            // generate DCF cache
            std::size_t cacheDataSize =
                FastFss::dcf::dcfGetCacheDataSize(bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
            thrust::device_vector<std::uint8_t> evalCache_d(cacheDataSize);

            // eval
            std::vector<T>           y0(elementNum * groupSize);
            std::vector<T>           y1(elementNum * groupSize);
            thrust::device_vector<T> y0_d(y0.size());
            thrust::device_vector<T> y1_d(y1.size());

            FastFss::dcf::cuda::dcfEval<T>(
                std::span<T>(thrust::raw_pointer_cast(y0_d.data()), y0_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), 0, bitWidthIn,
                bitWidthOut, groupSize, std::span<std::uint8_t>{}, nullptr);
            FastFss::dcf::cuda::dcfEval<T>(
                std::span<T>(thrust::raw_pointer_cast(y1_d.data()), y1_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), 1, bitWidthIn,
                bitWidthOut, groupSize,
                std::span<std::uint8_t>(thrust::raw_pointer_cast(evalCache_d.data()), evalCache_d.size()), nullptr);

            thrust::copy(y0_d.begin(), y0_d.end(), y0.begin());
            thrust::copy(y1_d.begin(), y1_d.end(), y1.begin());

            // check
            for (std::size_t i = 0; i < y0.size(); ++i)
            {
                T y = (y0[i] + y1[i]) & maskOut;
                ASSERT_EQ(y, testCase.y[i]);
            }
        }
    }
}

TEST_P(CudaDcfEvalTestU8, RunTestBody)
{
    RunCudaDcfEvalTestBody(this);
}

TEST_P(CudaDcfEvalTestU16, RunTestBody)
{
    RunCudaDcfEvalTestBody(this);
}

TEST_P(CudaDcfEvalTestU32, RunTestBody)
{
    RunCudaDcfEvalTestBody(this);
}

TEST_P(CudaDcfEvalTestU64, RunTestBody)
{
    RunCudaDcfEvalTestBody(this);
}

TEST_P(CudaDcfEvalTestU128, RunTestBody)
{
    RunCudaDcfEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CudaDcfEvalTestU8,
                         ::testing::Values(CudaDcfEvalTestParams{1, 1},
                                           CudaDcfEvalTestParams{1, 7},
                                           CudaDcfEvalTestParams{1, 8},
                                           CudaDcfEvalTestParams{7, 1},
                                           CudaDcfEvalTestParams{7, 7},
                                           CudaDcfEvalTestParams{7, 8},
                                           CudaDcfEvalTestParams{8, 1},
                                           CudaDcfEvalTestParams{8, 7},
                                           CudaDcfEvalTestParams{8, 8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDcfEvalTestU16,
                         ::testing::Values(CudaDcfEvalTestParams{1, 1},
                                           CudaDcfEvalTestParams{1, 15},
                                           CudaDcfEvalTestParams{1, 16},
                                           CudaDcfEvalTestParams{15, 1},
                                           CudaDcfEvalTestParams{15, 15},
                                           CudaDcfEvalTestParams{15, 16},
                                           CudaDcfEvalTestParams{16, 1},
                                           CudaDcfEvalTestParams{16, 15},
                                           CudaDcfEvalTestParams{16, 16}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaDcfEvalTestU32,
                         ::testing::Values(CudaDcfEvalTestParams{1, 1},
                                           CudaDcfEvalTestParams{1, 31},
                                           CudaDcfEvalTestParams{1, 32},
                                           CudaDcfEvalTestParams{31, 1},
                                           CudaDcfEvalTestParams{31, 31},
                                           CudaDcfEvalTestParams{31, 32},
                                           CudaDcfEvalTestParams{32, 1},
                                           CudaDcfEvalTestParams{32, 31},
                                           CudaDcfEvalTestParams{32, 32}));

INSTANTIATE_TEST_SUITE_P(,
                         CudaDcfEvalTestU64,
                         ::testing::Values(CudaDcfEvalTestParams{1, 1},
                                           CudaDcfEvalTestParams{1, 63},
                                           CudaDcfEvalTestParams{1, 64},
                                           CudaDcfEvalTestParams{63, 1},
                                           CudaDcfEvalTestParams{63, 63},
                                           CudaDcfEvalTestParams{63, 64},
                                           CudaDcfEvalTestParams{64, 1},
                                           CudaDcfEvalTestParams{64, 63},
                                           CudaDcfEvalTestParams{64, 64}));

INSTANTIATE_TEST_SUITE_P(,
                         CudaDcfEvalTestU128,
                         ::testing::Values(CudaDcfEvalTestParams{1, 1},
                                           CudaDcfEvalTestParams{1, 127},
                                           CudaDcfEvalTestParams{1, 128},
                                           CudaDcfEvalTestParams{127, 1},
                                           CudaDcfEvalTestParams{127, 127},
                                           CudaDcfEvalTestParams{127, 128},
                                           CudaDcfEvalTestParams{128, 1},
                                           CudaDcfEvalTestParams{128, 127},
                                           CudaDcfEvalTestParams{128, 128}));

} // namespace FastFss::tests::dcf
