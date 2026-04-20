#include <FastFssPP/grotto.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "grotto_interval_lut_eval_test_case.h"

namespace {

template <typename T>
T bit_mask(std::size_t bitWidth)
{
    return (bitWidth == sizeof(T) * 8) ? ~T(0) : (T(1) << bitWidth) - T(1);
}

} // namespace

namespace FastFss::tests::grotto {

struct CudaGrottoIntervalLutEvalTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
    std::size_t intervalCount;
    std::size_t lutNum;
};

template <typename T>
class CudaGrottoIntervalLutEvalTestBase : public ::testing::Test,
                                          public ::testing::WithParamInterface<CudaGrottoIntervalLutEvalTestParams>
{
public:
    using Type = T;
};

class CudaGrottoIntervalLutEvalTestU8 : public CudaGrottoIntervalLutEvalTestBase<std::uint8_t>
{
};
class CudaGrottoIntervalLutEvalTestU16 : public CudaGrottoIntervalLutEvalTestBase<std::uint16_t>
{
};
class CudaGrottoIntervalLutEvalTestU32 : public CudaGrottoIntervalLutEvalTestBase<std::uint32_t>
{
};
class CudaGrottoIntervalLutEvalTestU64 : public CudaGrottoIntervalLutEvalTestBase<std::uint64_t>
{
};
class CudaGrottoIntervalLutEvalTestU128 : public CudaGrottoIntervalLutEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCudaGrottoIntervalLutEvalTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 17};

    using T = typename Fixture::Type;

    CudaGrottoIntervalLutEvalTestParams params = self->GetParam();

    const std::size_t bitWidthIn    = params.bitWidthIn;
    const std::size_t bitWidthOut   = params.bitWidthOut;
    const std::size_t intervalCount = params.intervalCount;
    const std::size_t lutNum        = params.lutNum;
    const std::size_t elementSize   = sizeof(T);
    const T           maskOut       = bit_mask<T>(bitWidthOut);

    for (std::size_t elementNum : kElementNumConfigs)
    {
        GrottoIntervalLutEvalTestCase<T> testCase(bitWidthIn, bitWidthOut, elementNum, intervalCount, lutNum);

        std::size_t keySize   = FastFss::grotto::grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum);
        std::size_t cacheSize = FastFss::grotto::grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);

        thrust::device_vector<T>            alpha_d(testCase.alpha.size());
        thrust::device_vector<T>            maskedX_d(testCase.maskedX.size());
        thrust::device_vector<std::uint8_t> seed0_d(testCase.seed0.size());
        thrust::device_vector<std::uint8_t> seed1_d(testCase.seed1.size());
        thrust::device_vector<T>            left_d(testCase.left.size());
        thrust::device_vector<T>            right_d(testCase.right.size());
        thrust::device_vector<T>            table_d(testCase.table.size());
        thrust::copy(testCase.alpha.begin(), testCase.alpha.end(), alpha_d.begin());
        thrust::copy(testCase.maskedX.begin(), testCase.maskedX.end(), maskedX_d.begin());
        thrust::copy(testCase.seed0.begin(), testCase.seed0.end(), seed0_d.begin());
        thrust::copy(testCase.seed1.begin(), testCase.seed1.end(), seed1_d.begin());
        thrust::copy(testCase.left.begin(), testCase.left.end(), left_d.begin());
        thrust::copy(testCase.right.begin(), testCase.right.end(), right_d.begin());
        thrust::copy(testCase.table.begin(), testCase.table.end(), table_d.begin());

        thrust::device_vector<std::uint8_t> key_d(keySize);
        FastFss::grotto::cuda::grottoKeyGen<T>(
            std::span<std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(alpha_d.data()), alpha_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), bitWidthIn,
            nullptr);

        thrust::device_vector<T>            shareE0_d(elementNum), shareE1_d(elementNum);
        thrust::device_vector<T>            shareT0_d(elementNum * lutNum), shareT1_d(elementNum * lutNum);
        thrust::device_vector<std::uint8_t> cache_d(cacheSize);

        FastFss::grotto::cuda::grottoIntervalLutEval<T>(
            std::span<T>(thrust::raw_pointer_cast(shareE0_d.data()), shareE0_d.size()),
            std::span<T>(thrust::raw_pointer_cast(shareT0_d.data()), shareT0_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), 0,
            std::span<const T>(thrust::raw_pointer_cast(left_d.data()), left_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(right_d.data()), right_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(table_d.data()), table_d.size()), bitWidthIn, bitWidthOut,
            std::span<std::uint8_t>(thrust::raw_pointer_cast(cache_d.data()), cache_d.size()), nullptr);
        FastFss::grotto::cuda::grottoIntervalLutEval<T>(
            std::span<T>(thrust::raw_pointer_cast(shareE1_d.data()), shareE1_d.size()),
            std::span<T>(thrust::raw_pointer_cast(shareT1_d.data()), shareT1_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), 1,
            std::span<const T>(thrust::raw_pointer_cast(left_d.data()), left_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(right_d.data()), right_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(table_d.data()), table_d.size()), bitWidthIn, bitWidthOut,
            std::span<std::uint8_t>{}, nullptr);

        std::vector<T> shareE0(elementNum), shareE1(elementNum);
        std::vector<T> shareT0(elementNum * lutNum), shareT1(elementNum * lutNum);
        thrust::copy(shareE0_d.begin(), shareE0_d.end(), shareE0.begin());
        thrust::copy(shareE1_d.begin(), shareE1_d.end(), shareE1.begin());
        thrust::copy(shareT0_d.begin(), shareT0_d.end(), shareT0.begin());
        thrust::copy(shareT1_d.begin(), shareT1_d.end(), shareT1.begin());

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            for (std::size_t j = 0; j < lutNum; ++j)
            {
                T e   = shareE0[i] + shareE1[i];
                T t   = (shareT0[i * lutNum + j] + shareT1[i * lutNum + j]) & maskOut;
                T out = (e * t) & maskOut;
                ASSERT_EQ(out, testCase.expectedT[i * lutNum + j]);
            }
        }
    }
}

TEST_P(CudaGrottoIntervalLutEvalTestU8, RunTestBody)
{
    RunCudaGrottoIntervalLutEvalTestBody(this);
}
TEST_P(CudaGrottoIntervalLutEvalTestU16, RunTestBody)
{
    RunCudaGrottoIntervalLutEvalTestBody(this);
}
TEST_P(CudaGrottoIntervalLutEvalTestU32, RunTestBody)
{
    RunCudaGrottoIntervalLutEvalTestBody(this);
}
TEST_P(CudaGrottoIntervalLutEvalTestU64, RunTestBody)
{
    RunCudaGrottoIntervalLutEvalTestBody(this);
}
TEST_P(CudaGrottoIntervalLutEvalTestU128, RunTestBody)
{
    RunCudaGrottoIntervalLutEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoIntervalLutEvalTestU8,
                         ::testing::Values(CudaGrottoIntervalLutEvalTestParams{6, 8, 3, 2},
                                           CudaGrottoIntervalLutEvalTestParams{7, 8, 3, 2},
                                           CudaGrottoIntervalLutEvalTestParams{8, 8, 3, 2}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoIntervalLutEvalTestU16,
                         ::testing::Values(CudaGrottoIntervalLutEvalTestParams{6, 16, 3, 2},
                                           CudaGrottoIntervalLutEvalTestParams{15, 16, 3, 2},
                                           CudaGrottoIntervalLutEvalTestParams{16, 16, 3, 2}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoIntervalLutEvalTestU32,
                         ::testing::Values(CudaGrottoIntervalLutEvalTestParams{6, 32, 3, 2},
                                           CudaGrottoIntervalLutEvalTestParams{31, 32, 3, 2},
                                           CudaGrottoIntervalLutEvalTestParams{32, 32, 3, 2}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoIntervalLutEvalTestU64,
                         ::testing::Values(CudaGrottoIntervalLutEvalTestParams{6, 64, 3, 2},
                                           CudaGrottoIntervalLutEvalTestParams{63, 64, 3, 2},
                                           CudaGrottoIntervalLutEvalTestParams{64, 64, 3, 2}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoIntervalLutEvalTestU128,
                         ::testing::Values(CudaGrottoIntervalLutEvalTestParams{6, 128, 3, 2},
                                           CudaGrottoIntervalLutEvalTestParams{127, 128, 3, 2},
                                           CudaGrottoIntervalLutEvalTestParams{128, 128, 3, 2}));

} // namespace FastFss::tests::grotto
