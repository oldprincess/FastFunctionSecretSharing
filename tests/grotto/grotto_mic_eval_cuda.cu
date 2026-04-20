#include <FastFssPP/grotto.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "grotto_mic_eval_test_case.h"

namespace FastFss::tests::grotto {

struct CudaGrottoMicEvalTestParams
{
    std::size_t bitWidthIn;
};

template <typename T>
class CudaGrottoMicEvalTestBase : public ::testing::Test,
                                  public ::testing::WithParamInterface<CudaGrottoMicEvalTestParams>
{
public:
    using Type = T;
};

class CudaGrottoMicEvalTestU8 : public CudaGrottoMicEvalTestBase<std::uint8_t>
{
};
class CudaGrottoMicEvalTestU16 : public CudaGrottoMicEvalTestBase<std::uint16_t>
{
};
class CudaGrottoMicEvalTestU32 : public CudaGrottoMicEvalTestBase<std::uint32_t>
{
};
class CudaGrottoMicEvalTestU64 : public CudaGrottoMicEvalTestBase<std::uint64_t>
{
};
class CudaGrottoMicEvalTestU128 : public CudaGrottoMicEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCudaGrottoMicEvalTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs    = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kIntervalCountConfigs = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    CudaGrottoMicEvalTestParams params = self->GetParam();

    const std::size_t bitWidthIn  = params.bitWidthIn;
    const std::size_t elementSize = sizeof(T);

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t intervalCount : kIntervalCountConfigs)
        {
            GrottoMicEvalTestCase<T> testCase(bitWidthIn, elementNum, intervalCount);

            std::size_t keySize   = FastFss::grotto::grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum);
            std::size_t cacheSize = FastFss::grotto::grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);

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
            FastFss::grotto::cuda::grottoKeyGen<T>(
                std::span<std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(alpha_d.data()), alpha_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), bitWidthIn,
                nullptr);

            thrust::device_vector<T>            share0_d(elementNum * intervalCount);
            thrust::device_vector<T>            share1_d(elementNum * intervalCount);
            thrust::device_vector<std::uint8_t> cache_d(cacheSize);

            FastFss::grotto::cuda::grottoMICEval<T>(
                std::span<T>(thrust::raw_pointer_cast(share0_d.data()), share0_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), 0,
                std::span<const T>(thrust::raw_pointer_cast(left_d.data()), left_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(right_d.data()), right_d.size()), bitWidthIn,
                std::span<std::uint8_t>(thrust::raw_pointer_cast(cache_d.data()), cache_d.size()), nullptr);
            FastFss::grotto::cuda::grottoMICEval<T>(
                std::span<T>(thrust::raw_pointer_cast(share1_d.data()), share1_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), 1,
                std::span<const T>(thrust::raw_pointer_cast(left_d.data()), left_d.size()),
                std::span<const T>(thrust::raw_pointer_cast(right_d.data()), right_d.size()), bitWidthIn,
                std::span<std::uint8_t>{}, nullptr);

            std::vector<T> share0(elementNum * intervalCount), share1(elementNum * intervalCount);
            thrust::copy(share0_d.begin(), share0_d.end(), share0.begin());
            thrust::copy(share1_d.begin(), share1_d.end(), share1.begin());

            for (std::size_t i = 0; i < share0.size(); ++i)
            {
                const T out = (share0[i] + share1[i]) & T(1);
                ASSERT_EQ(out, testCase.expected[i]);
            }
        }
    }
}

TEST_P(CudaGrottoMicEvalTestU8, RunTestBody)
{
    RunCudaGrottoMicEvalTestBody(this);
}
TEST_P(CudaGrottoMicEvalTestU16, RunTestBody)
{
    RunCudaGrottoMicEvalTestBody(this);
}
TEST_P(CudaGrottoMicEvalTestU32, RunTestBody)
{
    RunCudaGrottoMicEvalTestBody(this);
}
TEST_P(CudaGrottoMicEvalTestU64, RunTestBody)
{
    RunCudaGrottoMicEvalTestBody(this);
}
TEST_P(CudaGrottoMicEvalTestU128, RunTestBody)
{
    RunCudaGrottoMicEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoMicEvalTestU8,
                         ::testing::Values(CudaGrottoMicEvalTestParams{6},
                                           CudaGrottoMicEvalTestParams{7},
                                           CudaGrottoMicEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoMicEvalTestU16,
                         ::testing::Values(CudaGrottoMicEvalTestParams{6},
                                           CudaGrottoMicEvalTestParams{15},
                                           CudaGrottoMicEvalTestParams{16}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoMicEvalTestU32,
                         ::testing::Values(CudaGrottoMicEvalTestParams{6},
                                           CudaGrottoMicEvalTestParams{31},
                                           CudaGrottoMicEvalTestParams{32}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoMicEvalTestU64,
                         ::testing::Values(CudaGrottoMicEvalTestParams{6},
                                           CudaGrottoMicEvalTestParams{63},
                                           CudaGrottoMicEvalTestParams{64}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoMicEvalTestU128,
                         ::testing::Values(CudaGrottoMicEvalTestParams{6},
                                           CudaGrottoMicEvalTestParams{127},
                                           CudaGrottoMicEvalTestParams{128}));

} // namespace FastFss::tests::grotto
