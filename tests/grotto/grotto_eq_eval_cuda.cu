#include <FastFssPP/grotto.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "grotto_eq_eval_test_case.h"

namespace FastFss::tests::grotto {

struct CudaGrottoEqEvalTestParams
{
    std::size_t bitWidthIn;
};

template <typename T>
class CudaGrottoEqEvalTestBase : public ::testing::Test,
                                 public ::testing::WithParamInterface<CudaGrottoEqEvalTestParams>
{
public:
    using Type = T;
};

class CudaGrottoEqEvalTestU8 : public CudaGrottoEqEvalTestBase<std::uint8_t>
{
};
class CudaGrottoEqEvalTestU16 : public CudaGrottoEqEvalTestBase<std::uint16_t>
{
};
class CudaGrottoEqEvalTestU32 : public CudaGrottoEqEvalTestBase<std::uint32_t>
{
};
class CudaGrottoEqEvalTestU64 : public CudaGrottoEqEvalTestBase<std::uint64_t>
{
};
class CudaGrottoEqEvalTestU128 : public CudaGrottoEqEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCudaGrottoEqEvalTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};

    using T = typename Fixture::Type;

    CudaGrottoEqEvalTestParams params = self->GetParam();

    const std::size_t bitWidthIn  = params.bitWidthIn;
    const std::size_t elementSize = sizeof(T);

    for (std::size_t elementNum : kElementNumConfigs)
    {
        GrottoEqEvalTestCase<T> testCase(bitWidthIn, elementNum);

        std::size_t keySize   = FastFss::grotto::grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum);
        std::size_t cacheSize = FastFss::grotto::grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);

        thrust::device_vector<T>            alpha_d(testCase.alpha.size());
        thrust::device_vector<T>            maskedX_d(testCase.maskedX.size());
        thrust::device_vector<std::uint8_t> seed0_d(testCase.seed0.size());
        thrust::device_vector<std::uint8_t> seed1_d(testCase.seed1.size());
        thrust::copy(testCase.alpha.begin(), testCase.alpha.end(), alpha_d.begin());
        thrust::copy(testCase.maskedX.begin(), testCase.maskedX.end(), maskedX_d.begin());
        thrust::copy(testCase.seed0.begin(), testCase.seed0.end(), seed0_d.begin());
        thrust::copy(testCase.seed1.begin(), testCase.seed1.end(), seed1_d.begin());

        thrust::device_vector<std::uint8_t> key_d(keySize);
        FastFss::grotto::cuda::grottoKeyGen<T>(
            std::span<std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(alpha_d.data()), alpha_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), bitWidthIn,
            nullptr);

        thrust::device_vector<T>            eq0_d(elementNum), eq1_d(elementNum);
        thrust::device_vector<std::uint8_t> cache_d(cacheSize);

        FastFss::grotto::cuda::grottoEqEval<T>(
            std::span<T>(thrust::raw_pointer_cast(eq0_d.data()), eq0_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), 0, bitWidthIn,
            std::span<std::uint8_t>(thrust::raw_pointer_cast(cache_d.data()), cache_d.size()), nullptr);
        FastFss::grotto::cuda::grottoEqEval<T>(
            std::span<T>(thrust::raw_pointer_cast(eq1_d.data()), eq1_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), 1, bitWidthIn,
            std::span<std::uint8_t>{}, nullptr);

        std::vector<T> eq0(elementNum), eq1(elementNum);
        thrust::copy(eq0_d.begin(), eq0_d.end(), eq0.begin());
        thrust::copy(eq1_d.begin(), eq1_d.end(), eq1.begin());

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            const T eqOut = (eq0[i] + eq1[i]) & T(1);
            ASSERT_EQ(eqOut, testCase.expected[i]);
        }
    }
}

TEST_P(CudaGrottoEqEvalTestU8, RunTestBody)
{
    RunCudaGrottoEqEvalTestBody(this);
}
TEST_P(CudaGrottoEqEvalTestU16, RunTestBody)
{
    RunCudaGrottoEqEvalTestBody(this);
}
TEST_P(CudaGrottoEqEvalTestU32, RunTestBody)
{
    RunCudaGrottoEqEvalTestBody(this);
}
TEST_P(CudaGrottoEqEvalTestU64, RunTestBody)
{
    RunCudaGrottoEqEvalTestBody(this);
}
TEST_P(CudaGrottoEqEvalTestU128, RunTestBody)
{
    RunCudaGrottoEqEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoEqEvalTestU8,
                         ::testing::Values(CudaGrottoEqEvalTestParams{6},
                                           CudaGrottoEqEvalTestParams{7},
                                           CudaGrottoEqEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoEqEvalTestU16,
                         ::testing::Values(CudaGrottoEqEvalTestParams{6},
                                           CudaGrottoEqEvalTestParams{15},
                                           CudaGrottoEqEvalTestParams{16}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoEqEvalTestU32,
                         ::testing::Values(CudaGrottoEqEvalTestParams{6},
                                           CudaGrottoEqEvalTestParams{31},
                                           CudaGrottoEqEvalTestParams{32}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoEqEvalTestU64,
                         ::testing::Values(CudaGrottoEqEvalTestParams{6},
                                           CudaGrottoEqEvalTestParams{63},
                                           CudaGrottoEqEvalTestParams{64}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoEqEvalTestU128,
                         ::testing::Values(CudaGrottoEqEvalTestParams{6},
                                           CudaGrottoEqEvalTestParams{127},
                                           CudaGrottoEqEvalTestParams{128}));

} // namespace FastFss::tests::grotto
