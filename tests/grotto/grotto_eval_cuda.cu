#include <FastFssPP/grotto.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "grotto_eval_test_case.h"

namespace FastFss::tests::grotto {

struct CudaGrottoEvalTestParams
{
    std::size_t bitWidthIn;
};

template <typename T>
class CudaGrottoEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CudaGrottoEvalTestParams>
{
public:
    using Type = T;
};

class CudaGrottoEvalTestU8 : public CudaGrottoEvalTestBase<std::uint8_t>
{
};
class CudaGrottoEvalTestU16 : public CudaGrottoEvalTestBase<std::uint16_t>
{
};
class CudaGrottoEvalTestU32 : public CudaGrottoEvalTestBase<std::uint32_t>
{
};
class CudaGrottoEvalTestU64 : public CudaGrottoEvalTestBase<std::uint64_t>
{
};
class CudaGrottoEvalTestU128 : public CudaGrottoEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCudaGrottoEvalTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};

    using T = typename Fixture::Type;

    CudaGrottoEvalTestParams params = self->GetParam();

    const std::size_t bitWidthIn  = params.bitWidthIn;
    const std::size_t elementSize = sizeof(T);

    for (std::size_t elementNum : kElementNumConfigs)
    {
        GrottoEvalTestCase<T> testCase(bitWidthIn, elementNum);

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

        thrust::device_vector<T>            lt0_d(elementNum), lt1_d(elementNum);
        thrust::device_vector<T>            le0_d(elementNum), le1_d(elementNum);
        thrust::device_vector<std::uint8_t> cache_d(cacheSize);

        FastFss::grotto::cuda::grottoEval<T>(
            std::span<T>(thrust::raw_pointer_cast(lt0_d.data()), lt0_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), false, 0,
            bitWidthIn, std::span<std::uint8_t>(thrust::raw_pointer_cast(cache_d.data()), cache_d.size()), nullptr);
        FastFss::grotto::cuda::grottoEval<T>(
            std::span<T>(thrust::raw_pointer_cast(lt1_d.data()), lt1_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), false, 1,
            bitWidthIn, std::span<std::uint8_t>{}, nullptr);
        FastFss::grotto::cuda::grottoEval<T>(
            std::span<T>(thrust::raw_pointer_cast(le0_d.data()), le0_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), true, 0,
            bitWidthIn, std::span<std::uint8_t>{}, nullptr);
        FastFss::grotto::cuda::grottoEval<T>(
            std::span<T>(thrust::raw_pointer_cast(le1_d.data()), le1_d.size()),
            std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
            std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), true, 1,
            bitWidthIn, std::span<std::uint8_t>{}, nullptr);

        std::vector<T> lt0(elementNum), lt1(elementNum), le0(elementNum), le1(elementNum);
        thrust::copy(lt0_d.begin(), lt0_d.end(), lt0.begin());
        thrust::copy(lt1_d.begin(), lt1_d.end(), lt1.begin());
        thrust::copy(le0_d.begin(), le0_d.end(), le0.begin());
        thrust::copy(le1_d.begin(), le1_d.end(), le1.begin());

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            const T ltOut = (lt0[i] + lt1[i]) & T(1);
            const T leOut = (le0[i] + le1[i]) & T(1);
            ASSERT_EQ(ltOut, testCase.expectedLt[i]);
            ASSERT_EQ(leOut, testCase.expectedLe[i]);
        }
    }
}

TEST_P(CudaGrottoEvalTestU8, RunTestBody)
{
    RunCudaGrottoEvalTestBody(this);
}
TEST_P(CudaGrottoEvalTestU16, RunTestBody)
{
    RunCudaGrottoEvalTestBody(this);
}
TEST_P(CudaGrottoEvalTestU32, RunTestBody)
{
    RunCudaGrottoEvalTestBody(this);
}
TEST_P(CudaGrottoEvalTestU64, RunTestBody)
{
    RunCudaGrottoEvalTestBody(this);
}
TEST_P(CudaGrottoEvalTestU128, RunTestBody)
{
    RunCudaGrottoEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoEvalTestU8,
                         ::testing::Values(CudaGrottoEvalTestParams{6},
                                           CudaGrottoEvalTestParams{7},
                                           CudaGrottoEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoEvalTestU16,
                         ::testing::Values(CudaGrottoEvalTestParams{6},
                                           CudaGrottoEvalTestParams{15},
                                           CudaGrottoEvalTestParams{16}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoEvalTestU32,
                         ::testing::Values(CudaGrottoEvalTestParams{6},
                                           CudaGrottoEvalTestParams{31},
                                           CudaGrottoEvalTestParams{32}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoEvalTestU64,
                         ::testing::Values(CudaGrottoEvalTestParams{6},
                                           CudaGrottoEvalTestParams{63},
                                           CudaGrottoEvalTestParams{64}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaGrottoEvalTestU128,
                         ::testing::Values(CudaGrottoEvalTestParams{6},
                                           CudaGrottoEvalTestParams{127},
                                           CudaGrottoEvalTestParams{128}));

} // namespace FastFss::tests::grotto
