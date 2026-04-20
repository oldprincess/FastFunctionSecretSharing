#include <FastFssPP/ottt.h>
#include <FastFssPP/prng.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "ottt_lut_eval_case.h"

namespace FastFss::tests::ottt {

struct CudaOtttEvalTestParams
{
    std::size_t bitWidthIn;
};

template <typename T>
class CudaOtttLutEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CudaOtttEvalTestParams>
{
public:
    using Type = T;
};

class CudaOtttLutEvalTestU8 : public CudaOtttLutEvalTestBase<std::uint8_t>
{
};
class CudaOtttLutEvalTestU16 : public CudaOtttLutEvalTestBase<std::uint16_t>
{
};
class CudaOtttLutEvalTestU32 : public CudaOtttLutEvalTestBase<std::uint32_t>
{
};
class CudaOtttLutEvalTestU64 : public CudaOtttLutEvalTestBase<std::uint64_t>
{
};
class CudaOtttLutEvalTestU128 : public CudaOtttLutEvalTestBase<wideint::uint128_t>
{
};

template <typename Fixture>
void RunCudaOtttLutEvalTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 1023, 1024, 2047, 2048};
    static std::vector<std::size_t> kLutsNumConfigs    = {1, 2, 3, 4};

    using T = typename Fixture::Type;

    std::size_t bitWidthIn = self->GetParam().bitWidthIn;

    FastFss::prng::cpu::Prng prng;

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t lutsNum : kLutsNumConfigs)
        {
            OtttLutEvalTestCase<T> testCase(bitWidthIn, lutsNum, elementNum);

            std::size_t keySize = FastFss::ottt::otttGetKeyDataSize(bitWidthIn, elementNum);

            std::vector<std::uint8_t> key0(keySize);
            std::vector<std::uint8_t> key1(keySize);
            prng.generate(key0.data(), 8, key0.size());
            key1.assign(key0.begin(), key0.end());

            thrust::device_vector<std::uint8_t> dKey0(key0.begin(), key0.end());
            thrust::device_vector<std::uint8_t> dKey1(key1.begin(), key1.end());
            thrust::device_vector<T>            dAlpha(testCase.alpha.begin(), testCase.alpha.end());
            thrust::device_vector<T>            dMaskedX(testCase.maskedX.begin(), testCase.maskedX.end());
            thrust::device_vector<T>            dLut(testCase.lut.begin(), testCase.lut.end());
            thrust::device_vector<T>            dE0(elementNum);
            thrust::device_vector<T>            dE1(elementNum);
            thrust::device_vector<T>            dT0(elementNum * lutsNum);
            thrust::device_vector<T>            dT1(elementNum * lutsNum);

            FastFss::ottt::cuda::otttKeyGen<T>(
                std::span<std::uint8_t>(thrust::raw_pointer_cast(dKey1.data()), dKey1.size()),
                std::span<const T>(thrust::raw_pointer_cast(dAlpha.data()), dAlpha.size()), bitWidthIn, nullptr);

            FastFss::ottt::cuda::otttLutEval<T>(
                std::span<T>(thrust::raw_pointer_cast(dE0.data()), dE0.size()),
                std::span<T>(thrust::raw_pointer_cast(dT0.data()), dT0.size()),
                std::span<const T>(thrust::raw_pointer_cast(dMaskedX.data()), dMaskedX.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(dKey0.data()), dKey0.size()), 0,
                std::span<const T>(thrust::raw_pointer_cast(dLut.data()), dLut.size()), bitWidthIn, nullptr);
            FastFss::ottt::cuda::otttLutEval<T>(
                std::span<T>(thrust::raw_pointer_cast(dE1.data()), dE1.size()),
                std::span<T>(thrust::raw_pointer_cast(dT1.data()), dT1.size()),
                std::span<const T>(thrust::raw_pointer_cast(dMaskedX.data()), dMaskedX.size()),
                std::span<const std::uint8_t>(thrust::raw_pointer_cast(dKey1.data()), dKey1.size()), 1,
                std::span<const T>(thrust::raw_pointer_cast(dLut.data()), dLut.size()), bitWidthIn, nullptr);

            std::vector<T> shareE0(elementNum), shareE1(elementNum);
            std::vector<T> shareT0(elementNum * lutsNum), shareT1(elementNum * lutsNum);
            thrust::copy(dE0.begin(), dE0.end(), shareE0.begin());
            thrust::copy(dE1.begin(), dE1.end(), shareE1.begin());
            thrust::copy(dT0.begin(), dT0.end(), shareT0.begin());
            thrust::copy(dT1.begin(), dT1.end(), shareT1.begin());

            for (std::size_t i = 0; i < elementNum; ++i)
            {
                T e = shareE0[i] + shareE1[i];
                for (std::size_t j = 0; j < lutsNum; ++j)
                {
                    T t   = (shareT0[i * lutsNum + j] + shareT1[i * lutsNum + j]) & testCase.maskOut;
                    T out = (e * t) & testCase.maskOut;
                    ASSERT_EQ(out, testCase.expected[i * lutsNum + j]);
                }
            }
        }
    }
}

TEST_P(CudaOtttLutEvalTestU8, RunTestBody)
{
    RunCudaOtttLutEvalTestBody(this);
}
TEST_P(CudaOtttLutEvalTestU16, RunTestBody)
{
    RunCudaOtttLutEvalTestBody(this);
}
TEST_P(CudaOtttLutEvalTestU32, RunTestBody)
{
    RunCudaOtttLutEvalTestBody(this);
}
TEST_P(CudaOtttLutEvalTestU64, RunTestBody)
{
    RunCudaOtttLutEvalTestBody(this);
}
TEST_P(CudaOtttLutEvalTestU128, RunTestBody)
{
    RunCudaOtttLutEvalTestBody(this);
}

INSTANTIATE_TEST_SUITE_P(,
                         CudaOtttLutEvalTestU8,
                         ::testing::Values(CudaOtttEvalTestParams{3},
                                           CudaOtttEvalTestParams{4},
                                           CudaOtttEvalTestParams{5},
                                           CudaOtttEvalTestParams{6},
                                           CudaOtttEvalTestParams{7},
                                           CudaOtttEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaOtttLutEvalTestU16,
                         ::testing::Values(CudaOtttEvalTestParams{3},
                                           CudaOtttEvalTestParams{4},
                                           CudaOtttEvalTestParams{5},
                                           CudaOtttEvalTestParams{6},
                                           CudaOtttEvalTestParams{7},
                                           CudaOtttEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaOtttLutEvalTestU32,
                         ::testing::Values(CudaOtttEvalTestParams{3},
                                           CudaOtttEvalTestParams{4},
                                           CudaOtttEvalTestParams{5},
                                           CudaOtttEvalTestParams{6},
                                           CudaOtttEvalTestParams{7},
                                           CudaOtttEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaOtttLutEvalTestU64,
                         ::testing::Values(CudaOtttEvalTestParams{3},
                                           CudaOtttEvalTestParams{4},
                                           CudaOtttEvalTestParams{5},
                                           CudaOtttEvalTestParams{6},
                                           CudaOtttEvalTestParams{7},
                                           CudaOtttEvalTestParams{8}));
INSTANTIATE_TEST_SUITE_P(,
                         CudaOtttLutEvalTestU128,
                         ::testing::Values(CudaOtttEvalTestParams{3},
                                           CudaOtttEvalTestParams{4},
                                           CudaOtttEvalTestParams{5},
                                           CudaOtttEvalTestParams{6},
                                           CudaOtttEvalTestParams{7},
                                           CudaOtttEvalTestParams{8}));

} // namespace FastFss::tests::ottt
