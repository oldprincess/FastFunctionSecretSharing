#include <FastFssPP/prng.h>
#include <FastFssPP/spline.h>
#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <sstream>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "spline_eval_test_case.h"

namespace FastFss::tests::spline {

struct CudaSplineEvalTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

template <typename T>
class CudaSplineEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CudaSplineEvalTestParams>
{
public:
    using Type = T;
};

class CudaSplineEvalTestU8 : public CudaSplineEvalTestBase<std::uint8_t> {};
class CudaSplineEvalTestU16 : public CudaSplineEvalTestBase<std::uint16_t> {};
class CudaSplineEvalTestU32 : public CudaSplineEvalTestBase<std::uint32_t> {};
class CudaSplineEvalTestU64 : public CudaSplineEvalTestBase<std::uint64_t> {};
class CudaSplineEvalTestU128 : public CudaSplineEvalTestBase<wideint::uint128_t> {};

template <typename Fixture>
void RunCudaSplineEvalTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs = {1, 8, 255, 2047};
    static std::vector<std::size_t> kIntervalConfigs   = {1, 2, 4, 7};
    static std::vector<std::size_t> kDegreeConfigs     = {0, 1, 2};

    using T = typename Fixture::Type;

    CudaSplineEvalTestParams params = self->GetParam();
    std::size_t bitWidthIn  = params.bitWidthIn;
    std::size_t bitWidthOut = params.bitWidthOut;
    std::size_t elementSize = sizeof(T);
    std::size_t valueBits   = sizeof(T) * 8;
    T           maskOut     = (bitWidthOut >= valueBits) ? ~T(0) : ((T(1) << bitWidthOut) - T(1));

    for (std::size_t elementNum : kElementNumConfigs)
    {
        for (std::size_t intervalCount : kIntervalConfigs)
        {
            if (bitWidthIn < 16 && intervalCount > (std::size_t(1) << bitWidthIn))
            {
                continue;
            }
            for (std::size_t degree : kDegreeConfigs)
            {
                SplineEvalTestCase<T> testCase(bitWidthIn, bitWidthOut, degree, elementNum, intervalCount);
                std::size_t coeffNum  = degree + 1;
                std::size_t groupSize = intervalCount * coeffNum;

                std::size_t keySize = FastFss::spline::dcfSplineGetKeyDataSize(degree, intervalCount, bitWidthIn,
                                                                               bitWidthOut, elementSize, elementNum);
                std::size_t cacheSize = FastFss::spline::dcfSplineGetCacheDataSize(
                    degree, intervalCount, bitWidthIn, bitWidthOut, elementSize, elementNum);

                thrust::device_vector<T>            alpha_d(testCase.alpha.begin(), testCase.alpha.end());
                thrust::device_vector<T>            maskedX_d(testCase.maskedX.begin(), testCase.maskedX.end());
                thrust::device_vector<std::uint8_t> seed0_d(testCase.seed0.begin(), testCase.seed0.end());
                thrust::device_vector<std::uint8_t> seed1_d(testCase.seed1.begin(), testCase.seed1.end());
                thrust::device_vector<T>            left_d(testCase.leftEndpoints.begin(), testCase.leftEndpoints.end());
                thrust::device_vector<T>            right_d(testCase.rightEndpoints.begin(), testCase.rightEndpoints.end());
                thrust::device_vector<T>            coefficients_d(testCase.coefficients.begin(), testCase.coefficients.end());

                thrust::device_vector<std::uint8_t> key_d(keySize);
                thrust::device_vector<T>            e_d(elementNum * groupSize);
                thrust::device_vector<T>            beta_d(elementNum * groupSize);
                std::vector<std::uint8_t>          key_cpu(keySize);
                std::vector<T>                     e_cpu(elementNum * groupSize);
                std::vector<T>                     beta_cpu(elementNum * groupSize);

                FastFss::spline::cuda::dcfSplineKeyGen<T>(
                    std::span<std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                    std::span<T>(thrust::raw_pointer_cast(e_d.data()), e_d.size()),
                    std::span<T>(thrust::raw_pointer_cast(beta_d.data()), beta_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(alpha_d.data()), alpha_d.size()),
                    std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()),
                    std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(coefficients_d.data()), coefficients_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(left_d.data()), left_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(right_d.data()), right_d.size()), degree, bitWidthIn,
                    bitWidthOut, nullptr);
                FastFss::spline::cpu::dcfSplineKeyGen<T>(
                    std::span<std::uint8_t>(key_cpu.data(), key_cpu.size()), std::span<T>(e_cpu.data(), e_cpu.size()),
                    std::span<T>(beta_cpu.data(), beta_cpu.size()),
                    std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
                    std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
                    std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()),
                    std::span<const T>(testCase.coefficients.data(), testCase.coefficients.size()),
                    std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                    std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), degree, bitWidthIn,
                    bitWidthOut);

                std::vector<std::uint8_t> key(keySize);
                std::vector<T> e(elementNum * groupSize);
                std::vector<T> beta(elementNum * groupSize);
                thrust::copy(key_d.begin(), key_d.end(), key.begin());
                thrust::copy(e_d.begin(), e_d.end(), e.begin());
                thrust::copy(beta_d.begin(), beta_d.end(), beta.begin());

                ASSERT_EQ(key, key_cpu);
                ASSERT_EQ(e, e_cpu);
                ASSERT_EQ(beta, beta_cpu);

                std::vector<T> sharedE0(e.size());
                std::vector<T> sharedE1(e.size());
                std::vector<T> sharedBeta0(beta.size());
                std::vector<T> sharedBeta1(beta.size());
                {
                    FastFss::prng::cpu::Prng prng;
                    ASSERT_NO_THROW(prng.generate(sharedE0.data(), bitWidthOut, sharedE0.size()));
                    ASSERT_NO_THROW(prng.generate(sharedBeta0.data(), bitWidthOut, sharedBeta0.size()));
                    for (std::size_t i = 0; i < e.size(); ++i)
                    {
                        sharedE0[i] &= maskOut;
                        sharedBeta0[i] &= maskOut;
                        sharedE1[i] = (e[i] - sharedE0[i]) & maskOut;
                        sharedBeta1[i] = (beta[i] - sharedBeta0[i]) & maskOut;
                    }
                }

                thrust::device_vector<T> sharedE0_d(sharedE0.begin(), sharedE0.end());
                thrust::device_vector<T> sharedE1_d(sharedE1.begin(), sharedE1.end());
                thrust::device_vector<T> sharedBeta0_d(sharedBeta0.begin(), sharedBeta0.end());
                thrust::device_vector<T> sharedBeta1_d(sharedBeta1.begin(), sharedBeta1.end());
                thrust::device_vector<T> share0_d(elementNum);
                thrust::device_vector<T> share1_d(elementNum);
                thrust::device_vector<std::uint8_t> cache0_d(cacheSize);
                thrust::device_vector<std::uint8_t> cache1_d(cacheSize);
                std::vector<T> share0_cpu(elementNum);
                std::vector<T> share1_cpu(elementNum);
                std::vector<std::uint8_t> cache0_cpu(cacheSize);
                std::vector<std::uint8_t> cache1_cpu(cacheSize);

                FastFss::spline::cuda::dcfSplineEval<T>(
                    std::span<T>(thrust::raw_pointer_cast(share0_d.data()), share0_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                    std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(sharedE0_d.data()), sharedE0_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(sharedBeta0_d.data()), sharedBeta0_d.size()),
                    std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed0_d.data()), seed0_d.size()), 0,
                    std::span<const T>(thrust::raw_pointer_cast(left_d.data()), left_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(right_d.data()), right_d.size()), degree, bitWidthIn,
                    bitWidthOut, std::span<std::uint8_t>(thrust::raw_pointer_cast(cache0_d.data()), cache0_d.size()),
                    nullptr);
                FastFss::spline::cuda::dcfSplineEval<T>(
                    std::span<T>(thrust::raw_pointer_cast(share1_d.data()), share1_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(maskedX_d.data()), maskedX_d.size()),
                    std::span<const std::uint8_t>(thrust::raw_pointer_cast(key_d.data()), key_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(sharedE1_d.data()), sharedE1_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(sharedBeta1_d.data()), sharedBeta1_d.size()),
                    std::span<const std::uint8_t>(thrust::raw_pointer_cast(seed1_d.data()), seed1_d.size()), 1,
                    std::span<const T>(thrust::raw_pointer_cast(left_d.data()), left_d.size()),
                    std::span<const T>(thrust::raw_pointer_cast(right_d.data()), right_d.size()), degree, bitWidthIn,
                    bitWidthOut, std::span<std::uint8_t>(thrust::raw_pointer_cast(cache1_d.data()), cache1_d.size()),
                    nullptr);
                FastFss::spline::cpu::dcfSplineEval<T>(
                    std::span<T>(share0_cpu.data(), share0_cpu.size()),
                    std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                    std::span<const std::uint8_t>(key_cpu.data(), key_cpu.size()),
                    std::span<const T>(sharedE0.data(), sharedE0.size()),
                    std::span<const T>(sharedBeta0.data(), sharedBeta0.size()),
                    std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), 0,
                    std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                    std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), degree, bitWidthIn,
                    bitWidthOut, std::span<std::uint8_t>(cache0_cpu.data(), cache0_cpu.size()));
                FastFss::spline::cpu::dcfSplineEval<T>(
                    std::span<T>(share1_cpu.data(), share1_cpu.size()),
                    std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                    std::span<const std::uint8_t>(key_cpu.data(), key_cpu.size()),
                    std::span<const T>(sharedE1.data(), sharedE1.size()),
                    std::span<const T>(sharedBeta1.data(), sharedBeta1.size()),
                    std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), 1,
                    std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                    std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), degree, bitWidthIn,
                    bitWidthOut, std::span<std::uint8_t>(cache1_cpu.data(), cache1_cpu.size()));

                std::vector<T> share0(elementNum);
                std::vector<T> share1(elementNum);
                thrust::copy(share0_d.begin(), share0_d.end(), share0.begin());
                thrust::copy(share1_d.begin(), share1_d.end(), share1.begin());

                ASSERT_EQ(share0, share0_cpu);
                ASSERT_EQ(share1, share1_cpu);

                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    T out = (share0[i] + share1[i]) & maskOut;
                    T cpuOut = (share0_cpu[i] + share1_cpu[i]) & maskOut;
                    if (out != testCase.expected[i] || cpuOut != testCase.expected[i])
                    {
                        std::ostringstream oss;
                        oss << "elementNum=" << elementNum << " intervalCount=" << intervalCount
                            << " degree=" << degree << " index=" << i << " x="
                            << static_cast<unsigned long long>(testCase.x[i]) << " alpha="
                            << static_cast<unsigned long long>(testCase.alpha[i]) << " maskedX="
                            << static_cast<unsigned long long>(testCase.maskedX[i]) << " cudaOut="
                            << static_cast<unsigned long long>(out) << " cpuOut="
                            << static_cast<unsigned long long>(cpuOut) << " expected="
                            << static_cast<unsigned long long>(testCase.expected[i]);
                        ADD_FAILURE() << oss.str();
                        return;
                    }
                }
            }
        }
    }
}

TEST_P(CudaSplineEvalTestU8, RunTestBody) { RunCudaSplineEvalTestBody(this); }
TEST_P(CudaSplineEvalTestU16, RunTestBody) { RunCudaSplineEvalTestBody(this); }
TEST_P(CudaSplineEvalTestU32, RunTestBody) { RunCudaSplineEvalTestBody(this); }
TEST_P(CudaSplineEvalTestU64, RunTestBody) { RunCudaSplineEvalTestBody(this); }
TEST_P(CudaSplineEvalTestU128, RunTestBody) { RunCudaSplineEvalTestBody(this); }

INSTANTIATE_TEST_SUITE_P(, CudaSplineEvalTestU8, ::testing::Values(CudaSplineEvalTestParams{8, 8}));
INSTANTIATE_TEST_SUITE_P(, CudaSplineEvalTestU16, ::testing::Values(CudaSplineEvalTestParams{16, 16}));
INSTANTIATE_TEST_SUITE_P(, CudaSplineEvalTestU32, ::testing::Values(CudaSplineEvalTestParams{32, 32}));
INSTANTIATE_TEST_SUITE_P(, CudaSplineEvalTestU64, ::testing::Values(CudaSplineEvalTestParams{64, 64}));
INSTANTIATE_TEST_SUITE_P(, CudaSplineEvalTestU128, ::testing::Values(CudaSplineEvalTestParams{128, 128}));

} // namespace FastFss::tests::spline
