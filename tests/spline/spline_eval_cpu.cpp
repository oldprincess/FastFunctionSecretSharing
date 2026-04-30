#include <FastFssPP/prng.h>
#include <FastFssPP/spline.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <span>
#include <vector>
#include <wideint/wideint.hpp>

#include "spline_eval_test_case.h"

namespace FastFss::tests::spline {

struct CpuSplineEvalTestParams
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
};

template <typename T>
class CpuSplineEvalTestBase : public ::testing::Test, public ::testing::WithParamInterface<CpuSplineEvalTestParams>
{
public:
    using Type = T;
};

class CpuSplineEvalTestU8 : public CpuSplineEvalTestBase<std::uint8_t> {};
class CpuSplineEvalTestU16 : public CpuSplineEvalTestBase<std::uint16_t> {};
class CpuSplineEvalTestU32 : public CpuSplineEvalTestBase<std::uint32_t> {};
class CpuSplineEvalTestU64 : public CpuSplineEvalTestBase<std::uint64_t> {};
class CpuSplineEvalTestU128 : public CpuSplineEvalTestBase<wideint::uint128_t> {};

template <typename Fixture>
void RunCpuSplineEvalTestBody(Fixture *self)
{
    static std::vector<std::size_t> kElementNumConfigs  = {1, 8, 255, 2047};
    static std::vector<std::size_t> kIntervalConfigs    = {1, 2, 4, 7};
    static std::vector<std::size_t> kDegreeConfigs      = {0, 1, 2};

    using T = typename Fixture::Type;

    CpuSplineEvalTestParams params = self->GetParam();
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
                std::size_t coeffNum = degree + 1;
                std::size_t groupSize = intervalCount * coeffNum;

                std::size_t keySize = FastFss::spline::dcfSplineGetKeyDataSize(degree, intervalCount, bitWidthIn,
                                                                               bitWidthOut, elementSize, elementNum);
                std::size_t cacheSize = FastFss::spline::dcfSplineGetCacheDataSize(
                    degree, intervalCount, bitWidthIn, bitWidthOut, elementSize, elementNum);

                std::vector<std::uint8_t> key(keySize);
                std::vector<T> e(elementNum * groupSize);
                std::vector<T> beta(elementNum * groupSize);
                FastFss::spline::cpu::dcfSplineKeyGen<T>(
                    std::span<std::uint8_t>(key.data(), key.size()), std::span<T>(e.data(), e.size()),
                    std::span<T>(beta.data(), beta.size()), std::span<const T>(testCase.alpha.data(), testCase.alpha.size()),
                    std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()),
                    std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()),
                    std::span<const T>(testCase.coefficients.data(), testCase.coefficients.size()),
                    std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                    std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), degree, bitWidthIn,
                    bitWidthOut);

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

                std::vector<T> share0(elementNum);
                std::vector<T> share1(elementNum);
                std::vector<std::uint8_t> cache0(cacheSize);
                std::vector<std::uint8_t> cache1(cacheSize);
                FastFss::spline::cpu::dcfSplineEval<T>(
                    std::span<T>(share0.data(), share0.size()),
                    std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                    std::span<const std::uint8_t>(key.data(), key.size()),
                    std::span<const T>(sharedE0.data(), sharedE0.size()),
                    std::span<const T>(sharedBeta0.data(), sharedBeta0.size()),
                    std::span<const std::uint8_t>(testCase.seed0.data(), testCase.seed0.size()), 0,
                    std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                    std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), degree, bitWidthIn,
                    bitWidthOut, std::span<std::uint8_t>(cache0.data(), cache0.size()));
                FastFss::spline::cpu::dcfSplineEval<T>(
                    std::span<T>(share1.data(), share1.size()),
                    std::span<const T>(testCase.maskedX.data(), testCase.maskedX.size()),
                    std::span<const std::uint8_t>(key.data(), key.size()),
                    std::span<const T>(sharedE1.data(), sharedE1.size()),
                    std::span<const T>(sharedBeta1.data(), sharedBeta1.size()),
                    std::span<const std::uint8_t>(testCase.seed1.data(), testCase.seed1.size()), 1,
                    std::span<const T>(testCase.leftEndpoints.data(), testCase.leftEndpoints.size()),
                    std::span<const T>(testCase.rightEndpoints.data(), testCase.rightEndpoints.size()), degree, bitWidthIn,
                    bitWidthOut, std::span<std::uint8_t>(cache1.data(), cache1.size()));

                for (std::size_t i = 0; i < elementNum; ++i)
                {
                    T out = (share0[i] + share1[i]) & maskOut;
                    if (out != testCase.expected[i])
                    {
                        std::ostringstream oss;
                        oss << "elementNum=" << elementNum << " intervalCount=" << intervalCount
                            << " degree=" << degree << " index=" << i << " x="
                            << static_cast<unsigned long long>(testCase.x[i]) << " alpha="
                            << static_cast<unsigned long long>(testCase.alpha[i]) << " maskedX="
                            << static_cast<unsigned long long>(testCase.maskedX[i]) << " out="
                            << static_cast<unsigned long long>(out) << " expected="
                            << static_cast<unsigned long long>(testCase.expected[i]) << " rights=[";
                        for (std::size_t j = 0; j < testCase.rightEndpoints.size(); ++j)
                        {
                            if (j != 0)
                            {
                                oss << ",";
                            }
                            oss << static_cast<unsigned long long>(testCase.rightEndpoints[j]);
                        }
                        oss << "] coeffs=[";
                        for (std::size_t j = 0; j < testCase.coefficients.size(); ++j)
                        {
                            if (j != 0)
                            {
                                oss << ",";
                            }
                            oss << static_cast<unsigned long long>(testCase.coefficients[j]);
                        }
                        oss << "]";
                        ADD_FAILURE() << oss.str();
                        return;
                    }
                }
            }
        }
    }
}

TEST_P(CpuSplineEvalTestU8, RunTestBody) { RunCpuSplineEvalTestBody(this); }
TEST_P(CpuSplineEvalTestU16, RunTestBody) { RunCpuSplineEvalTestBody(this); }
TEST_P(CpuSplineEvalTestU32, RunTestBody) { RunCpuSplineEvalTestBody(this); }
TEST_P(CpuSplineEvalTestU64, RunTestBody) { RunCpuSplineEvalTestBody(this); }
TEST_P(CpuSplineEvalTestU128, RunTestBody) { RunCpuSplineEvalTestBody(this); }

INSTANTIATE_TEST_SUITE_P(, CpuSplineEvalTestU8, ::testing::Values(CpuSplineEvalTestParams{8, 8}));
INSTANTIATE_TEST_SUITE_P(, CpuSplineEvalTestU16, ::testing::Values(CpuSplineEvalTestParams{16, 16}));
INSTANTIATE_TEST_SUITE_P(, CpuSplineEvalTestU32, ::testing::Values(CpuSplineEvalTestParams{32, 32}));
INSTANTIATE_TEST_SUITE_P(, CpuSplineEvalTestU64, ::testing::Values(CpuSplineEvalTestParams{64, 64}));
INSTANTIATE_TEST_SUITE_P(, CpuSplineEvalTestU128, ::testing::Values(CpuSplineEvalTestParams{128, 128}));

} // namespace FastFss::tests::spline
