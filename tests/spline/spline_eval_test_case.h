#ifndef FAST_FSS_TESTS_SPLINE_EVAL_TEST_CASE_H
#define FAST_FSS_TESTS_SPLINE_EVAL_TEST_CASE_H

#include <FastFssPP/prng.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace FastFss::tests::spline {

template <typename T>
inline T maskToBits(T value, std::size_t bitWidth)
{
    if (bitWidth >= sizeof(T) * 8)
    {
        return value;
    }
    return value & ((T(1) << bitWidth) - T(1));
}

template <typename T>
inline T evalPolynomial(const T *coefficients, std::size_t degree, T x, std::size_t bitWidth)
{
    T result = 0;
    T power  = 1;
    x        = maskToBits(x, bitWidth);
    for (std::size_t i = 0; i <= degree; ++i)
    {
        result = maskToBits<T>(result + maskToBits<T>(coefficients[i] * power, bitWidth), bitWidth);
        power  = maskToBits<T>(power * x, bitWidth);
    }
    return result;
}

template <typename T>
inline void normalizeStrictlyIncreasingRightEndpoints(std::vector<T> &rightEndpoints, T maskIn)
{
    if (rightEndpoints.empty())
    {
        return;
    }

    const std::size_t intervalCount = rightEndpoints.size();
    for (std::size_t i = 0; i + 1 < intervalCount; ++i)
    {
        T minBoundary = (i == 0) ? T(0) : static_cast<T>(rightEndpoints[i - 1] + T(1));
        T maxBoundary = static_cast<T>(maskIn - T(intervalCount - 1 - i));
        T span        = static_cast<T>(maxBoundary - minBoundary);
        rightEndpoints[i] = static_cast<T>(minBoundary + (rightEndpoints[i] % (span + T(1))));
    }
    rightEndpoints.back() = maskIn;
}

template <typename T>
struct SplineEvalTestCase
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
    std::size_t degree;
    std::size_t elementNum;
    std::size_t intervalCount;

    std::vector<T>            x;
    std::vector<std::uint8_t> seed0;
    std::vector<std::uint8_t> seed1;
    std::vector<T>            alpha;
    std::vector<T>            maskedX;
    std::vector<T>            leftEndpoints;
    std::vector<T>            rightEndpoints;
    std::vector<T>            coefficients;
    std::vector<T>            expected;

    SplineEvalTestCase(std::size_t bitWidthIn_,
                       std::size_t bitWidthOut_,
                       std::size_t degree_,
                       std::size_t elementNum_,
                       std::size_t intervalCount_)
        : bitWidthIn(bitWidthIn_),
          bitWidthOut(bitWidthOut_),
          degree(degree_),
          elementNum(elementNum_),
          intervalCount(intervalCount_),
          expected(elementNum_, T(0))
    {
        std::size_t valueBits   = sizeof(T) * 8;
        std::size_t elementSize = sizeof(T);
        T           maskIn      = (bitWidthIn >= valueBits) ? ~T(0) : ((T(1) << bitWidthIn) - T(1));
        T           maskOut     = (bitWidthOut >= valueBits) ? ~T(0) : ((T(1) << bitWidthOut) - T(1));
        std::size_t coeffNum    = degree + 1;

        FastFss::prng::cpu::Prng prng;

        seed0.resize(elementNum * 16);
        seed1.resize(elementNum * 16);
        alpha.resize(elementNum);
        x.resize(elementNum);
        maskedX.resize(elementNum);
        leftEndpoints.resize(intervalCount);
        rightEndpoints.resize(intervalCount);
        coefficients.resize(intervalCount * coeffNum);

        prng.generate(seed0.data(), 8, seed0.size());
        prng.generate(seed1.data(), 8, seed1.size());
        prng.generate(alpha.data(), bitWidthIn, elementSize, elementNum);
        prng.generate(x.data(), bitWidthIn, elementSize, elementNum);
        prng.generate(rightEndpoints.data(), bitWidthIn, elementSize, intervalCount);
        prng.generate(coefficients.data(), bitWidthOut, elementSize, coefficients.size());

        for (T &value : alpha)
        {
            value &= maskIn;
        }
        for (T &value : x)
        {
            value &= maskIn;
        }
        for (T &value : rightEndpoints)
        {
            value &= maskIn;
        }
        for (T &value : coefficients)
        {
            value &= maskOut;
        }

        std::sort(rightEndpoints.begin(), rightEndpoints.end());
        normalizeStrictlyIncreasingRightEndpoints(rightEndpoints, maskIn);
        if (intervalCount > 0)
        {
            leftEndpoints[0] = 0;
            for (std::size_t i = 1; i < intervalCount; ++i)
            {
                leftEndpoints[i] = static_cast<T>(rightEndpoints[i - 1] + T(1));
            }
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            maskedX[i] = (alpha[i] + x[i]) & maskIn;
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            std::size_t intervalIdx = 0;
            while (intervalIdx + 1 < intervalCount && rightEndpoints[intervalIdx] < x[i])
            {
                ++intervalIdx;
            }
            expected[i] = evalPolynomial(coefficients.data() + intervalIdx * coeffNum, degree, x[i], bitWidthOut);
        }
    }
};

} // namespace FastFss::tests::spline

#endif
