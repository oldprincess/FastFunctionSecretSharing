#ifndef FAST_FSS_TESTS_MIC_EVAL_TEST_CASE_H
#define FAST_FSS_TESTS_MIC_EVAL_TEST_CASE_H

#include <FastFssPP/prng.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../common/nbit_signed.hpp"

namespace FastFss::tests::mic {

template <typename T>
struct MicEvalTestCase
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
    std::size_t elementNum;
    std::size_t intervalCount;

    std::vector<T>            x;
    std::vector<std::uint8_t> seed0;
    std::vector<std::uint8_t> seed1;
    std::vector<T>            alpha;
    std::vector<T>            maskedX;
    std::vector<T>            leftEndpoints;
    std::vector<T>            rightEndpoints;
    std::vector<T>            expected;

    MicEvalTestCase(std::size_t bitWidthIn_,
                    std::size_t bitWidthOut_,
                    std::size_t elementNum_,
                    std::size_t intervalCount_)
        : bitWidthIn(bitWidthIn_),
          bitWidthOut(bitWidthOut_),
          elementNum(elementNum_),
          intervalCount(intervalCount_),
          expected(elementNum_ * intervalCount_, T(0))
    {
        std::size_t valueBits   = sizeof(T) * 8;
        std::size_t elementSize = sizeof(T);
        T           maskIn      = (bitWidthIn >= valueBits) ? ~T(0) : ((T(1) << bitWidthIn) - T(1));

        FastFss::prng::cpu::Prng prng;

        seed0.resize(elementNum * 16);
        seed1.resize(elementNum * 16);
        alpha.resize(elementNum);
        x.resize(elementNum);
        maskedX.resize(elementNum);
        leftEndpoints.resize(intervalCount);
        rightEndpoints.resize(intervalCount);

        prng.generate(seed0.data(), 8, elementNum * 16);
        prng.generate(seed1.data(), 8, elementNum * 16);
        prng.generate(alpha.data(), bitWidthIn, elementSize, elementNum);
        prng.generate(x.data(), bitWidthIn, elementSize, elementNum);
        prng.generate(leftEndpoints.data(), bitWidthIn, elementSize, intervalCount);
        prng.generate(rightEndpoints.data(), bitWidthIn, elementSize, intervalCount);

        for (T& value : alpha)
        {
            value = value & maskIn;
        }
        for (T& value : x)
        {
            value = value & maskIn;
        }
        for (T& value : leftEndpoints)
        {
            value = value & maskIn;
        }
        for (T& value : rightEndpoints)
        {
            value = value & maskIn;
        }

        for (std::size_t i = 0; i < intervalCount; ++i)
        {
            if (signedNBitLess(rightEndpoints[i], leftEndpoints[i], bitWidthIn))
            {
                std::swap(leftEndpoints[i], rightEndpoints[i]);
            }
        }

        for (std::size_t j = 0; j < elementNum; ++j)
        {
            maskedX[j] = (alpha[j] + x[j]) & maskIn;
        }

        for (std::size_t i = 0; i < intervalCount; ++i)
        {
            for (std::size_t j = 0; j < elementNum; ++j)
            {
                if (signedNBitIntervalContains(x[j], leftEndpoints[i], rightEndpoints[i], bitWidthIn))
                {
                    expected[j * intervalCount + i] = T(1);
                }
            }
        }
    }
};

} // namespace FastFss::tests::mic

#endif
