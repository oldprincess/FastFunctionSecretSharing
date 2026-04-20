#ifndef FAST_FSS_TESTS_GROTTO_INTERVAL_LUT_EVAL_TEST_CASE_H
#define FAST_FSS_TESTS_GROTTO_INTERVAL_LUT_EVAL_TEST_CASE_H

#include <FastFssPP/prng.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../common/nbit_signed.hpp"

namespace FastFss::tests::grotto {

template <typename T>
struct GrottoIntervalLutEvalTestCase
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
    std::size_t elementNum;
    std::size_t intervalCount;
    std::size_t lutNum;

    std::vector<std::uint8_t> seed0;
    std::vector<std::uint8_t> seed1;
    std::vector<T>            alpha;
    std::vector<T>            x;
    std::vector<T>            maskedX;
    std::vector<T>            left;
    std::vector<T>            right;
    std::vector<T>            table;
    std::vector<T>            expectedE;
    std::vector<T>            expectedT;

    GrottoIntervalLutEvalTestCase(std::size_t bitWidthIn_,
                                  std::size_t bitWidthOut_,
                                  std::size_t elementNum_,
                                  std::size_t intervalCount_,
                                  std::size_t lutNum_)
        : bitWidthIn(bitWidthIn_),
          bitWidthOut(bitWidthOut_),
          elementNum(elementNum_),
          intervalCount(intervalCount_),
          lutNum(lutNum_),
          expectedE(elementNum_, T(0)),
          expectedT(elementNum_ * lutNum_, T(0))
    {
        const std::size_t valueBits   = sizeof(T) * 8;
        const std::size_t elementSize = sizeof(T);
        const T           maskIn      = (bitWidthIn >= valueBits) ? ~T(0) : ((T(1) << bitWidthIn) - T(1));
        const T           maskOut     = (bitWidthOut >= valueBits) ? ~T(0) : ((T(1) << bitWidthOut) - T(1));

        FastFss::prng::cpu::Prng prng;

        seed0.resize(elementNum * 16);
        seed1.resize(elementNum * 16);
        alpha.resize(elementNum);
        x.resize(elementNum);
        maskedX.resize(elementNum);
        left.resize(intervalCount);
        right.resize(intervalCount);
        table.resize(intervalCount * lutNum);

        prng.generate(seed0.data(), 8, elementNum * 16);
        prng.generate(seed1.data(), 8, elementNum * 16);
        prng.generate(x.data(), bitWidthIn, elementSize, elementNum);
        prng.generate(table.data(), bitWidthOut, elementSize, intervalCount * lutNum);

        fillSignedNBitFullPartition(left, right, bitWidthIn, intervalCount);

        for (T& value : alpha)
        {
            value = T(0);
        }
        for (T& value : x)
        {
            value = value & maskIn;
        }
        for (T& value : table)
        {
            value = value & maskOut;
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            maskedX[i] = (alpha[i] + x[i]) & maskIn;
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            std::size_t matched = intervalCount;
            for (std::size_t k = 0; k < intervalCount; ++k)
            {
                if (signedNBitIntervalContains(maskedX[i], left[k], right[k], bitWidthIn))
                {
                    matched = k;
                    break;
                }
            }
            int cnt = 0;
            for (std::size_t k = 0; k < intervalCount; ++k)
            {
                if (signedNBitIntervalContains(maskedX[i], left[k], right[k], bitWidthIn))
                {
                    cnt++;
                }
            }
            if (cnt > 1)
            {
                throw std::runtime_error("Multiple matches");
            }
            expectedE[i] = matched == intervalCount ? T(0) : T(1);
            for (std::size_t j = 0; j < lutNum; ++j)
            {
                expectedT[i * lutNum + j] = (matched == intervalCount) ? T(0) : table[matched + j * intervalCount];
            }
        }
    }
};

} // namespace FastFss::tests::grotto

#endif
