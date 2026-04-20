#ifndef FAST_FSS_TESTS_OTTT_LUT_EVAL_CASE_H
#define FAST_FSS_TESTS_OTTT_LUT_EVAL_CASE_H

#include <FastFssPP/prng.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace FastFss::tests::ottt {

template <typename T>
struct OtttLutEvalTestCase
{
    std::size_t    bitWidthIn;
    std::size_t    lutsNum;
    std::size_t    elementNum;
    std::size_t    domainSize;
    T              maskIn;
    T              maskOut;
    std::vector<T> alpha;
    std::vector<T> x;
    std::vector<T> maskedX;
    std::vector<T> lut;
    std::vector<T> expected;

    OtttLutEvalTestCase(std::size_t bitWidthIn_, std::size_t lutsNum_, std::size_t elementNum_)
        : bitWidthIn(bitWidthIn_), lutsNum(lutsNum_), elementNum(elementNum_), domainSize(std::size_t{1} << bitWidthIn_)
    {
        std::size_t valueBits = sizeof(T) * 8;

        maskIn  = (bitWidthIn >= valueBits) ? ~T(0) : ((T(1) << bitWidthIn) - T(1));
        maskOut = ~T(0);

        FastFss::prng::cpu::Prng prng;

        alpha.resize(elementNum);
        x.resize(elementNum);
        maskedX.resize(elementNum);
        lut.resize(domainSize * lutsNum);

        prng.generate(alpha.data(), bitWidthIn, elementNum);
        prng.generate(x.data(), bitWidthIn, elementNum);
        prng.generate(lut.data(), valueBits, domainSize * lutsNum);

        for (T& value : alpha)
        {
            value &= maskIn;
        }
        for (T& value : x)
        {
            value &= maskIn;
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            maskedX[i] = (alpha[i] + x[i]) & maskIn;
        }

        expected.resize(elementNum * lutsNum);
        for (std::size_t i = 0; i < elementNum; ++i)
        {
            for (std::size_t j = 0; j < lutsNum; ++j)
            {
                expected[i * lutsNum + j] = lut[std::size_t(x[i] & maskOut) + j * domainSize];
            }
        }
    }
};

} // namespace FastFss::tests::ottt

#endif
