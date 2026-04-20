#ifndef FAST_FSS_TESTS_GROTTO_EVAL_TEST_CASE_H
#define FAST_FSS_TESTS_GROTTO_EVAL_TEST_CASE_H

#include <FastFssPP/prng.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace FastFss::tests::grotto {

template <typename T>
struct GrottoEvalTestCase
{
    std::size_t bitWidthIn;
    std::size_t elementNum;

    std::vector<std::uint8_t> seed0;
    std::vector<std::uint8_t> seed1;
    std::vector<T>            alpha;
    std::vector<T>            x;
    std::vector<T>            maskedX;
    std::vector<T>            expectedLt;
    std::vector<T>            expectedLe;

    GrottoEvalTestCase(std::size_t bitWidthIn_, std::size_t elementNum_)
        : bitWidthIn(bitWidthIn_), elementNum(elementNum_), expectedLt(elementNum_, T(0)), expectedLe(elementNum_, T(0))
    {
        const std::size_t valueBits   = sizeof(T) * 8;
        const std::size_t elementSize = sizeof(T);
        const T           maskIn      = (bitWidthIn >= valueBits) ? ~T(0) : ((T(1) << bitWidthIn) - T(1));

        FastFss::prng::cpu::Prng prng;

        seed0.resize(elementNum * 16);
        seed1.resize(elementNum * 16);
        alpha.resize(elementNum);
        x.resize(elementNum);
        maskedX.resize(elementNum);

        prng.generate(seed0.data(), 8, elementNum * 16);
        prng.generate(seed1.data(), 8, elementNum * 16);
        prng.generate(alpha.data(), bitWidthIn, elementNum);
        prng.generate(x.data(), bitWidthIn, elementNum);

        for (T& value : alpha)
        {
            value = value & maskIn;
        }
        for (T& value : x)
        {
            value = value & maskIn;
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            maskedX[i] = (alpha[i] + x[i]) & maskIn;
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            expectedLt[i] = maskedX[i] < alpha[i] ? T(1) : T(0);
            expectedLe[i] = maskedX[i] <= alpha[i] ? T(1) : T(0);
        }
    }
};

} // namespace FastFss::tests::grotto

#endif
