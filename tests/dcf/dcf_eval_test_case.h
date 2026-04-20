#ifndef FAST_FSS_TESTS_DCF_EVAL_TEST_CASE_H
#define FAST_FSS_TESTS_DCF_EVAL_TEST_CASE_H

#include <FastFssPP/prng.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace FastFss::tests::dcf {

template <typename T>
struct DcfEvalTestCase
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
    std::size_t elementNum;
    std::size_t groupSize;

    std::vector<T>            x;
    std::vector<std::uint8_t> seed0;
    std::vector<std::uint8_t> seed1;
    std::vector<T>            alpha;
    std::vector<T>            beta;
    std::vector<T>            maskedX;
    std::vector<T>            y;

    DcfEvalTestCase(std::size_t _bitWidthIn, std::size_t _bitWidthOut, std::size_t _elementNum, std::size_t _groupSize)
    {
        bitWidthIn  = _bitWidthIn;
        bitWidthOut = _bitWidthOut;
        elementNum  = _elementNum;
        groupSize   = _groupSize;

        FastFss::prng::cpu::Prng prng;

        std::size_t elementSize = sizeof(T);

        T maskIn  = (bitWidthIn == elementSize * 8) ? ~T(0) : (T(1) << bitWidthIn) - 1;
        T maskOut = (bitWidthOut == elementSize * 8) ? ~T(0) : (T(1) << bitWidthOut) - 1;

        // init x
        x.resize(elementNum);
        prng.generate(x.data(), bitWidthIn, elementNum);
        for (T& value : x)
        {
            value = value & maskIn;
        }

        // init seed
        seed0.resize(elementNum * 16);
        seed1.resize(elementNum * 16);
        prng.generate(seed0.data(), 8, elementNum * 16);
        prng.generate(seed1.data(), 8, elementNum * 16);

        // init alpha and beta
        alpha.resize(elementNum);
        beta.resize(elementNum * groupSize);
        prng.generate(alpha.data(), bitWidthIn, elementNum);
        prng.generate(beta.data(), bitWidthOut, elementNum * groupSize);
        for (T& value : alpha)
        {
            value = value & maskIn;
        }
        for (T& value : beta)
        {
            value = value & maskOut;
        }

        // eval
        maskedX.resize(elementNum);
        y.assign(elementNum * groupSize, T(0));
        for (std::size_t i = 0; i < elementNum; ++i)
        {
            maskedX[i] = (x[i] + alpha[i]) & maskIn;
            if (maskedX[i] < alpha[i])
            {
                for (std::size_t j = 0; j < groupSize; ++j)
                {
                    y[i * groupSize + j] = beta[i * groupSize + j];
                }
            }
        }
    }
};

} // namespace FastFss::tests::dcf

#endif
