#ifndef FAST_FSS_TESTS_DPF_EVAL_ALL_CASE_H
#define FAST_FSS_TESTS_DPF_EVAL_ALL_CASE_H

#include <FastFssPP/prng.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace FastFss::tests::dpf {

template <typename T>
struct DpfEvalAllTestCase
{
    std::size_t bitWidthIn;
    std::size_t bitWidthOut;
    std::size_t elementNum;
    std::size_t groupSize;

    std::vector<std::uint8_t> seed0;
    std::vector<std::uint8_t> seed1;
    std::vector<T>            alpha;
    std::vector<T>            beta;
    std::vector<T>            x;
    std::vector<T>            maskedX;
    std::vector<T>            expected;
    std::size_t               domainSize;

    DpfEvalAllTestCase(std::size_t bitWidthIn_,
                       std::size_t bitWidthOut_,
                       std::size_t elementNum_,
                       std::size_t groupSize_)
        : bitWidthIn(bitWidthIn_),
          bitWidthOut(bitWidthOut_),
          elementNum(elementNum_),
          groupSize(groupSize_),
          expected(elementNum_ * (std::size_t{1} << bitWidthIn_) * groupSize_, T(0)),
          domainSize(std::size_t{1} << bitWidthIn_)
    {
        std::size_t elementSize = sizeof(T);
        T           maskIn      = (bitWidthIn == elementSize * 8) ? ~T(0) : (T(1) << bitWidthIn) - 1;
        T           maskOut     = (bitWidthOut == elementSize * 8) ? ~T(0) : (T(1) << bitWidthOut) - 1;

        FastFss::prng::cpu::Prng prng;

        seed0.resize(elementNum * 16);
        seed1.resize(elementNum * 16);
        alpha.resize(elementNum);
        beta.resize(elementNum * groupSize);
        x.resize(elementNum);
        maskedX.resize(elementNum);

        prng.generate(seed0.data(), 8, elementNum * 16);
        prng.generate(seed1.data(), 8, elementNum * 16);
        prng.generate(alpha.data(), bitWidthIn, elementNum);
        prng.generate(beta.data(), bitWidthOut, elementNum * groupSize);
        prng.generate(x.data(), bitWidthIn, elementNum);

        for (T& value : alpha)
        {
            value &= maskIn;
        }
        for (T& value : beta)
        {
            value &= maskOut;
        }
        for (T& value : x)
        {
            value &= maskIn;
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            maskedX[i] = (alpha[i] + x[i]) & maskIn;
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            std::size_t point = static_cast<std::size_t>(x[i]);
            for (std::size_t j = 0; j < groupSize; ++j)
            {
                expected[i * domainSize * groupSize + point * groupSize + j] = beta[i * groupSize + j];
            }
        }
    }
};

} // namespace FastFss::tests::dpf

#endif
