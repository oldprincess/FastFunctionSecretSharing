#ifndef TEST_CPU_GROTTO_TEST_GROTTO_MIC_HPP
#define TEST_CPU_GROTTO_TEST_GROTTO_MIC_HPP

#include <FastFss/cpu/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"

template <typename GroupElement>
class TestGrottoMIC
{
    static constexpr GroupElement mod_bits(GroupElement x,
                                           int          bitWidth) noexcept
    {
        if (bitWidth == sizeof(GroupElement) * 8)
        {
            return x;
        }
        else
        {
            return x & (((GroupElement)1 << bitWidth) - 1);
        }
    }

public:
    static void run(std::size_t                     bitWidthIn,
                    std::size_t                     elementNum,
                    const std::vector<GroupElement> leftBoundary,
                    const std::vector<GroupElement> rightBoundary,
                    MT19937Rng&                     rng)
    {
        std::printf("[cpu test GrottoMIC] elementSize = %2d bitWidthIn = %3d "
                    "elementNum = %5d",
                    (int)(sizeof(GroupElement)), (int)bitWidthIn,
                    (int)elementNum);

        std::size_t intervalNum = leftBoundary.size();

        std::vector<GroupElement> x(elementNum);
        std::vector<GroupElement> maskedX(elementNum);
        std::vector<GroupElement> alpha(elementNum);
        std::vector<GroupElement> sharedOut0(elementNum * intervalNum);
        std::vector<GroupElement> sharedOut1(elementNum * intervalNum);

        std::vector<std::uint8_t> seed0(elementNum * 16);
        std::vector<std::uint8_t> seed1(elementNum * 16);

        void*       grottoMICKey = nullptr;
        std::size_t grottoMICKeyDataSize;
        FastFss_cpu_grottoGetKeyDataSize(&grottoMICKeyDataSize, bitWidthIn,
                                         sizeof(GroupElement), elementNum);
        grottoMICKey = malloc(grottoMICKeyDataSize);

        void*       cache = nullptr;
        std::size_t cacheDataSize;
        FastFss_cpu_grottoGetCacheDataSize(&cacheDataSize, bitWidthIn,
                                           sizeof(GroupElement), elementNum);
        cache = malloc(cacheDataSize);

        rng.gen(seed0.data(), seed0.size());
        rng.gen(seed1.data(), seed1.size());

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            x[i]       = rng.rand<GroupElement>();
            alpha[i]   = rng.rand<GroupElement>();
            maskedX[i] = x[i] + alpha[i];

            x[i]       = mod_bits(x[i], bitWidthIn);
            alpha[i]   = mod_bits(alpha[i], bitWidthIn);
            maskedX[i] = mod_bits(maskedX[i], bitWidthIn);
        }

        int ret0 = FastFss_cpu_grottoKeyGen(
            grottoMICKey, grottoMICKeyDataSize,  //
            alpha.data(),                        //
            alpha.size() * sizeof(GroupElement), //
            seed0.data(),                        //
            seed0.size(),                        //
            seed1.data(),                        //
            seed1.size(),                        //
            bitWidthIn, sizeof(GroupElement), elementNum);
        if (ret0 != 0)
        {
            std::printf("[err] FastFss_cpu_grottoKeyGen failed ret = %d\n",
                        ret0);
            return;
        }

        int ret1 = FastFss_cpu_grottoMICEval(
            sharedOut0.data(),                           //
            sharedOut0.size() * sizeof(GroupElement),    //
            maskedX.data(),                              //
            maskedX.size() * sizeof(GroupElement),       //
            grottoMICKey,                                //
            grottoMICKeyDataSize,                        //
            seed0.data(),                                //
            seed0.size(),                                //
            0,                                           //
            leftBoundary.data(),                         //
            leftBoundary.size() * sizeof(GroupElement),  //
            rightBoundary.data(),                        //
            rightBoundary.size() * sizeof(GroupElement), //
            bitWidthIn, sizeof(GroupElement), elementNum, cache, cacheDataSize);
        if (ret1 != 0)
        {
            std::free(grottoMICKey);
            std::printf("[err] FastFss_cpu_grottoMICEval failed ret = %d\n",
                        ret1);
            return;
        }

        int ret2 = FastFss_cpu_grottoMICEval(
            sharedOut1.data(),                           //
            sharedOut1.size() * sizeof(GroupElement),    //
            maskedX.data(),                              //
            maskedX.size() * sizeof(GroupElement),       //
            grottoMICKey,                                //
            grottoMICKeyDataSize,                        //
            seed1.data(),                                //
            seed1.size(),                                //
            1,                                           //
            leftBoundary.data(),                         //
            leftBoundary.size() * sizeof(GroupElement),  //
            rightBoundary.data(),                        //
            rightBoundary.size() * sizeof(GroupElement), //
            bitWidthIn, sizeof(GroupElement), elementNum, cache, cacheDataSize);
        if (ret2 != 0)
        {
            std::free(grottoMICKey);
            std::printf("[err] FastFss_cpu_grottoMICEval failed ret = %d\n",
                        ret2);
            return;
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            for (std::size_t k = 0; k < intervalNum; ++k)
            {
                GroupElement v = (sharedOut0[i * intervalNum + k] +
                                  sharedOut1[i * intervalNum + k]) &
                                 1;

                bool cmp1 = (                                                //
                    (leftBoundary[k] <= x[i] && x[i] <= rightBoundary[k])    //
                    == (v == 1)                                              //
                );                                                           //
                bool cmp2 = (                                                //
                    (!(leftBoundary[k] <= x[i] && x[i] <= rightBoundary[k])) //
                    == (v == 0)                                              //
                );                                                           //
                if (!(cmp1 && cmp2))
                {
                    std::printf("\n[err] [%d] x = %lld, alpha = %lld, left = "
                                "%lld, right = %lld, v = %lld\n",
                                (int)i, (long long)x[i], (long long)alpha[i],
                                (long long)leftBoundary[k],
                                (long long)rightBoundary[k], (long long)v);
                    std::exit(-1);
                }
            }
        }

        std::free(grottoMICKey);
        std::free(cache);
        std::puts("  pass");
    }
};

#endif