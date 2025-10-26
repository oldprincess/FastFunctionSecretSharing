#ifndef TEST_CPU_GROTTO_TEST_GROTTO_EVAL_HPP
#define TEST_CPU_GROTTO_TEST_GROTTO_EVAL_HPP

#include <FastFss/cpu/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"

template <typename GroupElement>
class TestGrottoEval
{
    static GroupElement mod_bits(GroupElement x, std::size_t bitWidth)
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
    static void run(std::size_t bitWidthIn,
                    std::size_t elementNum,
                    MT19937Rng &rng)
    {
        std::printf("[cpu test GrottoEval] "     //
                    "elementSize = %2d "         //
                    "bitWidthIn  = %3d "         //
                    "elementNum  = %5d .... ",   //
                    (int)(sizeof(GroupElement)), //
                    (int)bitWidthIn,             //
                    (int)elementNum              //
        );                                       //

        int ret = 0;

        std::size_t keyDataSize       = 0;
        std::size_t cacheDataSize     = 0;
        std::size_t elementSize       = sizeof(GroupElement);
        std::size_t alphaDataSize     = elementNum * elementSize;
        std::size_t seedDataSize0     = 16 * elementNum;
        std::size_t seedDataSize1     = 16 * elementNum;
        std::size_t maskedXDataSize   = elementNum * elementSize;
        std::size_t sharedOutDataSize = elementNum * elementSize;

        ret = FastFss_cpu_grottoGetKeyDataSize(               //
            &keyDataSize, bitWidthIn, elementSize, elementNum //
        );                                                    //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }
        ret = FastFss_cpu_grottoGetCacheDataSize(               //
            &cacheDataSize, bitWidthIn, elementSize, elementNum //
        );                                                      //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        auto x          = std::make_unique<GroupElement[]>(elementNum);
        auto key        = std::make_unique<std::uint8_t[]>(keyDataSize);
        auto cache      = std::make_unique<std::uint8_t[]>(cacheDataSize);
        auto alpha      = std::make_unique<GroupElement[]>(elementNum);
        auto seed0      = std::make_unique<std::uint8_t[]>(seedDataSize0);
        auto seed1      = std::make_unique<std::uint8_t[]>(seedDataSize1);
        auto maskedX    = std::make_unique<GroupElement[]>(elementNum);
        auto sharedOut0 = std::make_unique<GroupElement[]>(elementNum);
        auto sharedOut1 = std::make_unique<GroupElement[]>(elementNum);
        auto sharedOut  = std::make_unique<GroupElement[]>(elementNum);

        rng.gen(x.get(), elementNum * elementSize);
        rng.gen(alpha.get(), alphaDataSize);
        rng.gen(seed0.get(), seedDataSize0);
        rng.gen(seed1.get(), seedDataSize1);

        ret = FastFss_cpu_grottoKeyGen( //
            key.get(), keyDataSize,     //
            alpha.get(), alphaDataSize, //
            seed0.get(), seedDataSize0, //
            seed1.get(), seedDataSize1, //
            bitWidthIn,                 //
            elementSize,                //
            elementNum                  //
        );                              //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        for (std::size_t i = 0; i < elementNum; i++)
        {
            maskedX[i] = x[i] + alpha[i];
        }

        // ==================================================
        // ================== Equal Bound is False ==========
        // ==================================================

        ret = FastFss_cpu_grottoEval(            //
            sharedOut0.get(), sharedOutDataSize, //
            maskedX.get(), maskedXDataSize,      //
            key.get(), keyDataSize,              //
            seed0.get(), seedDataSize0,          //
            false,                               //
            0,                                   //
            bitWidthIn,                          //
            elementSize,                         //
            elementNum,                          //
            cache.get(), cacheDataSize           //
        );                                       //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        ret = FastFss_cpu_grottoEval(            //
            sharedOut1.get(), sharedOutDataSize, //
            maskedX.get(), maskedXDataSize,      //
            key.get(), keyDataSize,              //
            seed1.get(), seedDataSize1,          //
            false,                               //
            1,                                   //
            bitWidthIn,                          //
            elementSize,                         //
            elementNum,                          //
            nullptr, 0                           //
        );                                       //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        for (std::size_t i = 0; i < elementNum; i++)
        {
            sharedOut[i] = sharedOut0[i] + sharedOut1[i];
        }

        for (std::size_t i = 0; i < elementNum; i++)
        {
            maskedX[i]   = mod_bits(maskedX[i], bitWidthIn);
            alpha[i]     = mod_bits(alpha[i], bitWidthIn);
            sharedOut[i] = mod_bits(sharedOut[i], 1);

            bool cmp0 = ((maskedX[i] < alpha[i]) && (sharedOut[i] == 1));
            bool cmp1 = ((maskedX[i] >= alpha[i]) && (sharedOut[i] == 0));
            if (!(cmp0 || cmp1))
            {
                std::printf("Error. %s:%d\n", __FILE__, __LINE__);
                std::exit(-1);
            }
        }

        // ==================================================
        // ================== Equal Bound is True ===========
        // ==================================================

        ret = FastFss_cpu_grottoEval(            //
            sharedOut0.get(), sharedOutDataSize, //
            maskedX.get(), maskedXDataSize,      //
            key.get(), keyDataSize,              //
            seed0.get(), seedDataSize0,          //
            true,                                //
            0,                                   //
            bitWidthIn,                          //
            elementSize,                         //
            elementNum,                          //
            cache.get(), cacheDataSize           //
        );                                       //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        ret = FastFss_cpu_grottoEval(            //
            sharedOut1.get(), sharedOutDataSize, //
            maskedX.get(), maskedXDataSize,      //
            key.get(), keyDataSize,              //
            seed1.get(), seedDataSize1,          //
            true,                                //
            1,                                   //
            bitWidthIn,                          //
            elementSize,                         //
            elementNum,                          //
            nullptr, 0                           //
        );                                       //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        for (std::size_t i = 0; i < elementNum; i++)
        {
            sharedOut[i] = sharedOut0[i] + sharedOut1[i];
        }

        for (std::size_t i = 0; i < elementNum; i++)
        {
            maskedX[i]   = mod_bits(maskedX[i], bitWidthIn);
            alpha[i]     = mod_bits(alpha[i], bitWidthIn);
            sharedOut[i] = mod_bits(sharedOut[i], 1);

            bool cmp0 = ((maskedX[i] <= alpha[i]) && (sharedOut[i] == 1));
            bool cmp1 = ((maskedX[i] > alpha[i]) && (sharedOut[i] == 0));
            if (!(cmp0 || cmp1))
            {
                std::printf("Error. %s:%d\n", __FILE__, __LINE__);
                std::exit(-1);
            }
        }

        std::printf("pass \n");
    }
};

#endif