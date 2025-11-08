#ifndef TEST_CPU_DPF_TEST_DPF_EVAL_HPP
#define TEST_CPU_DPF_TEST_DPF_EVAL_HPP

#include <FastFss/cpu/dpf.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"

template <typename T>
class TestDpfEval
{
    static T mod_bits(T x, std::size_t bitWidth)
    {
        if (bitWidth == sizeof(T) * 8)
        {
            return x;
        }
        else
        {
            return x & (((T)1 << bitWidth) - 1);
        }
    }

public:
    static void run(std::size_t bitWidthIn,
                    std::size_t bitWidthOut,
                    std::size_t groupSize,
                    std::size_t elementNum,
                    MT19937Rng &rng)
    {
        std::printf("[cpu test DpfEval] "      //
                    "elementSize = %2d "       //
                    "bitWidthIn  = %3d "       //
                    "bitWidthOut = %3d "       //
                    "groupSize   = %2d "       //
                    "elementNum  = %5d .... ", //
                    (int)(sizeof(T)),          //
                    (int)bitWidthIn,           //
                    (int)bitWidthOut,          //
                    (int)groupSize,            //
                    (int)elementNum            //
        );                                     //

        int ret = 0;

        std::size_t keyDataSize       = 0;
        std::size_t cacheDataSize     = 0;
        std::size_t elementSize       = sizeof(T);
        std::size_t alphaDataSize     = elementNum * elementSize;
        std::size_t betaDataSize      = elementNum * elementSize * groupSize;
        std::size_t seedDataSize0     = 16 * elementNum;
        std::size_t seedDataSize1     = 16 * elementNum;
        std::size_t maskedXDataSize   = elementNum * elementSize;
        std::size_t sharedOutDataSize = elementNum * elementSize * groupSize;

        ret = FastFss_cpu_dpfGetKeyDataSize( //
            &keyDataSize,                    //
            bitWidthIn,                      //
            bitWidthOut,                     //
            groupSize,                       //
            elementSize,                     //
            elementNum                       //
        );                                   //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }
        ret = FastFss_cpu_dpfGetCacheDataSize(                  //
            &cacheDataSize, bitWidthIn, elementSize, elementNum //
        );                                                      //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        auto x          = std::make_unique<T[]>(elementNum);
        auto key        = std::make_unique<std::uint8_t[]>(keyDataSize);
        auto cache      = std::make_unique<std::uint8_t[]>(cacheDataSize);
        auto alpha      = std::make_unique<T[]>(elementNum);
        auto beta       = std::make_unique<T[]>(elementNum * groupSize);
        auto seed0      = std::make_unique<std::uint8_t[]>(seedDataSize0);
        auto seed1      = std::make_unique<std::uint8_t[]>(seedDataSize1);
        auto maskedX    = std::make_unique<T[]>(elementNum);
        auto sharedOut0 = std::make_unique<T[]>(elementNum * groupSize);
        auto sharedOut1 = std::make_unique<T[]>(elementNum * groupSize);
        auto sharedOut  = std::make_unique<T[]>(elementNum * groupSize);

        rng.gen(x.get(), elementNum * elementSize);
        rng.gen(alpha.get(), alphaDataSize);
        rng.gen(beta.get(), betaDataSize);
        rng.gen(seed0.get(), seedDataSize0);
        rng.gen(seed1.get(), seedDataSize1);

        ret = FastFss_cpu_dpfKeyGen(    //
            key.get(), keyDataSize,     //
            alpha.get(), alphaDataSize, //
            beta.get(), betaDataSize,   //
            seed0.get(), seedDataSize0, //
            seed1.get(), seedDataSize1, //
            bitWidthIn,                 //
            bitWidthOut,                //
            groupSize,                  //
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

        ret = FastFss_cpu_dpfEval(               //
            sharedOut0.get(), sharedOutDataSize, //
            maskedX.get(), maskedXDataSize,      //
            key.get(), keyDataSize,              //
            seed0.get(), seedDataSize0,          //
            0,                                   //
            bitWidthIn,                          //
            bitWidthOut,                         //
            groupSize,                           //
            elementSize,                         //
            elementNum,                          //
            cache.get(), cacheDataSize           //
        );                                       //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        ret = FastFss_cpu_dpfEval(               //
            sharedOut1.get(), sharedOutDataSize, //
            maskedX.get(), maskedXDataSize,      //
            key.get(), keyDataSize,              //
            seed1.get(), seedDataSize1,          //
            1,                                   //
            bitWidthIn,                          //
            bitWidthOut,                         //
            groupSize,                           //
            elementSize,                         //
            elementNum,                          //
            nullptr, 0                           //
        );                                       //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        for (std::size_t i = 0; i < elementNum * groupSize; i++)
        {
            sharedOut[i] = sharedOut0[i] + sharedOut1[i];
            sharedOut[i] = mod_bits(sharedOut[i], bitWidthOut);
        }
        for (std::size_t i = 0; i < elementNum * groupSize; i++)
        {
            beta[i] = mod_bits(beta[i], bitWidthOut);
        }

        for (std::size_t i = 0; i < elementNum; i++)
        {
            maskedX[i] = mod_bits(maskedX[i], bitWidthIn);
            alpha[i]   = mod_bits(alpha[i], bitWidthIn);
            if (maskedX[i] == alpha[i])
            {
                for (int j = 0; j < groupSize; j++)
                {
                    if (sharedOut[i * groupSize + j] != beta[i * groupSize + j])
                    {
                        std::printf("Error. %s:%d\n", __FILE__, __LINE__);
                        std::exit(-1);
                    }
                }
            }
            else
            {
                for (int j = 0; j < groupSize; j++)
                {
                    if (sharedOut[i * groupSize + j] != 0)
                    {
                        std::printf("Error. %s:%d\n", __FILE__, __LINE__);
                        std::exit(-1);
                    }
                }
            }
        }

        std::printf("pass \n");
    }
};

#endif