#ifndef TEST_CPU_DPF_TEST_DPF_MULTI_EVAL_HPP
#define TEST_CPU_DPF_TEST_DPF_MULTI_EVAL_HPP

#include <FastFss/cpu/dpf.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"

template <typename T>
class TestDpfMultiEval
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
    static void run(std::size_t pointNum,
                    std::size_t bitWidthIn,
                    std::size_t bitWidthOut,
                    std::size_t groupSize,
                    std::size_t elementNum,
                    MT19937Rng &rng)
    {
        std::printf("[cpu test DpfMultiEval] " //
                    "elementSize = %2d "       //
                    "pointNum    = %3d "       //
                    "bitWidthIn  = %3d "       //
                    "bitWidthOut = %3d "       //
                    "groupSize   = %3d "       //
                    "elementNum  = %5d .... ", //
                    (int)(sizeof(T)),          //
                    (int)pointNum,             //
                    (int)bitWidthIn,           //
                    (int)bitWidthOut,          //
                    (int)groupSize,            //
                    (int)elementNum            //
        );                                     //

        int ret = 0;

        std::size_t m                 = pointNum;
        std::size_t keyDataSize       = 0;
        std::size_t cacheDataSize     = 0;
        std::size_t elementSize       = sizeof(T);
        std::size_t alphaDataSize     = elementNum * elementSize;
        std::size_t betaDataSize      = elementNum * elementSize * groupSize;
        std::size_t seedDataSize0     = 16 * elementNum;
        std::size_t seedDataSize1     = 16 * elementNum;
        std::size_t maskedXDataSize   = elementNum * elementSize;
        std::size_t sharedOutDataSize = (            //
            elementNum * elementSize * m * groupSize //
        );                                           //
        std::size_t pointDataSize     = elementSize * m;

        ret = FastFss_cpu_dpfGetKeyDataSize( //
            &keyDataSize, bitWidthIn, bitWidthOut, groupSize, elementSize,
            elementNum //
        );             //
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
        auto sharedOut0 = std::make_unique<T[]>(elementNum * m * groupSize);
        auto sharedOut1 = std::make_unique<T[]>(elementNum * m * groupSize);
        auto sharedOut  = std::make_unique<T[]>(elementNum * m * groupSize);
        auto point      = std::make_unique<T[]>(m);

        rng.gen(x.get(), elementNum * elementSize);
        rng.gen(alpha.get(), alphaDataSize);
        rng.gen(beta.get(), betaDataSize);
        rng.gen(seed0.get(), seedDataSize0);
        rng.gen(seed1.get(), seedDataSize1);
        rng.gen(point.get(), pointDataSize);

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

        ret = FastFss_cpu_dpfMultiEval(          //
            sharedOut0.get(), sharedOutDataSize, //
            maskedX.get(), maskedXDataSize,      //
            key.get(), keyDataSize,              //
            seed0.get(), seedDataSize0,          //
            0,                                   //
            point.get(), pointDataSize,          //
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

        ret = FastFss_cpu_dpfMultiEval(          //
            sharedOut1.get(), sharedOutDataSize, //
            maskedX.get(), maskedXDataSize,      //
            key.get(), keyDataSize,              //
            seed1.get(), seedDataSize1,          //
            1,                                   //
            point.get(), pointDataSize,          //
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

        for (std::size_t i = 0; i < elementNum * m * groupSize; i++)
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
            x[i]              = mod_bits(x[i], bitWidthIn);
            maskedX[i]        = mod_bits(maskedX[i], bitWidthIn);
            alpha[i]          = mod_bits(alpha[i], bitWidthIn);
            auto sharedOutPtr = sharedOut.get() + m * i * groupSize;
            for (std::size_t j = 0; j < m; j++)
            {
                point[j] = mod_bits(point[j], bitWidthIn);
                if (x[i] == point[j])
                {
                    for (std::size_t k = 0; k < groupSize; k++)
                    {
                        if (sharedOutPtr[j * groupSize + k] !=
                            beta[i * groupSize + k])
                        {
                            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
                            std::exit(-1);
                        }
                    }
                }
                else
                {
                    for (std::size_t k = 0; k < groupSize; k++)
                    {
                        if (sharedOutPtr[j * groupSize + k] != 0)
                        {
                            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
                            std::exit(-1);
                        }
                    }
                }
            }
        }

        std::printf("pass \n");
    }
};

#endif