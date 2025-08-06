#ifndef TEST_CPU_GROTTO_TEST_GROTTO_LUT_EX2_HPP
#define TEST_CPU_GROTTO_TEST_GROTTO_LUT_EX2_HPP

#include <FastFss/cpu/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"

template <typename GroupElement>
class TestGrottoLut_ex2
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
    static void run(std::size_t lutBitWidth,
                    std::size_t bitWidthIn,
                    std::size_t elementNum,
                    MT19937Rng &rng)
    {
        using namespace std::chrono;

        std::printf("[cpu test GrottoLutEval_ex2] "
                    "elementSize = %3d "
                    "lutBitWidth = %3d "
                    "bitWidthIn = %3d "
                    "elementNum = %5d ",
                    (int)sizeof(GroupElement), //
                    (int)lutBitWidth,          //
                    (int)bitWidthIn,           //
                    (int)elementNum            //
        );

        std::size_t genKeyTimeUs = 0;
        std::size_t eval1TimeUs  = 0;
        std::size_t eval2TimeUs  = 0;

        std::size_t alphaDataSize   = sizeof(GroupElement) * elementNum;
        std::size_t maskedXDataSize = sizeof(GroupElement) * elementNum;
        std::size_t seedDataSize0   = 16 * elementNum;
        std::size_t seedDataSize1   = 16 * elementNum;
        std::size_t lutsDataSize = sizeof(GroupElement) * (1ULL << lutBitWidth);
        std::size_t grottoKeyDataSize;
        FastFss_cpu_grottoGetKeyDataSize(&grottoKeyDataSize, bitWidthIn,
                                         sizeof(GroupElement), elementNum);
        std::size_t cacheDataSize;
        FastFss_cpu_grottoGetCacheDataSize(&cacheDataSize, bitWidthIn,
                                           sizeof(GroupElement), elementNum);

        std::unique_ptr<GroupElement[]> sharedOutE0( //
            new GroupElement[elementNum]             //
        );                                           //
        std::unique_ptr<GroupElement[]> sharedOutE1( //
            new GroupElement[elementNum]             //
        );                                           //
        std::unique_ptr<GroupElement[]> sharedOutT0( //
            new GroupElement[elementNum]             //
        );                                           //
        std::unique_ptr<GroupElement[]> sharedOutT1( //
            new GroupElement[elementNum]             //
        );                                           //
        std::unique_ptr<GroupElement[]> alpha(new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> x(new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> maskedX(new GroupElement[elementNum]);
        std::unique_ptr<std::uint8_t[]> grottoKey( //
            new std::uint8_t[grottoKeyDataSize]    //
        );                                         //
        std::unique_ptr<std::uint8_t[]> seed0(     //
            new std::uint8_t[16 * elementNum]      //
        );                                         //
        std::unique_ptr<std::uint8_t[]> seed1(     //
            new std::uint8_t[16 * elementNum]      //
        );                                         //
        std::unique_ptr<std::uint8_t[]> cache(     //
            new std::uint8_t[cacheDataSize]        //
        );                                         //

        std::vector<GroupElement> luts(1ULL << lutBitWidth);
        for (std::size_t i = 0; i < luts.size(); i++)
        {
            luts[i] = mod_bits(luts.size() - i - 1, bitWidthIn);
        }
        std::vector<GroupElement> points(1ULL << lutBitWidth);
        for (std::size_t i = 0; i < points.size(); i++)
        {
            points[i] = i;
        }

        for (std::size_t i = 0; i < elementNum; i++)
        {
            alpha[i] = rng.rand<GroupElement>();
            alpha[i] = mod_bits(alpha[i], bitWidthIn);
        }

        rng.gen(seed0.get(), seedDataSize0);
        rng.gen(seed1.get(), seedDataSize1);

        for (std::size_t i = 0; i < elementNum; i++)
        {
            x[i] = rng.rand<GroupElement>();
            x[i] = mod_bits(x[i], lutBitWidth);
            if (rng.rand<int>() & 1)
            {
                x[i] = 0;
            }
            maskedX[i] = x[i] + alpha[i];
            maskedX[i] = mod_bits(maskedX[i], bitWidthIn);
        }
        {
            auto start_time = high_resolution_clock::now();
            int  ret1       = FastFss_cpu_grottoKeyGen( //
                grottoKey.get(),                 //
                grottoKeyDataSize,               //
                alpha.get(),                     ///
                alphaDataSize,                   //
                seed0.get(),                     //
                seedDataSize0,                   //
                seed1.get(),                     //
                seedDataSize1,                   //
                bitWidthIn,                      //
                sizeof(GroupElement),            //
                elementNum);
            auto stop_time  = high_resolution_clock::now();
            if (ret1 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_grottoKeyGen ret = %d\n",
                            __LINE__, ret1);
                std::exit(-1);
            }
            genKeyTimeUs =
                duration_cast<microseconds>(stop_time - start_time).count();
        }

        {
            auto start_time = high_resolution_clock::now();
            int  ret2       = FastFss_cpu_grottoLutEval_ex2( //
                sharedOutE0.get(),                    //
                sharedOutT0.get(),                    //
                maskedX.get(),                        //
                maskedXDataSize,                      //
                grottoKey.get(),                      //
                grottoKeyDataSize,                    //
                seed0.get(),                          //
                seedDataSize0,                        //
                0,                                    //
                points.data(),                        //
                points.size() * sizeof(GroupElement), //
                luts.data(),                          //
                lutsDataSize,                         //
                bitWidthIn,                           //
                sizeof(GroupElement) * 8,             //
                sizeof(GroupElement),                 //
                elementNum,                           //
                cache.get(),                          //
                cacheDataSize);
            auto stop_time  = high_resolution_clock::now();
            if (ret2 != 0)
            {
                std::printf(
                    "\n[%d] err. FastFss_cpu_grottoLutEval_ex2 ret = %d\n",
                    __LINE__, ret2);
                std::exit(-1);
            }
            eval1TimeUs =
                duration_cast<microseconds>(stop_time - start_time).count();
        }

        {
            auto start_time = high_resolution_clock::now();
            int  ret3       = FastFss_cpu_grottoLutEval_ex2( //
                sharedOutE1.get(),                    //
                sharedOutT1.get(),                    //
                maskedX.get(),                        //
                maskedXDataSize,                      //
                grottoKey.get(),                      //
                grottoKeyDataSize,                    //
                seed1.get(),                          //
                seedDataSize1,                        //
                1,                                    //
                points.data(),                        //
                points.size() * sizeof(GroupElement), //
                luts.data(),                          //
                lutsDataSize,                         //
                bitWidthIn,                           //
                sizeof(GroupElement) * 8,             //
                sizeof(GroupElement),                 //
                elementNum,                           //
                cache.get(),                          //
                cacheDataSize);
            auto stop_time  = high_resolution_clock::now();
            if (ret3 != 0)
            {
                std::printf(
                    "\n[%d] err. FastFss_cpu_grottoLutEval_ex2 ret = %d\n",
                    __LINE__, ret3);
                std::exit(-1);
            }
            eval2TimeUs =
                duration_cast<microseconds>(stop_time - start_time).count();
        }

        for (size_t i = 0; i < elementNum; i++)
        {
            std::size_t  idx = (std::size_t)x[i];
            GroupElement e   = (sharedOutE0[i] + sharedOutE1[i]) & 1;
            GroupElement t   = (sharedOutT0[i] + sharedOutT1[i]);
            GroupElement v =
                mod_bits(t + e * t * (GroupElement)(-2), bitWidthIn);

            if (luts[idx] == v)
            {
                continue;
            }
            std::printf("\n[%d] "
                        "alpha = %lld "
                        "x = %lld "
                        "v = %lld",
                        __LINE__,            //
                        (long long)alpha[i], //
                        (long long)x[i],     //
                        (long long)v);
            std::exit(-1);
        }

        std::puts("  pass");
        std::printf(
            "\tgenKeyTime = %zu us, eval1Time = %zu us, eval2Time = %zu us\n",
            genKeyTimeUs, eval1TimeUs, eval2TimeUs);
    }
};

#endif