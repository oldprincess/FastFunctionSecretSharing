#ifndef TEST_CPU_GROTTO_TEST_GROTTO_EQ_MULTI_HPP
#define TEST_CPU_GROTTO_TEST_GROTTO_EQ_MULTI_HPP

#include <FastFss/cpu/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"

template <typename GroupElement>
class TestGrottoEqMulti
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
    static void run(std::size_t               bitWidthIn,
                    std::size_t               elementNum,
                    std::vector<GroupElement> points,
                    MT19937Rng               &rng)
    {
        std::printf("[cpu test GrottoEqMulti] "
                    "elementSize = %3d "
                    "bitWidthIn = %3d "
                    "elementNum = %5d "
                    "pointsNum = %5d",
                    (int)sizeof(GroupElement), //
                    (int)bitWidthIn,           //
                    (int)elementNum,           //
                    (int)points.size()         //
        );

        std::size_t outDataSize =
            sizeof(GroupElement) * elementNum * points.size();
        std::size_t alphaDataSize   = sizeof(GroupElement) * elementNum;
        std::size_t maskedXDataSize = sizeof(GroupElement) * elementNum;
        std::size_t seedDataSize0   = 16 * elementNum;
        std::size_t seedDataSize1   = 16 * elementNum;
        std::size_t grottoKeyDataSize;
        FastFss_cpu_grottoGetKeyDataSize(&grottoKeyDataSize, bitWidthIn,
                                         sizeof(GroupElement), elementNum);

        std::unique_ptr<GroupElement[]> sharedOut0(      //
            new GroupElement[elementNum * points.size()] //
        );                                               //
        std::unique_ptr<GroupElement[]> sharedOut1(      //
            new GroupElement[elementNum * points.size()] //
        );                                               //
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
            x[i] = mod_bits(x[i], bitWidthIn);
            if (rng.rand<int>() & 1)
            {
                x[i] = 0;
            }
            maskedX[i] = x[i] + alpha[i];
            maskedX[i] = mod_bits(maskedX[i], bitWidthIn);
        }

        {
            int ret1 = FastFss_cpu_grottoKeyGen( //
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
            if (ret1 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_grottoKeyGen ret = %d\n",
                            __LINE__, ret1);
                std::exit(-1);
            }
        }

        {
            int ret2 = FastFss_cpu_grottoEvalEqMulti( //
                sharedOut0.get(),                     //
                outDataSize,                          //
                maskedX.get(),                        //
                maskedXDataSize,                      //
                grottoKey.get(),                      //
                grottoKeyDataSize,                    //
                seed0.get(),                          //
                seedDataSize0,                        //
                0,                                    //
                points.data(),                        //
                points.size() * sizeof(GroupElement), //
                bitWidthIn,                           //
                sizeof(GroupElement),                 //
                elementNum,                           //
                nullptr,                              //
                0);
            if (ret2 != 0)
            {
                std::printf(
                    "\n[%d] err. FastFss_cpu_grottoEvalEqMulti ret = %d\n",
                    __LINE__, ret2);
                std::exit(-1);
            }
        }

        {
            int ret3 = FastFss_cpu_grottoEvalEqMulti( //
                sharedOut1.get(),                     //
                outDataSize,                          //
                maskedX.get(),                        //
                maskedXDataSize,                      //
                grottoKey.get(),                      //
                grottoKeyDataSize,                    //
                seed1.get(),                          //
                seedDataSize1,                        //
                1,                                    //
                points.data(),                        //
                points.size() * sizeof(GroupElement), //
                bitWidthIn,                           //
                sizeof(GroupElement),                 //
                elementNum,                           //
                nullptr,                              //
                0);
            if (ret3 != 0)
            {
                std::printf(
                    "\n[%d] err. FastFss_cpu_grottoEvalEqMulti ret = %d\n",
                    __LINE__, ret3);
                std::exit(-1);
            }
        }

        for (size_t i = 0; i < elementNum; i++)
        {
            size_t pointNum = points.size();
            for (size_t j = 0; j < pointNum; j++)
            {
                size_t       offset = i * pointNum + j;
                GroupElement v = (sharedOut0[offset] + sharedOut1[offset]) & 1;
                if (x[i] == mod_bits(points[j], bitWidthIn) && v == 1)
                {
                    continue;
                }
                if (x[i] != mod_bits(points[j], bitWidthIn) && v == 0)
                {
                    continue;
                }
                std::printf("\n[%d] "
                            "alpha = %lld "
                            "x = %lld "
                            "point = %lld "
                            "v = %lld",
                            __LINE__,                                   //
                            (long long)alpha[i],                        //
                            (long long)x[i],                            //
                            (long long)mod_bits(points[j], bitWidthIn), //
                            (long long)v);
                std::exit(-1);
            }
        }

        std::puts("  pass");
    }
};

#endif