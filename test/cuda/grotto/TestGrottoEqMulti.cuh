#ifndef TEST_CUDA_GROTTO_TEST_GROTTO_EQ_MULTI_CUH
#define TEST_CUDA_GROTTO_TEST_GROTTO_EQ_MULTI_CUH

#include <FastFss/cuda/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"
#include "../utils.cuh"

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
        using namespace FastFss::cuda;

        std::printf("[cuda test GrottoEqMulti] "
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
        std::size_t seed0DataSize   = 16 * elementNum;
        std::size_t seed1DataSize   = 16 * elementNum;
        std::size_t grottoKeyDataSize;
        FastFss_cuda_grottoGetKeyDataSize(&grottoKeyDataSize, bitWidthIn,
                                          sizeof(GroupElement), elementNum);
        std::size_t pointDataSize = sizeof(GroupElement) * points.size();

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

        rng.gen(seed0.get(), seed0DataSize);
        rng.gen(seed1.get(), seed1DataSize);

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
            auto dGrottoKey = make_unique_gpu_ptr(grottoKeyDataSize);
            auto dAlpha     = make_unique_gpu_ptr(alphaDataSize);
            auto dSeed0     = make_unique_gpu_ptr(seed0DataSize);
            auto dSeed1     = make_unique_gpu_ptr(seed1DataSize);

            memcpy_cpu2gpu(dAlpha.get(), alpha.get(), alphaDataSize);
            memcpy_cpu2gpu(dSeed0.get(), seed0.get(), seed0DataSize);
            memcpy_cpu2gpu(dSeed1.get(), seed1.get(), seed1DataSize);

            int ret1 = FastFss_cuda_grottoKeyGen( //
                dGrottoKey.get(),                 //
                grottoKeyDataSize,                //
                dAlpha.get(),                     ///
                alphaDataSize,                    //
                dSeed0.get(),                     //
                seed0DataSize,                    //
                dSeed1.get(),                     //
                seed1DataSize,                    //
                bitWidthIn,                       //
                sizeof(GroupElement),             //
                elementNum,                       //
                nullptr);
            if (ret1 != 0)
            {
                std::printf("\n[%d] err. FastFss_cuda_grottoKeyGen ret = %d\n",
                            __LINE__, ret1);
                std::exit(-1);
            }

            memcpy_gpu2cpu(grottoKey.get(), dGrottoKey.get(),
                           grottoKeyDataSize);
        }

        {
            auto dSharedOut0 = make_unique_gpu_ptr(outDataSize);
            auto dMaskedX    = make_unique_gpu_ptr(maskedXDataSize);
            auto dGrottoKey  = make_unique_gpu_ptr(grottoKeyDataSize);
            auto dSeed0      = make_unique_gpu_ptr(seed0DataSize);
            auto dPoints     = make_unique_gpu_ptr(pointDataSize);

            memcpy_cpu2gpu(dMaskedX.get(), maskedX.get(), maskedXDataSize);
            memcpy_cpu2gpu(dGrottoKey.get(), grottoKey.get(),
                           grottoKeyDataSize);
            memcpy_cpu2gpu(dSeed0.get(), seed0.get(), seed0DataSize);
            memcpy_cpu2gpu(dPoints.get(), points.data(), pointDataSize);

            int ret2 = FastFss_cuda_grottoEqMultiEval( //
                dSharedOut0.get(),                     //
                outDataSize,                           //
                dMaskedX.get(),                        //
                maskedXDataSize,                       //
                dGrottoKey.get(),                      //
                grottoKeyDataSize,                     //
                dSeed0.get(),                          //
                seed0DataSize,                         //
                0,                                     //
                dPoints.get(),                         //
                pointDataSize,                         //
                bitWidthIn,                            //
                sizeof(GroupElement),                  //
                elementNum,                            //
                nullptr,                               //
                0, nullptr);
            if (ret2 != 0)
            {
                std::printf(
                    "\n[%d] err. FastFss_cuda_grottoEqMultiEval ret = %d\n",
                    __LINE__, ret2);
                std::exit(-1);
            }

            memcpy_gpu2cpu(sharedOut0.get(), dSharedOut0.get(), outDataSize);
        }

        {
            auto dSharedOut1 = make_unique_gpu_ptr(outDataSize);
            auto dMaskedX    = make_unique_gpu_ptr(maskedXDataSize);
            auto dGrottoKey  = make_unique_gpu_ptr(grottoKeyDataSize);
            auto dSeed1      = make_unique_gpu_ptr(seed1DataSize);
            auto dPoints     = make_unique_gpu_ptr(pointDataSize);

            memcpy_cpu2gpu(dMaskedX.get(), maskedX.get(), maskedXDataSize);
            memcpy_cpu2gpu(dGrottoKey.get(), grottoKey.get(),
                           grottoKeyDataSize);
            memcpy_cpu2gpu(dSeed1.get(), seed1.get(), seed1DataSize);
            memcpy_cpu2gpu(dPoints.get(), points.data(), pointDataSize);

            int ret3 = FastFss_cuda_grottoEqMultiEval( //
                dSharedOut1.get(),                     //
                outDataSize,                           //
                dMaskedX.get(),                        //
                maskedXDataSize,                       //
                dGrottoKey.get(),                      //
                grottoKeyDataSize,                     //
                dSeed1.get(),                          //
                seed1DataSize,                         //
                1,                                     //
                dPoints.get(),                         //
                pointDataSize,                         //
                bitWidthIn,                            //
                sizeof(GroupElement),                  //
                elementNum,                            //
                nullptr,                               //
                0, nullptr);
            if (ret3 != 0)
            {
                std::printf(
                    "\n[%d] err. FastFss_cuda_grottoEqMultiEval ret = %d\n",
                    __LINE__, ret3);
                std::exit(-1);
            }

            memcpy_gpu2cpu(sharedOut1.get(), dSharedOut1.get(), outDataSize);
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