#ifndef TEST_cuda_GROTTO_TEST_GROTTO_LUT_CUH
#define TEST_cuda_GROTTO_TEST_GROTTO_LUT_CUH

#include <FastFss/cuda/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"
#include "../utils.cuh"

template <typename GroupElement>
class TestGrottoLut
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
    static void run(std::size_t bitWidthIn,
                    std::size_t elementNum,
                    MT19937Rng &rng)
    {
        using namespace FastFss::cuda;

        std::printf("[cuda test GrottoLutEval] "
                    "elementSize = %3d "
                    "bitWidthIn = %3d "
                    "elementNum = %5d ",
                    (int)sizeof(GroupElement), //
                    (int)bitWidthIn,           //
                    (int)elementNum            //
        );

        std::size_t alphaDataSize   = sizeof(GroupElement) * elementNum;
        std::size_t maskedXDataSize = sizeof(GroupElement) * elementNum;
        std::size_t seedDataSize0   = 16 * elementNum;
        std::size_t seedDataSize1   = 16 * elementNum;
        std::size_t lutsDataSize = sizeof(GroupElement) * (1ULL << bitWidthIn);
        std::size_t grottoKeyDataSize;
        FastFss_cuda_grottoGetKeyDataSize(&grottoKeyDataSize, bitWidthIn,
                                          sizeof(GroupElement), elementNum);
        std::size_t cacheDataSize;
        FastFss_cuda_grottoGetCacheDataSize(&cacheDataSize, bitWidthIn,
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

        std::vector<GroupElement> luts(1ULL << bitWidthIn);
        for (std::size_t i = 0; i < luts.size(); i++)
        {
            luts[i] = mod_bits(luts.size() - i - 1, bitWidthIn);
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
            x[i] = mod_bits(x[i], bitWidthIn);
            if (rng.rand<int>() & 1)
            {
                x[i] = 0;
            }
            maskedX[i] = x[i] + alpha[i];
            maskedX[i] = mod_bits(maskedX[i], bitWidthIn);
        }
        auto dGrottoKey = make_unique_gpu_ptr(grottoKeyDataSize);
        {
            auto dAlpha = make_unique_gpu_ptr(alphaDataSize);
            auto dSeed0 = make_unique_gpu_ptr(seedDataSize0);
            auto dSeed1 = make_unique_gpu_ptr(seedDataSize1);

            memcpy_cpu2gpu(dAlpha.get(), alpha.get(), alphaDataSize);
            memcpy_cpu2gpu(dSeed0.get(), seed0.get(), seedDataSize0);
            memcpy_cpu2gpu(dSeed1.get(), seed1.get(), seedDataSize1);

            int ret1 = FastFss_cuda_grottoKeyGen( //
                dGrottoKey.get(),                 //
                grottoKeyDataSize,                //
                dAlpha.get(),                     ///
                alphaDataSize,                    //
                dSeed0.get(),                     //
                seedDataSize0,                    //
                dSeed1.get(),                     //
                seedDataSize1,                    //
                bitWidthIn,                       //
                sizeof(GroupElement),             //
                elementNum, nullptr);
            if (ret1 != 0)
            {
                std::printf("\n[%d] err. FastFss_cuda_grottoKeyGen ret = %d\n",
                            __LINE__, ret1);
                std::exit(-1);
            }
            memcpy_gpu2cpu(grottoKey.get(), dGrottoKey.get(),
                           grottoKeyDataSize);
        }
        auto dCache = make_unique_gpu_ptr(cacheDataSize);
        auto dLuts  = make_unique_gpu_ptr(lutsDataSize);
        memcpy_cpu2gpu(dLuts.get(), luts.data(), lutsDataSize);
        {
            auto dSharedOutE0 = make_unique_gpu_ptr(maskedXDataSize);
            auto dSharedOutT0 = make_unique_gpu_ptr(maskedXDataSize);
            auto dMaskedX     = make_unique_gpu_ptr(maskedXDataSize);
            auto dSeed0       = make_unique_gpu_ptr(seedDataSize0);

            memcpy_cpu2gpu(dMaskedX.get(), maskedX.get(), maskedXDataSize);
            memcpy_cpu2gpu(dSharedOutE0.get(), sharedOutE0.get(),
                           sizeof(GroupElement) * elementNum);
            memcpy_cpu2gpu(dSharedOutT0.get(), sharedOutT0.get(),
                           sizeof(GroupElement) * elementNum);
            memcpy_cpu2gpu(dSeed0.get(), seed0.get(), seedDataSize0);

            int ret2 = FastFss_cuda_grottoLutEval( //
                dSharedOutE0.get(),                //
                dSharedOutT0.get(),                //
                dMaskedX.get(),                    //
                maskedXDataSize,                   //
                dGrottoKey.get(),                  //
                grottoKeyDataSize,                 //
                dSeed0.get(),                      //
                seedDataSize0,                     //
                0,                                 //
                dLuts.get(),                       //
                lutsDataSize,                      //
                bitWidthIn,                        //
                sizeof(GroupElement) * 8,          //
                sizeof(GroupElement),              //
                elementNum,                        //
                dCache.get(),                      //
                cacheDataSize,                     //
                nullptr);
            if (ret2 != 0)
            {
                std::printf("\n[%d] err. FastFss_cuda_grottoLutEval ret = %d\n",
                            __LINE__, ret2);
                std::exit(-1);
            }
            memcpy_gpu2cpu(sharedOutE0.get(), dSharedOutE0.get(),
                           sizeof(GroupElement) * elementNum);
            memcpy_gpu2cpu(sharedOutT0.get(), dSharedOutT0.get(),
                           sizeof(GroupElement) * elementNum);
        }

        {
            auto dSharedOutE1 = make_unique_gpu_ptr(maskedXDataSize);
            auto dSharedOutT1 = make_unique_gpu_ptr(maskedXDataSize);
            auto dMaskedX     = make_unique_gpu_ptr(maskedXDataSize);
            auto dSeed1       = make_unique_gpu_ptr(seedDataSize1);

            memcpy_cpu2gpu(dMaskedX.get(), maskedX.get(), maskedXDataSize);
            memcpy_cpu2gpu(dSharedOutE1.get(), sharedOutE1.get(),
                           sizeof(GroupElement) * elementNum);
            memcpy_cpu2gpu(dSharedOutT1.get(), sharedOutT1.get(),
                           sizeof(GroupElement) * elementNum);
            memcpy_cpu2gpu(dSeed1.get(), seed1.get(), seedDataSize1);

            int ret3 = FastFss_cuda_grottoLutEval( //
                dSharedOutE1.get(),                //
                dSharedOutT1.get(),                //
                dMaskedX.get(),                    //
                maskedXDataSize,                   //
                dGrottoKey.get(),                  //
                grottoKeyDataSize,                 //
                dSeed1.get(),                      //
                seedDataSize1,                     //
                1,                                 //
                dLuts.get(),                       //
                lutsDataSize,                      //
                bitWidthIn,                        //
                sizeof(GroupElement) * 8,          //
                sizeof(GroupElement),              //
                elementNum,                        //
                dCache.get(),                      //
                cacheDataSize,                     //
                nullptr);
            if (ret3 != 0)
            {
                std::printf("\n[%d] err. FastFss_cuda_grottoLutEval ret = %d\n",
                            __LINE__, ret3);
                std::exit(-1);
            }
            memcpy_gpu2cpu(sharedOutE1.get(), dSharedOutE1.get(),
                           sizeof(GroupElement) * elementNum);
            memcpy_gpu2cpu(sharedOutT1.get(), dSharedOutT1.get(),
                           sizeof(GroupElement) * elementNum);
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
    }
};

#endif