#ifndef TEST_CUDA_DPF_TEST_DPF_EVAL_CUH
#define TEST_CUDA_DPF_TEST_DPF_EVAL_CUH

#include <FastFss/cuda/dpf.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"
#include "../utils.cuh"

using namespace FastFss::cuda;

template <typename GroupElement>
class TestDpfEval
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
                    std::size_t bitWidthOut,
                    std::size_t elementNum,
                    MT19937Rng &rng)
    {
        std::printf("[cuda test DpfEval] "       //
                    "elementSize = %2d "         //
                    "bitWidthIn  = %3d "         //
                    "bitWidthOut = %3d "         //
                    "elementNum  = %5d .... ",   //
                    (int)(sizeof(GroupElement)), //
                    (int)bitWidthIn,             //
                    (int)bitWidthOut,            //
                    (int)elementNum              //
        );                                       //

        int ret = 0;

        std::size_t keyDataSize       = 0;
        std::size_t cacheDataSize     = 0;
        std::size_t elementSize       = sizeof(GroupElement);
        std::size_t alphaDataSize     = elementNum * elementSize;
        std::size_t betaDataSize      = elementNum * elementSize;
        std::size_t seedDataSize0     = 16 * elementNum;
        std::size_t seedDataSize1     = 16 * elementNum;
        std::size_t maskedXDataSize   = elementNum * elementSize;
        std::size_t sharedOutDataSize = elementNum * elementSize;

        ret = FastFss_cuda_dpfGetKeyDataSize(                              //
            &keyDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum //
        );                                                                 //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }
        ret = FastFss_cuda_dpfGetCacheDataSize(                              //
            &cacheDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum //
        );                                                                   //
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        auto x          = std::make_unique<GroupElement[]>(elementNum);
        auto key        = std::make_unique<std::uint8_t[]>(keyDataSize);
        auto cache      = std::make_unique<std::uint8_t[]>(cacheDataSize);
        auto alpha      = std::make_unique<GroupElement[]>(elementNum);
        auto beta       = std::make_unique<GroupElement[]>(elementNum);
        auto seed0      = std::make_unique<std::uint8_t[]>(seedDataSize0);
        auto seed1      = std::make_unique<std::uint8_t[]>(seedDataSize1);
        auto maskedX    = std::make_unique<GroupElement[]>(elementNum);
        auto sharedOut0 = std::make_unique<GroupElement[]>(elementNum);
        auto sharedOut1 = std::make_unique<GroupElement[]>(elementNum);
        auto sharedOut  = std::make_unique<GroupElement[]>(elementNum);

        rng.gen(x.get(), elementNum * elementSize);
        rng.gen(alpha.get(), alphaDataSize);
        rng.gen(beta.get(), betaDataSize);
        rng.gen(seed0.get(), seedDataSize0);
        rng.gen(seed1.get(), seedDataSize1);

        {
            auto dKey   = make_unique_gpu_ptr(key.get(), keyDataSize);
            auto dAlpha = make_unique_gpu_ptr(alpha.get(), alphaDataSize);
            auto dBeta  = make_unique_gpu_ptr(beta.get(), betaDataSize);
            auto dSeed0 = make_unique_gpu_ptr(seed0.get(), seedDataSize0);
            auto dSeed1 = make_unique_gpu_ptr(seed1.get(), seedDataSize1);

            ret = FastFss_cuda_dpfKeyGen(    //
                dKey.get(), keyDataSize,     //
                dAlpha.get(), alphaDataSize, //
                dBeta.get(), betaDataSize,   //
                dSeed0.get(), seedDataSize0, //
                dSeed1.get(), seedDataSize1, //
                bitWidthIn,                  //
                bitWidthOut,                 //
                elementSize,                 //
                elementNum,                  //
                nullptr                      //
            );                               //
            CUDA_ERR_CHECK({ std::exit(-1); });
            memcpy_gpu2cpu(key.get(), dKey.get(), keyDataSize);
        }
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }
        for (std::size_t i = 0; i < elementNum; i++)
        {
            maskedX[i] = x[i] + alpha[i];
        }

        {
            auto dSharedOut = make_unique_gpu_ptr(sharedOutDataSize);
            auto dMaskedX = make_unique_gpu_ptr(maskedX.get(), maskedXDataSize);
            auto dKey     = make_unique_gpu_ptr(key.get(), keyDataSize);
            auto dSeed    = make_unique_gpu_ptr(seed0.get(), seedDataSize0);
            auto dCache   = make_unique_gpu_ptr(cache.get(), cacheDataSize);

            ret = FastFss_cuda_dpfEval(              //
                dSharedOut.get(), sharedOutDataSize, //
                dMaskedX.get(), maskedXDataSize,     //
                dKey.get(), keyDataSize,             //
                dSeed.get(), seedDataSize0,          //
                0,                                   //
                bitWidthIn,                          //
                bitWidthOut,                         //
                elementSize,                         //
                elementNum,                          //
                dCache.get(), cacheDataSize,         //
                nullptr                              //
            );                                       //
            CUDA_ERR_CHECK({ std::exit(-1); });
            memcpy_gpu2cpu(sharedOut0.get(), dSharedOut.get(),
                           sharedOutDataSize);
        }
        if (ret != 0)
        {
            std::printf("Error. %s:%d\n", __FILE__, __LINE__);
            std::exit(-1);
        }

        {
            auto dSharedOut = make_unique_gpu_ptr(sharedOutDataSize);
            auto dMaskedX = make_unique_gpu_ptr(maskedX.get(), maskedXDataSize);
            auto dKey     = make_unique_gpu_ptr(key.get(), keyDataSize);
            auto dSeed    = make_unique_gpu_ptr(seed1.get(), seedDataSize1);

            ret = FastFss_cuda_dpfEval(              //
                dSharedOut.get(), sharedOutDataSize, //
                dMaskedX.get(), maskedXDataSize,     //
                dKey.get(), keyDataSize,             //
                dSeed.get(), seedDataSize1,          //
                1,                                   //
                bitWidthIn,                          //
                bitWidthOut,                         //
                elementSize,                         //
                elementNum,                          //
                nullptr, 0,                          //
                nullptr                              //
            );                                       //
            CUDA_ERR_CHECK({ std::exit(-1); });
            memcpy_gpu2cpu(sharedOut1.get(), dSharedOut.get(),
                           sharedOutDataSize);
        }
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
            beta[i]      = mod_bits(beta[i], bitWidthOut);
            sharedOut[i] = mod_bits(sharedOut[i], bitWidthOut);
            if (maskedX[i] == alpha[i])
            {
                if (sharedOut[i] != beta[i])
                {
                    std::printf("Error. %s:%d\n", __FILE__, __LINE__);
                    std::exit(-1);
                }
            }
            else
            {
                if (sharedOut[i] != 0)
                {
                    std::printf("Error. %s:%d\n", __FILE__, __LINE__);
                    std::exit(-1);
                }
            }
        }

        std::printf("pass \n");
    }
};

#endif