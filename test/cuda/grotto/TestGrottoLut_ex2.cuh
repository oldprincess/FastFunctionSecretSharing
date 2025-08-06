#ifndef TEST_CUDA_GROTTO_TEST_GROTTO_LUT_EX2_CUH
#define TEST_CUDA_GROTTO_TEST_GROTTO_LUT_EX2_CUH

#include <FastFss/cuda/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"
#include "../utils.cuh"

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
        using namespace FastFss::cuda;

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

        auto cache = make_unique_gpu_ptr(cacheDataSize);

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

        auto d_grottoKey = make_unique_gpu_ptr(grottoKeyDataSize);
        auto d_seed0     = make_unique_gpu_ptr(seedDataSize0);
        auto d_seed1     = make_unique_gpu_ptr(seedDataSize1);
        {
            auto d_alpha = make_unique_gpu_ptr(alphaDataSize);

            memcpy_cpu2gpu(d_alpha.get(), alpha.get(), alphaDataSize);
            memcpy_cpu2gpu(d_seed0.get(), seed0.get(), seedDataSize0);
            memcpy_cpu2gpu(d_seed1.get(), seed1.get(), seedDataSize1);
            cudaDeviceSynchronize();

            auto start_time = high_resolution_clock::now();
            int  ret1       = FastFss_cuda_grottoKeyGen( //
                d_grottoKey.get(),                //
                grottoKeyDataSize,                //
                d_alpha.get(),                    ///
                alphaDataSize,                    //
                d_seed0.get(),                    //
                seedDataSize0,                    //
                d_seed1.get(),                    //
                seedDataSize1,                    //
                bitWidthIn,                       //
                sizeof(GroupElement),             //
                elementNum, nullptr);
            cudaDeviceSynchronize();
            auto stop_time = high_resolution_clock::now();
            if (ret1 != 0)
            {
                std::printf("\n[%d] err. FastFss_cuda_grottoKeyGen ret = %d\n",
                            __LINE__, ret1);
                std::exit(-1);
            }
            memcpy_gpu2cpu(grottoKey.get(), d_grottoKey.get(),
                           grottoKeyDataSize);

            genKeyTimeUs =
                duration_cast<microseconds>(stop_time - start_time).count();
        }

        auto d_maskedX = make_unique_gpu_ptr(maskedXDataSize);
        auto d_luts = make_unique_gpu_ptr(sizeof(GroupElement) * luts.size());
        auto d_points =
            make_unique_gpu_ptr(sizeof(GroupElement) * points.size());

        memcpy_cpu2gpu(d_maskedX.get(), maskedX.get(), maskedXDataSize);
        memcpy_cpu2gpu(d_luts.get(), luts.data(),
                       sizeof(GroupElement) * luts.size());
        memcpy_cpu2gpu(d_points.get(), points.data(),
                       sizeof(GroupElement) * points.size());
        {
            auto d_sharedOutE0 = make_unique_gpu_ptr(maskedXDataSize);
            auto d_sharedOutT0 = make_unique_gpu_ptr(maskedXDataSize);
            cudaDeviceSynchronize();

            auto start_time = high_resolution_clock::now();
            int  ret2       = FastFss_cuda_grottoLutEval_ex2( //
                d_sharedOutE0.get(),                   //
                d_sharedOutT0.get(),                   //
                d_maskedX.get(),                       //
                maskedXDataSize,                       //
                d_grottoKey.get(),                     //
                grottoKeyDataSize,                     //
                d_seed0.get(),                         //
                seedDataSize0,                         //
                0,                                     //
                d_points.get(),                        //
                points.size() * sizeof(GroupElement),  //
                d_luts.get(),                          //
                lutsDataSize,                          //
                bitWidthIn,                            //
                sizeof(GroupElement) * 8,              //
                sizeof(GroupElement),                  //
                elementNum,                            //
                cache.get(),                           //
                cacheDataSize, nullptr);
            cudaDeviceSynchronize();
            auto stop_time = high_resolution_clock::now();
            if (ret2 != 0)
            {
                std::printf(
                    "\n[%d] err. FastFss_cuda_grottoLutEval_ex2 ret = %d\n",
                    __LINE__, ret2);
                std::exit(-1);
            }
            memcpy_gpu2cpu(sharedOutE0.get(), d_sharedOutE0.get(),
                           maskedXDataSize);
            memcpy_gpu2cpu(sharedOutT0.get(), d_sharedOutT0.get(),
                           maskedXDataSize);
            eval1TimeUs =
                duration_cast<microseconds>(stop_time - start_time).count();
        }

        {
            auto d_sharedOutE1 = make_unique_gpu_ptr(maskedXDataSize);
            auto d_sharedOutT1 = make_unique_gpu_ptr(maskedXDataSize);
            cudaDeviceSynchronize();

            auto start_time = high_resolution_clock::now();
            int  ret3       = FastFss_cuda_grottoLutEval_ex2( //
                d_sharedOutE1.get(),                   //
                d_sharedOutT1.get(),                   //
                d_maskedX.get(),                       //
                maskedXDataSize,                       //
                d_grottoKey.get(),                     //
                grottoKeyDataSize,                     //
                d_seed1.get(),                         //
                seedDataSize1,                         //
                1,                                     //
                d_points.get(),                        //
                points.size() * sizeof(GroupElement),  //
                d_luts.get(),                          //
                lutsDataSize,                          //
                bitWidthIn,                            //
                sizeof(GroupElement) * 8,              //
                sizeof(GroupElement),                  //
                elementNum,                            //
                cache.get(),                           //
                cacheDataSize, nullptr);
            cudaDeviceSynchronize();
            auto stop_time = high_resolution_clock::now();
            if (ret3 != 0)
            {
                std::printf(
                    "\n[%d] err. FastFss_cuda_grottoLutEval_ex2 ret = %d\n",
                    __LINE__, ret3);
                std::exit(-1);
            }
            memcpy_gpu2cpu(sharedOutE1.get(), d_sharedOutE1.get(),
                           maskedXDataSize);
            memcpy_gpu2cpu(sharedOutT1.get(), d_sharedOutT1.get(),
                           maskedXDataSize);
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