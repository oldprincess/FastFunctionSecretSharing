// clang-format off
// nvcc -I include src/cuda/config.cpp src/cuda/grotto.cu test/cuda/grotto.cu -o cuda_grotto.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "grotto/TestGrottoEqMulti.cuh"
#include "grotto/TestGrottoLut.cuh"
#include "grotto/TestGrottoLut_ex.cuh"
#include "mt19937.hpp"
#include "uint128_t.h"
#include "utils.cuh"

MT19937Rng rng;

using namespace FastFss::cuda;

#define LOG(fmt, ...)                                                 \
    std::fprintf(stderr, "[FastFss] " fmt ". %s:%d\n", ##__VA_ARGS__, \
                 __FILE__, __LINE__)

#define CHECK(exp)                    \
    [&] {                             \
        cudaDeviceSynchronize();      \
        auto the_ret = exp;           \
        if (the_ret)                  \
        {                             \
            LOG("ret = %d", the_ret); \
            std::exit(-1);            \
        }                             \
    }()

template <typename GroupElement>
constexpr GroupElement mod_bits(GroupElement x, int bitWidth) noexcept
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

template <typename GroupElement>
class TestGrottoEq
{
public:
    static void run(std::size_t bitWidthIn, std::size_t elementNum)
    {
        std::printf("[cuda test GrottoEq] elementSize = %2d bitWidthIn = %3d "
                    "elementNum = %5d",
                    (int)(sizeof(GroupElement)), (int)bitWidthIn,
                    (int)elementNum);

        std::unique_ptr<GroupElement[]> alpha(new GroupElement[elementNum]);
        std::size_t alphaDataSize = sizeof(GroupElement) * elementNum;

        for (std::size_t i = 0; i < elementNum; i++)
        {
            alpha[i] = rng.rand<GroupElement>();
            alpha[i] = mod_bits<GroupElement>(alpha[i], bitWidthIn);
        }

        std::unique_ptr<std::uint8_t[]> seed0(new uint8_t[16 * elementNum]);
        std::unique_ptr<std::uint8_t[]> seed1(new uint8_t[16 * elementNum]);
        std::size_t                     seedDataSize0 = 16 * elementNum;
        std::size_t                     seedDataSize1 = 16 * elementNum;

        rng.gen(seed0.get(), seedDataSize0);
        rng.gen(seed1.get(), seedDataSize1);

        std::unique_ptr<GroupElement[]> x(new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> y(new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> maskedX(new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> sharedOut0(
            new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> sharedOut1(
            new GroupElement[elementNum]);
        std::size_t maskedXDataSize = sizeof(GroupElement) * elementNum;
        for (std::size_t i = 0; i < elementNum; i++)
        {
            x[i] = rng.rand<GroupElement>();
            x[i] = mod_bits<GroupElement>(x[i], bitWidthIn);
            if (rng.rand<int>() & 1)
            {
                x[i] = 0;
            }
            maskedX[i] = x[i] + alpha[i];
            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
        }

        int         ret;
        void*       grottoKey = nullptr;
        std::size_t grottoKeyDataSize;
        ret = FastFss_cuda_grottoGetKeyDataSize(
            &grottoKeyDataSize, bitWidthIn, sizeof(GroupElement), elementNum);
        CHECK(ret);
        grottoKey = malloc_gpu(grottoKeyDataSize);

        {
            void* deviceAlpha = malloc_gpu(alphaDataSize);
            void* deviceSeed0 = malloc_gpu(seedDataSize0);
            void* deviceSeed1 = malloc_gpu(seedDataSize1);
            memcpy_cpu2gpu(deviceAlpha, alpha.get(), alphaDataSize);
            memcpy_cpu2gpu(deviceSeed0, seed0.get(), seedDataSize0);
            memcpy_cpu2gpu(deviceSeed1, seed1.get(), seedDataSize0);

            int ret1 = FastFss_cuda_grottoKeyGen(
                grottoKey, grottoKeyDataSize, deviceAlpha, alphaDataSize,
                deviceSeed0, seedDataSize0, deviceSeed1, seedDataSize1,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr);

            free_gpu(deviceAlpha);
            free_gpu(deviceSeed0);
            free_gpu(deviceSeed1);

            CHECK(ret1);
        }

        {
            void* deviceSeed0      = malloc_gpu(seedDataSize0);
            void* deviceMaskedX    = malloc_gpu(maskedXDataSize);
            void* deviceSharedOut0 = malloc_gpu(maskedXDataSize);
            memcpy_cpu2gpu(deviceMaskedX, maskedX.get(), maskedXDataSize);
            memcpy_cpu2gpu(deviceSeed0, seed0.get(), seedDataSize0);

            int ret2 = FastFss_cuda_grottoEqEval(
                deviceSharedOut0, deviceMaskedX, maskedXDataSize, grottoKey,
                grottoKeyDataSize, deviceSeed0, seedDataSize0, 0, bitWidthIn,
                sizeof(GroupElement), elementNum, nullptr, 0, nullptr);

            memcpy_gpu2cpu(sharedOut0.get(), deviceSharedOut0, maskedXDataSize);
            free_gpu(deviceMaskedX);
            free_gpu(deviceSharedOut0);
            free_gpu(deviceSeed0);

            CHECK(ret2);
        }

        {
            void* deviceSeed1      = malloc_gpu(seedDataSize1);
            void* deviceMaskedX    = malloc_gpu(maskedXDataSize);
            void* deviceSharedOut1 = malloc_gpu(maskedXDataSize);
            memcpy_cpu2gpu(deviceMaskedX, maskedX.get(), maskedXDataSize);
            memcpy_cpu2gpu(deviceSeed1, seed1.get(), seedDataSize1);

            int ret3 = FastFss_cuda_grottoEqEval(
                deviceSharedOut1, deviceMaskedX, maskedXDataSize, grottoKey,
                grottoKeyDataSize, deviceSeed1, seedDataSize1, 1, bitWidthIn,
                sizeof(GroupElement), elementNum, nullptr, 0, nullptr);

            memcpy_gpu2cpu(sharedOut1.get(), deviceSharedOut1, maskedXDataSize);
            free_gpu(deviceMaskedX);
            free_gpu(deviceSharedOut1);
            free_gpu(deviceSeed1);

            CHECK(ret3);
        }

        for (int i = 0; i < elementNum; i++)
        {
            GroupElement v = (sharedOut0[i] + sharedOut1[i]) & 1;

            bool cmp0 = ((x[i] == 0) && (v == 1));
            bool cmp1 = ((x[i] != 0) && (v == 0));
            if (!(cmp0 || cmp1))
            {
                std::printf("\n[%d] alpha = %lld ", __LINE__,
                            (long long)alpha[i]);
                std::printf("maskedX = %lld v = %lld", (long long)maskedX[i],
                            (long long)v);
                std::exit(-1);
            }
        }
        free_gpu(grottoKey);

        std::puts("  pass");
    }
};

template <typename GroupElement>
class TestGrotto
{
public:
    static void run(std::size_t bitWidthIn, std::size_t elementNum)
    {
        std::printf("[cuda test GrottoEval] elementSize = %2d bitWidthIn = %3d "
                    "elementNum = %5d",
                    (int)(sizeof(GroupElement)), (int)bitWidthIn,
                    (int)elementNum);

        std::unique_ptr<GroupElement[]> alpha(new GroupElement[elementNum]);
        std::size_t alphaDataSize = sizeof(GroupElement) * elementNum;

        for (std::size_t i = 0; i < elementNum; i++)
        {
            alpha[i] = rng.rand<GroupElement>();
            alpha[i] = mod_bits<GroupElement>(alpha[i], bitWidthIn);
        }

        std::unique_ptr<std::uint8_t[]> seed0(new uint8_t[16 * elementNum]);
        std::unique_ptr<std::uint8_t[]> seed1(new uint8_t[16 * elementNum]);
        std::size_t                     seedDataSize0 = 16 * elementNum;
        std::size_t                     seedDataSize1 = 16 * elementNum;

        rng.gen(seed0.get(), seedDataSize0);
        rng.gen(seed1.get(), seedDataSize1);

        std::unique_ptr<GroupElement[]> x(new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> y(new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> maskedX(new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> sharedOut0(
            new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> sharedOut1(
            new GroupElement[elementNum]);
        std::size_t maskedXDataSize = sizeof(GroupElement) * elementNum;
        for (std::size_t i = 0; i < elementNum; i++)
        {
            x[i]       = rng.rand<GroupElement>();
            x[i]       = mod_bits<GroupElement>(x[i], bitWidthIn);
            maskedX[i] = x[i] + alpha[i];
            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
        }

        int         ret;
        void*       grottoKey = nullptr;
        std::size_t grottoKeyDataSize;
        ret = FastFss_cuda_grottoGetKeyDataSize(
            &grottoKeyDataSize, bitWidthIn, sizeof(GroupElement), elementNum);
        CHECK(ret);
        grottoKey = malloc_gpu(grottoKeyDataSize);

        {
            void* deviceAlpha = malloc_gpu(alphaDataSize);
            void* deviceSeed0 = malloc_gpu(seedDataSize0);
            void* deviceSeed1 = malloc_gpu(seedDataSize1);
            memcpy_cpu2gpu(deviceAlpha, alpha.get(), alphaDataSize);
            memcpy_cpu2gpu(deviceSeed0, seed0.get(), seedDataSize0);
            memcpy_cpu2gpu(deviceSeed1, seed1.get(), seedDataSize0);

            int ret1 = FastFss_cuda_grottoKeyGen(
                grottoKey, grottoKeyDataSize, deviceAlpha, alphaDataSize,
                deviceSeed0, seedDataSize0, deviceSeed1, seedDataSize1,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr);

            free_gpu(deviceAlpha);
            free_gpu(deviceSeed0);
            free_gpu(deviceSeed1);

            if (ret1 != 0)
            {
                std::printf("\n[%d] err. FastFss_cuda_grottoKeyGen ret = %d\n",
                            __LINE__, ret1);
                return;
            }
        }

        {
            void* deviceMaskedX    = malloc_gpu(maskedXDataSize);
            void* deviceSharedOut0 = malloc_gpu(maskedXDataSize);
            void* deviceSeed0      = malloc_gpu(seedDataSize0);
            memcpy_cpu2gpu(deviceMaskedX, maskedX.get(), maskedXDataSize);
            memcpy_cpu2gpu(deviceSeed0, seed0.get(), seedDataSize0);

            int ret2 = FastFss_cuda_grottoEval(
                deviceSharedOut0, deviceMaskedX, maskedXDataSize, grottoKey,
                grottoKeyDataSize, deviceSeed0, seedDataSize0, false, 0,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr, 0,
                nullptr);

            memcpy_gpu2cpu(sharedOut0.get(), deviceSharedOut0, maskedXDataSize);
            free_gpu(deviceMaskedX);
            free_gpu(deviceSharedOut0);
            free_gpu(deviceSeed0);

            if (ret2 != 0)
            {
                std::printf("\n[%d] err. FastFss_cuda_grottoEval ret = %d\n",
                            __LINE__, ret2);
                cudaFree(grottoKey);
                return;
            }
        }

        {
            void* deviceSeed1      = malloc_gpu(seedDataSize1);
            void* deviceMaskedX    = malloc_gpu(maskedXDataSize);
            void* deviceSharedOut1 = malloc_gpu(maskedXDataSize);
            memcpy_cpu2gpu(deviceMaskedX, maskedX.get(), maskedXDataSize);
            memcpy_cpu2gpu(deviceSeed1, seed1.get(), seedDataSize1);

            int ret3 = FastFss_cuda_grottoEval(
                deviceSharedOut1, deviceMaskedX, maskedXDataSize, grottoKey,
                grottoKeyDataSize, deviceSeed1, seedDataSize1, false, 1,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr, 0,
                nullptr);

            memcpy_gpu2cpu(sharedOut1.get(), deviceSharedOut1, maskedXDataSize);
            free_gpu(deviceMaskedX);
            free_gpu(deviceSharedOut1);
            free_gpu(deviceSeed1);

            if (ret3 != 0)
            {
                std::printf("\n[%d] err. FastFss_cuda_grottoEval ret = %d\n",
                            __LINE__, ret3);
                cudaFree(grottoKey);
                return;
            }
        }

        for (int i = 0; i < elementNum; i++)
        {
            GroupElement v = (sharedOut0[i] + sharedOut1[i]) & 1;

            bool cmp0 = ((maskedX[i] < alpha[i]) && (v == 1));
            bool cmp1 = ((maskedX[i] >= alpha[i]) && (v == 0));
            if (!(cmp0 || cmp1))
            {
                std::printf("\n%d", i);
                std::printf("\n[%d] alpha = %lld ", __LINE__,
                            (long long)alpha[i]);
                std::printf("maskedX = %lld v = %lld", (long long)maskedX[i],
                            (long long)v);
                std::exit(-1);
            }
        }

        {
            void* deviceMaskedX    = malloc_gpu(maskedXDataSize);
            void* deviceSharedOut0 = malloc_gpu(maskedXDataSize);
            void* deviceSeed0      = malloc_gpu(seedDataSize0);
            memcpy_cpu2gpu(deviceMaskedX, maskedX.get(), maskedXDataSize);
            memcpy_cpu2gpu(deviceSeed0, seed0.get(), seedDataSize0);

            int ret2 = FastFss_cuda_grottoEval(
                deviceSharedOut0, deviceMaskedX, maskedXDataSize, grottoKey,
                grottoKeyDataSize, deviceSeed0, seedDataSize0, true, 0,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr, 0,
                nullptr);

            memcpy_gpu2cpu(sharedOut0.get(), deviceSharedOut0, maskedXDataSize);
            free_gpu(deviceMaskedX);
            free_gpu(deviceSharedOut0);
            free_gpu(deviceSeed0);
            if (ret2 != 0)
            {
                std::printf("\n[%d] err. FastFss_cuda_grottoEval ret = %d\n",
                            __LINE__, ret2);
                cudaFree(grottoKey);
                return;
            }
        }

        {
            void* deviceSeed1      = malloc_gpu(seedDataSize1);
            void* deviceMaskedX    = malloc_gpu(maskedXDataSize);
            void* deviceSharedOut1 = malloc_gpu(maskedXDataSize);
            memcpy_cpu2gpu(deviceMaskedX, maskedX.get(), maskedXDataSize);
            memcpy_cpu2gpu(deviceSeed1, seed1.get(), seedDataSize1);

            int ret3 = FastFss_cuda_grottoEval(
                deviceSharedOut1, deviceMaskedX, maskedXDataSize, grottoKey,
                grottoKeyDataSize, deviceSeed1, seedDataSize1, true, 1,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr, 0,
                nullptr);
            memcpy_gpu2cpu(sharedOut1.get(), deviceSharedOut1, maskedXDataSize);
            free_gpu(deviceMaskedX);
            free_gpu(deviceSharedOut1);
            free_gpu(deviceSeed1);

            if (ret3 != 0)
            {
                std::printf("\n[%d] err. FastFss_cuda_grottoEval ret = %d\n",
                            __LINE__, ret3);
                std::free(grottoKey);
                return;
            }
        }

        for (int i = 0; i < elementNum; i++)
        {
            GroupElement v = (sharedOut0[i] + sharedOut1[i]) & 1;

            bool cmp0 = ((maskedX[i] <= alpha[i]) && (v == 1));
            bool cmp1 = ((maskedX[i] > alpha[i]) && (v == 0));
            if (!(cmp0 || cmp1))
            {
                std::printf("\n%d", i);
                std::printf("\n[%d] alpha = %lld ", __LINE__,
                            (long long)alpha[i]);
                std::printf("maskedX = %lld v = %lld", (long long)maskedX[i],
                            (long long)v);
                std::exit(-1);
            }
        }

        cudaFree(grottoKey);

        std::puts("  pass");
    }
};

template <typename GroupElement>
class TestGrottoMIC
{
public:
    static void run(std::size_t                     bitWidthIn,
                    std::size_t                     elementNum,
                    const std::vector<GroupElement> leftBoundary,
                    const std::vector<GroupElement> rightBoundary)
    {
        std::printf("[cuda test GrottoMIC] elementSize = %2d bitWidthIn = %3d "
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

        std::size_t alphaDataSize         = elementNum * sizeof(GroupElement);
        std::size_t seedDataSize0         = elementNum * 16;
        std::size_t seedDataSize1         = elementNum * 16;
        std::size_t maskedXDataSize       = elementNum * sizeof(GroupElement);
        std::size_t leftBoundaryDataSize  = intervalNum * sizeof(GroupElement);
        std::size_t rightBoundaryDataSize = intervalNum * sizeof(GroupElement);

        int         ret;
        void*       grottoMICKey         = nullptr;
        std::size_t grottoMICKeyDataSize = 0;
        void*       grottoCache          = nullptr;
        std::size_t cacheDataSize        = 0;
        ret =
            FastFss_cuda_grottoGetKeyDataSize(&grottoMICKeyDataSize, bitWidthIn,
                                              sizeof(GroupElement), elementNum);
        CHECK(ret);
        ret = FastFss_cuda_grottoGetCacheDataSize(
            &cacheDataSize, bitWidthIn, sizeof(GroupElement), elementNum);
        CHECK(ret);
        grottoMICKey = malloc_gpu(grottoMICKeyDataSize);
        grottoCache  = malloc_gpu(cacheDataSize);

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            x[i]       = rng.rand<GroupElement>();
            alpha[i]   = rng.rand<GroupElement>();
            maskedX[i] = x[i] + alpha[i];

            x[i]       = mod_bits<GroupElement>(x[i], bitWidthIn);
            alpha[i]   = mod_bits<GroupElement>(alpha[i], bitWidthIn);
            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
        }

        rng.gen(seed0.data(), seed0.size());
        rng.gen(seed1.data(), seed1.size());

        {
            void* deviceAlpha = malloc_gpu(alphaDataSize);
            void* deviceSeed0 = malloc_gpu(seedDataSize0);
            void* deviceSeed1 = malloc_gpu(seedDataSize1);
            memcpy_cpu2gpu(deviceAlpha, alpha.data(), alphaDataSize);
            memcpy_cpu2gpu(deviceSeed0, seed0.data(), seedDataSize0);
            memcpy_cpu2gpu(deviceSeed1, seed1.data(), seedDataSize1);

            int ret1 = FastFss_cuda_grottoKeyGen(
                grottoMICKey, grottoMICKeyDataSize, deviceAlpha, alphaDataSize,
                deviceSeed0, seedDataSize0, deviceSeed1, seedDataSize1,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr);
            CHECK(ret1);

            free_gpu(deviceAlpha);
            free_gpu(deviceSeed0);
            free_gpu(deviceSeed1);
        }

        {
            std::size_t sharedOutDataSize =
                elementNum * intervalNum * sizeof(GroupElement);

            void* deviceLeftBoundary  = malloc_gpu(leftBoundaryDataSize);
            void* deviceRightBoundary = malloc_gpu(rightBoundaryDataSize);
            void* deviceSeed0         = malloc_gpu(seedDataSize0);
            void* deviceMaskedX       = malloc_gpu(maskedXDataSize);
            void* deviceSharedOut0    = malloc_gpu(sharedOutDataSize);
            memcpy_cpu2gpu(deviceMaskedX, maskedX.data(), maskedXDataSize);
            memcpy_cpu2gpu(deviceSeed0, seed0.data(), seedDataSize0);
            memcpy_cpu2gpu(deviceLeftBoundary, leftBoundary.data(),
                           leftBoundaryDataSize);
            memcpy_cpu2gpu(deviceRightBoundary, rightBoundary.data(),
                           rightBoundaryDataSize);

            int ret1 = FastFss_cuda_grottoMICEval(
                deviceSharedOut0,                             //
                sharedOutDataSize,                            //
                deviceMaskedX,                                //
                maskedXDataSize,                              //
                grottoMICKey,                                 //
                grottoMICKeyDataSize,                         //
                deviceSeed0,                                  //
                seedDataSize0,                                //
                0,                                            //
                deviceLeftBoundary,                           //
                leftBoundaryDataSize,                         //
                deviceRightBoundary,                          //
                rightBoundaryDataSize,                        //
                bitWidthIn, sizeof(GroupElement), elementNum, //
                nullptr, 0, nullptr);
            CHECK(ret1);

            memcpy_gpu2cpu(sharedOut0.data(), deviceSharedOut0,
                           sharedOutDataSize);

            free_gpu(deviceLeftBoundary);
            free_gpu(deviceRightBoundary);
            free_gpu(deviceSeed0);
            free_gpu(deviceMaskedX);
            free_gpu(deviceSharedOut0);
        }

        {
            std::size_t sharedOutDataSize =
                elementNum * intervalNum * sizeof(GroupElement);

            void* deviceLeftBoundary  = malloc_gpu(leftBoundaryDataSize);
            void* deviceRightBoundary = malloc_gpu(rightBoundaryDataSize);
            void* deviceSeed1         = malloc_gpu(seedDataSize1);
            void* deviceMaskedX       = malloc_gpu(maskedXDataSize);
            void* deviceSharedOut1    = malloc_gpu(sharedOutDataSize);
            memcpy_cpu2gpu(deviceMaskedX, maskedX.data(), maskedXDataSize);
            memcpy_cpu2gpu(deviceSeed1, seed1.data(), seedDataSize1);
            memcpy_cpu2gpu(deviceLeftBoundary, leftBoundary.data(),
                           leftBoundaryDataSize);
            memcpy_cpu2gpu(deviceRightBoundary, rightBoundary.data(),
                           rightBoundaryDataSize);

            int ret2 = FastFss_cuda_grottoMICEval(
                deviceSharedOut1,                             //
                sharedOutDataSize,                            //
                deviceMaskedX,                                //
                maskedXDataSize,                              //
                grottoMICKey,                                 //
                grottoMICKeyDataSize,                         //
                deviceSeed1,                                  //
                seedDataSize1,                                //
                1,                                            //
                deviceLeftBoundary,                           //
                leftBoundaryDataSize,                         //
                deviceRightBoundary,                          //
                rightBoundaryDataSize,                        //
                bitWidthIn, sizeof(GroupElement), elementNum, //
                grottoCache, cacheDataSize, nullptr);
            CHECK(ret2);

            memcpy_gpu2cpu(sharedOut1.data(), deviceSharedOut1,
                           sharedOutDataSize);

            free_gpu(deviceLeftBoundary);
            free_gpu(deviceRightBoundary);
            free_gpu(deviceSeed1);
            free_gpu(deviceMaskedX);
            free_gpu(deviceSharedOut1);
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

        cudaFree(grottoMICKey);
        free_gpu(grottoCache);
        std::puts("  pass");
    }
};

int main()
{
    rng.reseed(7);
    {
        // uint8
        TestGrottoEq<std::uint8_t>::run(7, 1024 - 1);
        TestGrottoEq<std::uint8_t>::run(8, 1024 - 1);
        // uint16
        TestGrottoEq<std::uint16_t>::run(12, 1024 - 1);
        TestGrottoEq<std::uint16_t>::run(16, 1024 - 1);
        // uint32
        TestGrottoEq<std::uint32_t>::run(18, 1024 - 1);
        TestGrottoEq<std::uint32_t>::run(18, 1024 - 1);
        // uint64
        TestGrottoEq<std::uint64_t>::run(63, 1024 - 1);
        // uint128
        TestGrottoEq<uint128_t>::run(127, 1024 - 1);
        TestGrottoEq<uint128_t>::run(128, 1024 - 1);
    }
    {
        // uint8
        TestGrotto<std::uint8_t>::run(7, 1024 - 1);
        TestGrotto<std::uint8_t>::run(8, 1024 - 1);
        // uint16
        TestGrotto<std::uint16_t>::run(12, 1024 - 1);
        TestGrotto<std::uint16_t>::run(16, 1024 - 1);
        // uint32
        TestGrotto<std::uint32_t>::run(18, 1024 - 1);
        TestGrotto<std::uint32_t>::run(18, 1024 - 1);
        // uint64
        TestGrotto<std::uint64_t>::run(63, 1024 - 1);
        // uint128
        TestGrotto<uint128_t>::run(127, 1024 - 1);
        TestGrotto<uint128_t>::run(128, 1024 - 1);
    }
    {
        constexpr int elementNum = 1024 - 1;
        TestGrottoMIC<std::uint8_t>::run(8, elementNum, {1, 2, 3, 4},
                                         {2, 3, 4, 5});
        TestGrottoMIC<std::uint16_t>::run(12, elementNum,               //
                                          {10, 20, 30, 40, 50, 60, 70}, //
                                          {20, 30, 40, 50, 60, 70, 80}  //
        );
        TestGrottoMIC<std::uint16_t>::run(16, elementNum,               //
                                          {10, 20, 30, 40, 50, 60, 70}, //
                                          {20, 30, 40, 50, 60, 70, 80}  //
        );
        TestGrottoMIC<std::uint32_t>::run(
            24, elementNum,                      //
            {100, 200, 300, 400, 500, 600, 700}, //
            {200, 300, 400, 500, 600, 700, 800}  //
        );
        TestGrottoMIC<std::uint32_t>::run(
            32, elementNum,                      //
            {100, 200, 300, 400, 500, 600, 700}, //
            {200, 300, 400, 500, 600, 700, 800}  //
        );

        TestGrottoMIC<std::uint64_t>::run(
            32, elementNum,                             //
            {1000, 2000, 3000, 4000, 5000, 6000, 7000}, //
            {2000, 3000, 4000, 5000, 6000, 7000, 8000}  //
        );

        TestGrottoMIC<std::uint64_t>::run(
            48, elementNum,                                    //
            {10000, 20000, 30000, 40000, 50000, 60000, 70000}, //
            {20000, 30000, 40000, 50000, 60000, 70000, 80000}  //
        );
    }

    {
        constexpr int elementNum = 1024 - 1;
        TestGrottoEqMulti<std::uint8_t>::run(                               //
            8, elementNum, {1, 2, 3, 4}, rng                                //
        );                                                                  //
        TestGrottoEqMulti<std::uint16_t>::run(                              //
            12, elementNum, {10, 20, 30, 40, 50, 60, 70}, rng               //
        );                                                                  //
        TestGrottoEqMulti<std::uint16_t>::run(                              //
            16, elementNum, {10, 20, 30, 40, 50, 60, 70}, rng               //
        );                                                                  //
        TestGrottoEqMulti<std::uint32_t>::run(                              //
            24, elementNum, {100, 200, 300, 400, 500, 600, 700}, rng        //
        );                                                                  //
        TestGrottoEqMulti<std::uint32_t>::run(                              //
            32, elementNum, {100, 200, 300, 400, 500, 600, 700}, rng        //
        );                                                                  //
        TestGrottoEqMulti<std::uint64_t>::run(                              //
            32, elementNum, {1000, 2000, 3000, 4000, 5000, 6000, 7000}, rng //
        );                                                                  //
        TestGrottoEqMulti<std::uint64_t>::run(                              //
            48, elementNum,                                                 //
            {10000, 20000, 30000, 40000, 50000, 60000, 70000}, rng          //
        );                                                                  //
    }

    {
        TestGrottoLut<std::uint8_t>::run(7, 1024 - 1, rng);
        TestGrottoLut<std::uint8_t>::run(8, 1024 - 1, rng);
        TestGrottoLut<std::uint16_t>::run(11, 1024 - 1, rng);
        TestGrottoLut<std::uint32_t>::run(11, 1024 - 1, rng);
        TestGrottoLut<std::uint64_t>::run(11, 1024 - 1, rng);
        TestGrottoLut<uint128_t>::run(11, 1024 - 1, rng);
    }

    {
        TestGrottoLut_ex<std::uint8_t>::run(7, 7, 1024 - 1, rng);
        TestGrottoLut_ex<std::uint8_t>::run(7, 8, 1024 - 1, rng);
        TestGrottoLut_ex<std::uint16_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLut_ex<std::uint32_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLut_ex<std::uint64_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLut_ex<uint128_t>::run(9, 11, 1024 - 1, rng);

        TestGrottoLut_ex<std::uint64_t>::run(9, 30, 128 * 3072, rng);
    }

    return 0;
}