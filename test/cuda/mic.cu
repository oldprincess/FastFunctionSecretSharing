// clang-format off
// nvcc -I include src/cuda/mic.cu test/cuda/mic.cu -o cuda_mic.exe -std=c++17 --expt-relaxed-constexpr
// nvcc -I include src/cuda/mic.cu test/cuda/mic.cu -o cuda_mic.exe -std=c++17 -lineinfo -O3 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/mic.h>
#include <cuda_runtime.h>

#include <initializer_list>
#include <vector>

#include "mt19937.hpp"
#include "uint128_t.h"
#include "utils.cuh"

using namespace FastFss;

MT19937Rng rng;

#define LOG(fmt, ...)                                                 \
    std::fprintf(stderr, "[FastFss] " fmt ". %s:%d\n", ##__VA_ARGS__, \
                 __FILE__, __LINE__)

#define CHECK(exp)                    \
    [&] {                             \
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
class Test
{
public:
    static void run(std::size_t                     bitWidthIn,
                    std::size_t                     bitWidthOut,
                    std::size_t                     elementNum,
                    const std::vector<GroupElement> leftBoundary,
                    const std::vector<GroupElement> rightBoundary)
    {
        std::printf(
            "[cuda test] elementSize = %2d bitWidthIn = %3d bitWidthOut = "
            "%3d elementNum = %5d",
            (int)(sizeof(GroupElement)), (int)bitWidthIn, (int)bitWidthOut,
            (int)elementNum);

        std::size_t intervalNum = leftBoundary.size();
        std::size_t elementSize = sizeof(GroupElement);

        std::vector<GroupElement> x(elementNum);
        std::vector<GroupElement> maskedX(elementNum);
        std::vector<GroupElement> alpha(elementNum);
        std::vector<GroupElement> z(elementNum * intervalNum);
        std::vector<GroupElement> sharedZ0(elementNum * intervalNum);
        std::vector<GroupElement> sharedZ1(elementNum * intervalNum);
        std::vector<GroupElement> sharedOut0(elementNum * intervalNum);
        std::vector<GroupElement> sharedOut1(elementNum * intervalNum);

        std::vector<std::uint8_t> seed0(elementNum * 16);
        std::vector<std::uint8_t> seed1(elementNum * 16);

        std::size_t alphaDataSize = elementNum * sizeof(GroupElement);
        std::size_t zDataSize = elementNum * intervalNum * sizeof(GroupElement);
        std::size_t sharedZDataSize =
            elementNum * intervalNum * sizeof(GroupElement);
        std::size_t sharedOutDataSize =
            elementNum * intervalNum * sizeof(GroupElement);
        std::size_t seedDataSize    = elementNum * 16;
        std::size_t maskedXDataSize = elementNum * sizeof(GroupElement);
        std::size_t leftBoundaryDataSize =
            leftBoundary.size() * sizeof(GroupElement);
        std::size_t rightBoundaryDataSize =
            rightBoundary.size() * sizeof(GroupElement);

        int         ret;
        void*       deviceDcfMICKey   = nullptr;
        std::size_t dcfMICKeyDataSize = 0;
        void*       deviceCache       = nullptr;
        std::size_t cacheDataSize     = 0;

        {
            ret = FastFss_cuda_dcfMICGetKeyDataSize(&dcfMICKeyDataSize,
                                                    bitWidthIn, bitWidthOut,
                                                    elementSize, elementNum);
            CHECK(ret);
            deviceDcfMICKey = cuda::malloc_gpu(dcfMICKeyDataSize);
            ret = FastFss_cuda_dcfMICGetCacheDataSize(&cacheDataSize,
                                                      bitWidthIn, bitWidthOut,
                                                      elementSize, elementNum);
            CHECK(ret);
            deviceCache = cuda::malloc_gpu(cacheDataSize);
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            x[i]       = rng.rand<GroupElement>();
            alpha[i]   = rng.rand<GroupElement>();
            maskedX[i] = x[i] + alpha[i];

            x[i]       = mod_bits<GroupElement>(x[i], bitWidthIn);
            alpha[i]   = mod_bits<GroupElement>(alpha[i], bitWidthIn);
            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
        }

        {
            void* deviceZ             = cuda::malloc_gpu(zDataSize);
            void* deviceAlpha         = cuda::malloc_gpu(alphaDataSize);
            void* deviceSeed0         = cuda::malloc_gpu(seedDataSize);
            void* deviceSeed1         = cuda::malloc_gpu(seedDataSize);
            void* deviceLeftBoundary  = cuda::malloc_gpu(leftBoundaryDataSize);
            void* deviceRightBoundary = cuda::malloc_gpu(rightBoundaryDataSize);

            cuda::memcpy_cpu2gpu(deviceZ, z.data(), zDataSize);
            cuda::memcpy_cpu2gpu(deviceAlpha, alpha.data(), alphaDataSize);
            cuda::memcpy_cpu2gpu(deviceSeed0, seed0.data(), seedDataSize);
            cuda::memcpy_cpu2gpu(deviceSeed1, seed1.data(), seedDataSize);
            cuda::memcpy_cpu2gpu(deviceLeftBoundary, leftBoundary.data(),
                                 leftBoundaryDataSize);
            cuda::memcpy_cpu2gpu(deviceRightBoundary, rightBoundary.data(),
                                 rightBoundaryDataSize);

            ret = FastFss_cuda_dcfMICKeyGen(
                deviceDcfMICKey, dcfMICKeyDataSize, deviceZ, zDataSize,
                deviceAlpha, alphaDataSize, deviceSeed0, seedDataSize,
                deviceSeed1, seedDataSize, deviceLeftBoundary,
                leftBoundaryDataSize, deviceRightBoundary,
                rightBoundaryDataSize, bitWidthIn, bitWidthOut,
                sizeof(GroupElement), elementNum, nullptr);

            cuda::memcpy_gpu2cpu(z.data(), deviceZ, zDataSize);

            cuda::free_gpu(deviceZ);
            cuda::free_gpu(deviceAlpha);
            cuda::free_gpu(deviceSeed0);
            cuda::free_gpu(deviceSeed1);
            cuda::free_gpu(deviceLeftBoundary);
            cuda::free_gpu(deviceRightBoundary);

            CHECK(ret);
        }

        // split share
        for (std::size_t i = 0; i < z.size(); ++i)
        {
            sharedZ0[i] = rng.rand<GroupElement>();
            sharedZ1[i] = z[i] - sharedZ0[i];
        }

        {
            void* deviceSharedOut0    = cuda::malloc_gpu(sharedOutDataSize);
            void* deviceSharedZ0      = cuda::malloc_gpu(sharedZDataSize);
            void* deviceMaskedX       = cuda::malloc_gpu(maskedXDataSize);
            void* deviceSeed0         = cuda::malloc_gpu(seedDataSize);
            void* deviceLeftBoundary  = cuda::malloc_gpu(leftBoundaryDataSize);
            void* deviceRightBoundary = cuda::malloc_gpu(rightBoundaryDataSize);

            cuda::memcpy_cpu2gpu(deviceSharedZ0, sharedZ0.data(),
                                 sharedZDataSize);
            cuda::memcpy_cpu2gpu(deviceMaskedX, maskedX.data(),
                                 maskedXDataSize);
            cuda::memcpy_cpu2gpu(deviceSeed0, seed0.data(), seedDataSize);
            cuda::memcpy_cpu2gpu(deviceLeftBoundary, leftBoundary.data(),
                                 leftBoundaryDataSize);
            cuda::memcpy_cpu2gpu(deviceRightBoundary, rightBoundary.data(),
                                 rightBoundaryDataSize);

            ret = FastFss_cuda_dcfMICEval(
                deviceSharedOut0, sharedOutDataSize, deviceMaskedX,
                maskedXDataSize, deviceDcfMICKey, dcfMICKeyDataSize,
                deviceSharedZ0, sharedZDataSize, deviceSeed0, seedDataSize, 0,
                deviceLeftBoundary, leftBoundaryDataSize, deviceRightBoundary,
                rightBoundaryDataSize, bitWidthIn, bitWidthOut,
                sizeof(GroupElement), elementNum, deviceCache, cacheDataSize,
                nullptr);

            cuda::memcpy_gpu2cpu(sharedOut0.data(), deviceSharedOut0,
                                 sharedOutDataSize);

            cuda::free_gpu(deviceSharedOut0);
            cuda::free_gpu(deviceSharedZ0);
            cuda::free_gpu(deviceMaskedX);
            cuda::free_gpu(deviceSeed0);
            cuda::free_gpu(deviceLeftBoundary);
            cuda::free_gpu(deviceRightBoundary);

            CHECK(ret);
        }

        {
            void* deviceSharedOut1    = cuda::malloc_gpu(sharedOutDataSize);
            void* deviceSharedZ1      = cuda::malloc_gpu(sharedZDataSize);
            void* deviceMaskedX       = cuda::malloc_gpu(maskedXDataSize);
            void* deviceSeed1         = cuda::malloc_gpu(seedDataSize);
            void* deviceLeftBoundary  = cuda::malloc_gpu(leftBoundaryDataSize);
            void* deviceRightBoundary = cuda::malloc_gpu(rightBoundaryDataSize);

            cuda::memcpy_cpu2gpu(deviceSharedZ1, sharedZ1.data(),
                                 sharedZDataSize);
            cuda::memcpy_cpu2gpu(deviceMaskedX, maskedX.data(),
                                 maskedXDataSize);
            cuda::memcpy_cpu2gpu(deviceSeed1, seed1.data(), seedDataSize);
            cuda::memcpy_cpu2gpu(deviceLeftBoundary, leftBoundary.data(),
                                 leftBoundaryDataSize);
            cuda::memcpy_cpu2gpu(deviceRightBoundary, rightBoundary.data(),
                                 rightBoundaryDataSize);

            ret = FastFss_cuda_dcfMICEval(
                deviceSharedOut1, sharedOutDataSize, deviceMaskedX,
                maskedXDataSize, deviceDcfMICKey, dcfMICKeyDataSize,
                deviceSharedZ1, sharedZDataSize, deviceSeed1, seedDataSize, 1,
                deviceLeftBoundary, leftBoundaryDataSize, deviceRightBoundary,
                rightBoundaryDataSize, bitWidthIn, bitWidthOut,
                sizeof(GroupElement), elementNum, nullptr, 0, nullptr);

            cuda::memcpy_gpu2cpu(sharedOut1.data(), deviceSharedOut1,
                                 sharedOutDataSize);

            cuda::free_gpu(deviceSharedOut1);
            cuda::free_gpu(deviceSharedZ1);
            cuda::free_gpu(deviceMaskedX);
            cuda::free_gpu(deviceSeed1);
            cuda::free_gpu(deviceLeftBoundary);
            cuda::free_gpu(deviceRightBoundary);

            CHECK(ret);
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            for (std::size_t k = 0; k < intervalNum; ++k)
            {
                GroupElement v = sharedOut0[i * intervalNum + k] +
                                 sharedOut1[i * intervalNum + k];
                v = mod_bits<GroupElement>(v, bitWidthOut);

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
                    std::printf("[err] x = %lld, left = "
                                "%lld, right = %lld, v = %lld\n",
                                (long long)x[i], (long long)leftBoundary[k],
                                (long long)rightBoundary[k], (long long)v);
                    std::exit(-1);
                }
            }
        }
        cuda::free_gpu(deviceCache);
        cuda::free_gpu(deviceDcfMICKey);
        std::puts("  pass");
    }
};

int main()
{
    constexpr std::size_t elementNum = 256 * 512;

    rng.reseed(7);
    Test<std::uint8_t>::run(4, 8, elementNum, {1, 2, 3, 4}, {1, 2, 3, 4});
    Test<std::uint8_t>::run(8, 8, elementNum, {1, 2, 3, 4}, {1, 2, 3, 4});
    Test<std::uint16_t>::run(12, 8, elementNum,            //
                             {10, 20, 30, 40, 50, 60, 70}, //
                             {19, 29, 39, 49, 59, 69, 79}  //
    );
    Test<std::uint16_t>::run(16, 8, elementNum,            //
                             {10, 20, 30, 40, 50, 60, 70}, //
                             {19, 29, 39, 49, 59, 69, 79}  //
    );
    Test<std::uint32_t>::run(24, 8, elementNum,                   //
                             {100, 200, 300, 400, 500, 600, 700}, //
                             {199, 299, 399, 499, 599, 699, 799}  //
    );
    Test<std::uint32_t>::run(32, 8, elementNum,                   //
                             {100, 200, 300, 400, 500, 600, 700}, //
                             {199, 299, 399, 499, 599, 699, 799}  //
    );

    Test<std::uint64_t>::run(32, 8, elementNum,                          //
                             {1000, 2000, 3000, 4000, 5000, 6000, 7000}, //
                             {1999, 2999, 3999, 4999, 5999, 6999, 7999}  //
    );

    Test<std::uint64_t>::run(
        48, 8, elementNum,                                 //
        {10000, 20000, 30000, 40000, 50000, 60000, 70000}, //
        {19999, 29999, 39999, 49999, 59999, 69999, 79999}  //
    );
    
    Test<uint128_t>::run(128, 127, 1024 - 1,                                //
                         {10000, 20000, 30000, 40000, 50000, 60000, 70000}, //
                         {20000, 30000, 40000, 50000, 60000, 70000, 80000}  //
    );

    Test<uint128_t>::run(48, 128, 1024 - 1,                                 //
                         {10000, 20000, 30000, 40000, 50000, 60000, 70000}, //
                         {20000, 30000, 40000, 50000, 60000, 70000, 80000}  //
    );

    return 0;
}