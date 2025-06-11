// clang-format off
// nvcc -I include src/cuda/dcf.cu test/cuda/dcf.cu -o cuda_dcf.exe -std=c++17 --expt-relaxed-constexpr
// nvcc -I include src/cuda/dcf.cu test/cuda/dcf.cu -o cuda_dcf.exe -std=c++17 -lineinfo -O3 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/dcf.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "mt19937.hpp"
#include "uint128_t.h"
#include "utils.cuh"

MT19937Rng rng;

#define LOG(fmt, ...)                                                 \
    std::fprintf(stderr, "[FastFss] " fmt ". %s:%d\n", ##__VA_ARGS__, \
                 __FILE__, __LINE__)

#define CHECK(exp)                    \
    [&] {                             \
        cudaDeviceSynchronize();      \
        auto the_ret = (exp);         \
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

using namespace FastFss;

template <typename GroupElement>
class TestDcf
{
public:
    static void run(std::size_t bitWidthIn,
                    std::size_t bitWidthOut,
                    std::size_t elementNum)
    {
        std::printf(
            "[cuda test] elementSize = %2d bitWidthIn = %3d bitWidthOut = "
            "%3d elementNum = %5d",
            (int)(sizeof(GroupElement)), (int)bitWidthIn, (int)bitWidthOut,
            (int)elementNum);

        std::unique_ptr<GroupElement[]> alpha(new GroupElement[elementNum]);
        std::unique_ptr<GroupElement[]> beta(new GroupElement[elementNum]);
        std::size_t alphaDataSize = sizeof(GroupElement) * elementNum;
        std::size_t betaDataSize  = sizeof(GroupElement) * elementNum;

        for (std::size_t i = 0; i < elementNum; i++)
        {
            alpha[i] = rng.rand<GroupElement>();
            beta[i]  = rng.rand<GroupElement>();
            alpha[i] = mod_bits<GroupElement>(alpha[i], bitWidthIn);
            beta[i]  = mod_bits<GroupElement>(beta[i], bitWidthOut);
        }

        std::unique_ptr<std::uint8_t[]> seed0(new uint8_t[16 * elementNum]);
        std::unique_ptr<std::uint8_t[]> seed1(new uint8_t[16 * elementNum]);
        std::size_t                     seedDataSize0 = 16 * elementNum;
        std::size_t                     seedDataSize1 = 16 * elementNum;

        rng.gen(seed0.get(), seedDataSize0);
        rng.gen(seed1.get(), seedDataSize1);

        std::unique_ptr<GroupElement[]> x(new GroupElement[elementNum]);
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

            sharedOut0[i] = 0;
            sharedOut1[i] = 0;
        }

        void*       deviceDcfKey = nullptr;
        std::size_t dcfKeyDataSize;
        int         ret;

        ret = FastFss_cuda_dcfGetKeyDataSize(&dcfKeyDataSize, bitWidthIn,
                                             bitWidthOut, sizeof(GroupElement),
                                             elementNum);
        CHECK(ret);
        deviceDcfKey = cuda::malloc_gpu(dcfKeyDataSize);

        {
            void* deviceAlpha = cuda::malloc_gpu(alphaDataSize);
            void* deviceBeta  = cuda::malloc_gpu(betaDataSize);
            void* deviceSeed0 = cuda::malloc_gpu(seedDataSize0);
            void* deviceSeed1 = cuda::malloc_gpu(seedDataSize1);

            cuda::memcpy_cpu2gpu(deviceAlpha, alpha.get(), alphaDataSize);
            cuda::memcpy_cpu2gpu(deviceBeta, beta.get(), betaDataSize);
            cuda::memcpy_cpu2gpu(deviceSeed0, seed0.get(), seedDataSize0);
            cuda::memcpy_cpu2gpu(deviceSeed1, seed1.get(), seedDataSize1);

            ret = FastFss_cuda_dcfKeyGen(
                deviceDcfKey, dcfKeyDataSize, deviceAlpha, alphaDataSize,
                deviceBeta, betaDataSize, deviceSeed0, seedDataSize0,
                deviceSeed1, seedDataSize1, bitWidthIn, bitWidthOut,
                sizeof(GroupElement), elementNum, nullptr);
            CHECK(ret);

            cuda::free_gpu(deviceAlpha);
            cuda::free_gpu(deviceBeta);
            cuda::free_gpu(deviceSeed0);
            cuda::free_gpu(deviceSeed1);
        }

        {
            void* deviceSharedOut0 = cuda::malloc_gpu(maskedXDataSize);
            void* deviceMaskedX    = cuda::malloc_gpu(maskedXDataSize);
            void* deviceSeed0      = cuda::malloc_gpu(seedDataSize0);

            cuda::memcpy_cpu2gpu(deviceMaskedX, maskedX.get(), maskedXDataSize);
            cuda::memcpy_cpu2gpu(deviceSeed0, seed0.get(), seedDataSize0);

            ret = FastFss_cuda_dcfEval(
                deviceSharedOut0, deviceMaskedX, maskedXDataSize, deviceDcfKey,
                dcfKeyDataSize, deviceSeed0, seedDataSize0, 0, bitWidthIn,
                bitWidthOut, sizeof(GroupElement), elementNum, nullptr, 0,
                nullptr);

            cuda::memcpy_gpu2cpu(sharedOut0.get(), deviceSharedOut0,
                                 maskedXDataSize);

            cuda::free_gpu(deviceSharedOut0);
            cuda::free_gpu(deviceMaskedX);
            cuda::free_gpu(deviceSeed0);
            CHECK(ret);
        }

        {
            void* deviceSharedOut1 = cuda::malloc_gpu(maskedXDataSize);
            void* deviceMaskedX    = cuda::malloc_gpu(maskedXDataSize);
            void* deviceSeed1      = cuda::malloc_gpu(seedDataSize1);

            cuda::memcpy_cpu2gpu(deviceMaskedX, maskedX.get(), maskedXDataSize);
            cuda::memcpy_cpu2gpu(deviceSeed1, seed1.get(), seedDataSize1);

            ret = FastFss_cuda_dcfEval(
                deviceSharedOut1, deviceMaskedX, maskedXDataSize, deviceDcfKey,
                dcfKeyDataSize, deviceSeed1, seedDataSize1, 1, bitWidthIn,
                bitWidthOut, sizeof(GroupElement), elementNum, nullptr, 0,
                nullptr);

            cuda::memcpy_gpu2cpu(sharedOut1.get(), deviceSharedOut1,
                                 maskedXDataSize);

            cuda::free_gpu(deviceSharedOut1);
            cuda::free_gpu(deviceMaskedX);
            cuda::free_gpu(deviceSeed1);

            CHECK(ret);
        }

        for (int i = 0; i < elementNum; i++)
        {
            GroupElement v = sharedOut0[i] + sharedOut1[i];
            v              = mod_bits<GroupElement>(v, bitWidthOut);

            bool cmp0 = ((maskedX[i] < alpha[i]) && (v == beta[i]));
            bool cmp1 = ((maskedX[i] >= alpha[i]) && (v == 0));
            if (!(cmp0 || cmp1))
            {
                std::printf("\n[%d] alpha = %lld, beta = %lld ", __LINE__,
                            (long long)alpha[i], (long long)beta[i]);
                std::printf("maskedX = %lld v = %lld", (long long)maskedX[i],
                            (long long)v);
                std::exit(-1);
            }
        }
        cuda::free_gpu(deviceDcfKey);

        std::puts("  pass");
    }
};

int main()
{
    constexpr std::size_t elementNum = 256 * 512;

    rng.reseed(7);
    // uint8
    TestDcf<std::uint8_t>::run(1, 8, elementNum);
    TestDcf<std::uint8_t>::run(2, 8, elementNum);
    TestDcf<std::uint8_t>::run(3, 8, elementNum);
    TestDcf<std::uint8_t>::run(4, 8, elementNum);
    TestDcf<std::uint8_t>::run(5, 8, elementNum);
    TestDcf<std::uint8_t>::run(6, 8, elementNum);
    TestDcf<std::uint8_t>::run(7, 8, elementNum);
    TestDcf<std::uint8_t>::run(8, 8, elementNum);
    // // uint16
    TestDcf<std::uint16_t>::run(12, 8, elementNum);
    TestDcf<std::uint16_t>::run(16, 8, elementNum);
    // // uint32
    TestDcf<std::uint32_t>::run(18, 16, elementNum);
    TestDcf<std::uint32_t>::run(18, 8, elementNum);
    // uint64
    TestDcf<std::uint64_t>::run(18, 16, elementNum);
    // uint128
    TestDcf<uint128_t>::run(127, 128, 1024 - 1);
    TestDcf<uint128_t>::run(128, 127, 1024 - 1);
    return 0;
}