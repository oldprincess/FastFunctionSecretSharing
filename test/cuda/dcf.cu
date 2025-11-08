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

template <typename T>
constexpr T mod_bits(T x, int bitWidth) noexcept
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

using namespace FastFss;

template <typename T>
class TestDcf
{
public:
    static void run(std::size_t bitWidthIn,
                    std::size_t bitWidthOut,
                    std::size_t groupSize,
                    std::size_t elementNum)
    {
        std::printf("[cuda test] "
                    "elementSize = %2d "
                    "groupSize = %3d "
                    "bitWidthIn = %3d "
                    "bitWidthOut = %3d "
                    "elementNum = %5d",
                    (int)(sizeof(T)), //
                    (int)groupSize,   //
                    (int)bitWidthIn,  //
                    (int)bitWidthOut, //
                    (int)elementNum);

        std::unique_ptr<T[]> alpha(new T[elementNum]);
        std::unique_ptr<T[]> beta(new T[elementNum * groupSize]);
        std::size_t          alphaDataSize = sizeof(T) * elementNum;
        std::size_t          betaDataSize  = sizeof(T) * elementNum * groupSize;

        for (std::size_t i = 0; i < elementNum; i++)
        {
            alpha[i] = rng.rand<T>();
            alpha[i] = mod_bits<T>(alpha[i], bitWidthIn);
        }
        for (std::size_t i = 0; i < elementNum * groupSize; i++)
        {
            beta[i] = rng.rand<T>();
            beta[i] = mod_bits<T>(beta[i], bitWidthOut);
        }

        std::unique_ptr<std::uint8_t[]> seed0(new uint8_t[16 * elementNum]);
        std::unique_ptr<std::uint8_t[]> seed1(new uint8_t[16 * elementNum]);
        std::size_t                     seedDataSize0 = 16 * elementNum;
        std::size_t                     seedDataSize1 = 16 * elementNum;

        rng.gen(seed0.get(), seedDataSize0);
        rng.gen(seed1.get(), seedDataSize1);

        std::unique_ptr<T[]> x(new T[elementNum]);
        std::unique_ptr<T[]> maskedX(new T[elementNum]);
        std::unique_ptr<T[]> sharedOut0(new T[elementNum * groupSize]);
        std::unique_ptr<T[]> sharedOut1(new T[elementNum * groupSize]);
        std::size_t          maskedXDataSize = sizeof(T) * elementNum;
        for (std::size_t i = 0; i < elementNum; i++)
        {
            x[i]       = rng.rand<T>();
            x[i]       = mod_bits<T>(x[i], bitWidthIn);
            maskedX[i] = x[i] + alpha[i];
            maskedX[i] = mod_bits<T>(maskedX[i], bitWidthIn);
        }
        std::size_t sharedOutDataSize = sizeof(T) * elementNum * groupSize;
        void       *deviceDcfKey      = nullptr;
        std::size_t dcfKeyDataSize;
        int         ret;

        ret = FastFss_cuda_dcfGetKeyDataSize(&dcfKeyDataSize, bitWidthIn,
                                             bitWidthOut, groupSize, sizeof(T),
                                             elementNum);
        CHECK(ret);
        deviceDcfKey = cuda::malloc_gpu(dcfKeyDataSize);

        {
            void *deviceAlpha = cuda::malloc_gpu(alphaDataSize);
            void *deviceBeta  = cuda::malloc_gpu(betaDataSize);
            void *deviceSeed0 = cuda::malloc_gpu(seedDataSize0);
            void *deviceSeed1 = cuda::malloc_gpu(seedDataSize1);

            cuda::memcpy_cpu2gpu(deviceAlpha, alpha.get(), alphaDataSize);
            cuda::memcpy_cpu2gpu(deviceBeta, beta.get(), betaDataSize);
            cuda::memcpy_cpu2gpu(deviceSeed0, seed0.get(), seedDataSize0);
            cuda::memcpy_cpu2gpu(deviceSeed1, seed1.get(), seedDataSize1);

            ret = FastFss_cuda_dcfKeyGen(
                deviceDcfKey, dcfKeyDataSize, deviceAlpha, alphaDataSize,
                deviceBeta, betaDataSize, deviceSeed0, seedDataSize0,
                deviceSeed1, seedDataSize1, bitWidthIn, bitWidthOut, groupSize,
                sizeof(T), elementNum, nullptr);
            CHECK(ret);

            cuda::free_gpu(deviceAlpha);
            cuda::free_gpu(deviceBeta);
            cuda::free_gpu(deviceSeed0);
            cuda::free_gpu(deviceSeed1);
        }

        {
            void *deviceSharedOut0 = cuda::malloc_gpu(sharedOutDataSize);
            void *deviceMaskedX    = cuda::malloc_gpu(maskedXDataSize);
            void *deviceSeed0      = cuda::malloc_gpu(seedDataSize0);

            cuda::memcpy_cpu2gpu(deviceMaskedX, maskedX.get(), maskedXDataSize);
            cuda::memcpy_cpu2gpu(deviceSeed0, seed0.get(), seedDataSize0);

            ret = FastFss_cuda_dcfEval(
                deviceSharedOut0, sharedOutDataSize, deviceMaskedX,
                maskedXDataSize, deviceDcfKey, dcfKeyDataSize, deviceSeed0,
                seedDataSize0, 0, bitWidthIn, bitWidthOut, groupSize, sizeof(T),
                elementNum, nullptr, 0, nullptr);

            cuda::memcpy_gpu2cpu(sharedOut0.get(), deviceSharedOut0,
                                 sharedOutDataSize);

            cuda::free_gpu(deviceSharedOut0);
            cuda::free_gpu(deviceMaskedX);
            cuda::free_gpu(deviceSeed0);
            CHECK(ret);
        }

        {
            void *deviceSharedOut1 = cuda::malloc_gpu(sharedOutDataSize);
            void *deviceMaskedX    = cuda::malloc_gpu(maskedXDataSize);
            void *deviceSeed1      = cuda::malloc_gpu(seedDataSize1);

            cuda::memcpy_cpu2gpu(deviceMaskedX, maskedX.get(), maskedXDataSize);
            cuda::memcpy_cpu2gpu(deviceSeed1, seed1.get(), seedDataSize1);

            ret = FastFss_cuda_dcfEval(
                deviceSharedOut1, sharedOutDataSize, deviceMaskedX,
                maskedXDataSize, deviceDcfKey, dcfKeyDataSize, deviceSeed1,
                seedDataSize1, 1, bitWidthIn, bitWidthOut, groupSize, sizeof(T),
                elementNum, nullptr, 0, nullptr);

            cuda::memcpy_gpu2cpu(sharedOut1.get(), deviceSharedOut1,
                                 sharedOutDataSize);

            cuda::free_gpu(deviceSharedOut1);
            cuda::free_gpu(deviceMaskedX);
            cuda::free_gpu(deviceSeed1);

            CHECK(ret);
        }

        for (int i = 0; i < elementNum; i++)
        {
            if (maskedX[i] < alpha[i])
            {
                for (std::size_t j = 0; j < groupSize; j++)
                {
                    T v = sharedOut0[i * groupSize + j] +
                          sharedOut1[i * groupSize + j];
                    v = mod_bits<T>(v, bitWidthOut);
                    if (v != beta[i * groupSize + j])
                    {
                        std::printf("Error in %s:%d\n", __FILE__, __LINE__);
                        std::exit(-1);
                    }
                }
            }
            else
            {
                for (std::size_t j = 0; j < groupSize; j++)
                {
                    T v = sharedOut0[i * groupSize + j] +
                          sharedOut1[i * groupSize + j];
                    v = mod_bits<T>(v, bitWidthOut);
                    if (v != 0)
                    {
                        std::printf("Error in %s:%d\n", __FILE__, __LINE__);
                        std::exit(-1);
                    }
                }
            }
        }
        cuda::free_gpu(deviceDcfKey);

        std::puts("  pass");
    }
};

int main()
{
    rng.reseed(7);
    // uint8
    TestDcf<std::uint8_t>::run(1, 8, 1, 1024 - 1);
    TestDcf<std::uint8_t>::run(2, 8, 1, 1024 - 1);
    TestDcf<std::uint8_t>::run(3, 8, 1, 1024 - 1);
    TestDcf<std::uint8_t>::run(4, 8, 1, 1024 - 1);
    TestDcf<std::uint8_t>::run(5, 8, 1, 1024 - 1);
    TestDcf<std::uint8_t>::run(6, 8, 1, 1024 - 1);
    TestDcf<std::uint8_t>::run(7, 8, 1, 1024 - 1);
    TestDcf<std::uint8_t>::run(8, 8, 1, 1024 - 1);
    TestDcf<std::uint8_t>::run(8, 8, 4, 1024 - 1);
    TestDcf<std::uint8_t>::run(8, 8, 30, 1024 - 1);
    // uint16
    TestDcf<std::uint16_t>::run(12, 8, 1, 1024 - 1);
    TestDcf<std::uint16_t>::run(16, 8, 1, 1024 - 1);
    TestDcf<std::uint16_t>::run(16, 8, 4, 1024 - 1);
    TestDcf<std::uint16_t>::run(16, 8, 30, 1024 - 1);
    // uint32
    TestDcf<std::uint32_t>::run(18, 16, 1, 1024 - 1);
    TestDcf<std::uint32_t>::run(18, 8, 1, 1024 - 1);
    TestDcf<std::uint32_t>::run(18, 8, 4, 1024 - 1);
    TestDcf<std::uint32_t>::run(18, 8, 30, 1024 - 1);
    // uint64
    TestDcf<std::uint64_t>::run(63, 16, 1, 1024 - 1);
    TestDcf<std::uint64_t>::run(63, 16, 4, 1024 - 1);
    TestDcf<std::uint64_t>::run(63, 16, 30, 1024 - 1);
    // uint128
    TestDcf<uint128_t>::run(127, 128, 1, 1024 - 1);
    TestDcf<uint128_t>::run(128, 127, 1, 1024 - 1);
    TestDcf<uint128_t>::run(128, 127, 4, 1024 - 1);
    TestDcf<uint128_t>::run(128, 127, 30, 1024 - 1);
    return 0;
}