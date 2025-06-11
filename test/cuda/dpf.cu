// clang-format off
// nvcc -I include src/cuda/dpf.cu test/cuda/dpf.cu -o cuda_dpf.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/dpf.h>

#include <chrono>
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

template <typename GroupElement>
class TestDpf
{
public:
    static void run(std::size_t bitWidthIn,
                    std::size_t bitWidthOut,
                    std::size_t elementNum)
    {
        using namespace FastFss;

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

            if (rng.rand<int>() % 2 == 0)
            {
                x[i] = 0;
            }

            maskedX[i] = x[i] + alpha[i];
            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
        }
        int         ret;
        void*       dpfKey = nullptr;
        std::size_t dpfKeyDataSize;

        ret = FastFss_cuda_dpfGetKeyDataSize(&dpfKeyDataSize, bitWidthIn,
                                             bitWidthOut, sizeof(GroupElement),
                                             elementNum);
        CHECK(ret);
        dpfKey = cuda::malloc_gpu(dpfKeyDataSize);
        CHECK(dpfKey == nullptr);

        {
            void* dAlpha = cuda::malloc_gpu(alphaDataSize);
            void* dBeta  = cuda::malloc_gpu(betaDataSize);
            void* dSeed0 = cuda::malloc_gpu(seedDataSize0);
            void* dSeed1 = cuda::malloc_gpu(seedDataSize1);

            cuda::memcpy_cpu2gpu(dAlpha, alpha.get(), alphaDataSize);
            cuda::memcpy_cpu2gpu(dBeta, beta.get(), betaDataSize);
            cuda::memcpy_cpu2gpu(dSeed0, seed0.get(), seedDataSize0);
            cuda::memcpy_cpu2gpu(dSeed1, seed1.get(), seedDataSize1);

            ret = FastFss_cuda_dpfKeyGen(
                dpfKey, dpfKeyDataSize, dAlpha, alphaDataSize, dBeta,
                betaDataSize, dSeed0, seedDataSize0, dSeed1, seedDataSize1,
                bitWidthIn, bitWidthOut, sizeof(GroupElement), elementNum,
                nullptr);
            CHECK(ret);
            cuda::free_gpu(dAlpha);
            cuda::free_gpu(dBeta);
            cuda::free_gpu(dSeed0);
            cuda::free_gpu(dSeed1);
        }

        {
            void* dSharedOut0 = cuda::malloc_gpu(maskedXDataSize);
            void* dMaskedX    = cuda::malloc_gpu(maskedXDataSize);
            void* dSeed0      = cuda::malloc_gpu(seedDataSize0);

            cuda::memcpy_cpu2gpu(dMaskedX, maskedX.get(), maskedXDataSize);
            cuda::memcpy_cpu2gpu(dSeed0, seed0.get(), seedDataSize0);

            ret = FastFss_cuda_dpfEval(
                dSharedOut0, dMaskedX, maskedXDataSize, dpfKey, dpfKeyDataSize,
                dSeed0, seedDataSize0, 0, bitWidthIn, bitWidthOut,
                sizeof(GroupElement), elementNum, nullptr, 0, nullptr);
            CHECK(ret);
            cuda::memcpy_gpu2cpu(sharedOut0.get(), dSharedOut0,
                                 maskedXDataSize);
            cuda::free_gpu(dSeed0);
            cuda::free_gpu(dSharedOut0);
            cuda::free_gpu(dMaskedX);
        }

        {
            void* dSharedOut1 = cuda::malloc_gpu(maskedXDataSize);
            void* dMaskedX    = cuda::malloc_gpu(maskedXDataSize);
            void* dSeed1      = cuda::malloc_gpu(seedDataSize1);

            cuda::memcpy_cpu2gpu(dMaskedX, maskedX.get(), maskedXDataSize);
            cuda::memcpy_cpu2gpu(dSeed1, seed1.get(), seedDataSize1);

            ret = FastFss_cuda_dpfEval(
                dSharedOut1, dMaskedX, maskedXDataSize, dpfKey, dpfKeyDataSize,
                dSeed1, seedDataSize1, 1, bitWidthIn, bitWidthOut,
                sizeof(GroupElement), elementNum, nullptr, 0, nullptr);
            CHECK(ret);

            cuda::memcpy_gpu2cpu(sharedOut1.get(), dSharedOut1,
                                 maskedXDataSize);
            cuda::free_gpu(dSeed1);
            cuda::free_gpu(dSharedOut1);
            cuda::free_gpu(dMaskedX);
        }

        for (int i = 0; i < elementNum; i++)
        {
            GroupElement v = sharedOut0[i] + sharedOut1[i];
            v              = mod_bits<GroupElement>(v, bitWidthOut);

            bool cmp0 = ((maskedX[i] == alpha[i]) && (v == beta[i]));
            bool cmp1 = ((maskedX[i] != alpha[i]) && (v == 0));
            if (!(cmp0 || cmp1))
            {
                std::printf("\n[%d] alpha = %lld, beta = %lld ", __LINE__,
                            (long long)alpha[i], (long long)beta[i]);
                std::printf("maskedX = %lld v = %lld", (long long)maskedX[i],
                            (long long)v);
                std::exit(-1);
            }
        }
        cuda::free_gpu(dpfKey);

        std::puts("  pass");
    }
};

template <typename GroupElement>
class TestDpfMulti
{
public:
    static void run(std::size_t               bitWidthIn,
                    std::size_t               bitWidthOut,
                    std::size_t               elementNum,
                    std::vector<GroupElement> point)
    {
        using namespace FastFss;

        std::printf("[cuda test] [TestDpfMulti] elementSize = %2d bitWidthIn = "
                    "%3d bitWidthOut = "
                    "%3d elementNum = %5d",
                    (int)(sizeof(GroupElement)), (int)bitWidthIn,
                    (int)bitWidthOut, (int)elementNum);

        std::size_t cacheDataSize;

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

        std::size_t pointNum      = point.size();
        std::size_t pointDataSize = sizeof(GroupElement) * pointNum;

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
            new GroupElement[elementNum * pointNum]);
        std::unique_ptr<GroupElement[]> sharedOut1(
            new GroupElement[elementNum * pointNum]);
        std::size_t sharedOutDataSize =
            sizeof(GroupElement) * elementNum * pointNum;
        std::size_t maskedXDataSize = sizeof(GroupElement) * elementNum;
        for (std::size_t i = 0; i < elementNum; i++)
        {
            x[i] = rng.rand<GroupElement>();
            x[i] = mod_bits<GroupElement>(x[i], bitWidthIn);

            if (rng.rand<int>() % 2 == 0)
            {
                int idx = rng.rand<int>() % pointNum;
                x[i]    = point[idx];
            }

            maskedX[i] = x[i] + alpha[i];
            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
        }
        int         ret;
        void*       dpfKey   = nullptr;
        void*       dpfCache = nullptr;
        std::size_t dpfKeyDataSize;

        ret = FastFss_cuda_dpfGetCacheDataSize(
            &cacheDataSize, bitWidthIn, bitWidthOut, sizeof(GroupElement),
            elementNum);
        CHECK(ret);
        dpfCache = cuda::malloc_gpu(cacheDataSize);
        CHECK(dpfCache == nullptr);

        ret = FastFss_cuda_dpfGetKeyDataSize(&dpfKeyDataSize, bitWidthIn,
                                             bitWidthOut, sizeof(GroupElement),
                                             elementNum);
        CHECK(ret);
        dpfKey = cuda::malloc_gpu(dpfKeyDataSize);
        CHECK(dpfKey == nullptr);

        {
            void* dAlpha = cuda::malloc_gpu(alphaDataSize);
            void* dBeta  = cuda::malloc_gpu(betaDataSize);
            void* dSeed0 = cuda::malloc_gpu(seedDataSize0);
            void* dSeed1 = cuda::malloc_gpu(seedDataSize1);

            cuda::memcpy_cpu2gpu(dAlpha, alpha.get(), alphaDataSize);
            cuda::memcpy_cpu2gpu(dBeta, beta.get(), betaDataSize);
            cuda::memcpy_cpu2gpu(dSeed0, seed0.get(), seedDataSize0);
            cuda::memcpy_cpu2gpu(dSeed1, seed1.get(), seedDataSize1);

            ret = FastFss_cuda_dpfKeyGen(
                dpfKey, dpfKeyDataSize, dAlpha, alphaDataSize, dBeta,
                betaDataSize, dSeed0, seedDataSize0, dSeed1, seedDataSize1,
                bitWidthIn, bitWidthOut, sizeof(GroupElement), elementNum,
                nullptr);
            CHECK(ret);

            cuda::free_gpu(dAlpha);
            cuda::free_gpu(dBeta);
            cuda::free_gpu(dSeed0);
            cuda::free_gpu(dSeed1);
        }

        {
            void* dSharedOut = cuda::malloc_gpu(sharedOutDataSize);
            void* dMaskedX   = cuda::malloc_gpu(maskedXDataSize);
            void* dSeed0     = cuda::malloc_gpu(seedDataSize0);
            void* dPoint     = cuda::malloc_gpu(pointDataSize);

            cuda::memcpy_cpu2gpu(dMaskedX, maskedX.get(), maskedXDataSize);
            cuda::memcpy_cpu2gpu(dSeed0, seed0.get(), seedDataSize0);
            cuda::memcpy_cpu2gpu(dPoint, point.data(), pointDataSize);

            ret = FastFss_cuda_dpfEvalMulti(
                dSharedOut, sharedOutDataSize, dMaskedX, maskedXDataSize,
                dpfKey, dpfKeyDataSize, dSeed0, seedDataSize0, 0, dPoint,
                pointDataSize, bitWidthIn, bitWidthOut, sizeof(GroupElement),
                elementNum, dpfCache, cacheDataSize, nullptr);
            CHECK(ret);

            cuda::memcpy_gpu2cpu(sharedOut0.get(), dSharedOut,
                                 sharedOutDataSize);

            cuda::free_gpu(dSeed0);
            cuda::free_gpu(dMaskedX);
            cuda::free_gpu(dSharedOut);
            cuda::free_gpu(dPoint);
        }

        {
            void* dSharedOut = cuda::malloc_gpu(sharedOutDataSize);
            void* dMaskedX   = cuda::malloc_gpu(maskedXDataSize);
            void* dSeed1     = cuda::malloc_gpu(seedDataSize1);
            void* dPoint     = cuda::malloc_gpu(pointDataSize);

            cuda::memcpy_cpu2gpu(dMaskedX, maskedX.get(), maskedXDataSize);
            cuda::memcpy_cpu2gpu(dSeed1, seed1.get(), seedDataSize1);
            cuda::memcpy_cpu2gpu(dPoint, point.data(), pointDataSize);

            ret = FastFss_cuda_dpfEvalMulti(
                dSharedOut, sharedOutDataSize, dMaskedX, maskedXDataSize,
                dpfKey, dpfKeyDataSize, dSeed1, seedDataSize1, 1, dPoint,
                pointDataSize, bitWidthIn, bitWidthOut, sizeof(GroupElement),
                elementNum, nullptr, 0, nullptr);
            CHECK(ret);

            cuda::memcpy_gpu2cpu(sharedOut1.get(), dSharedOut,
                                 sharedOutDataSize);

            cuda::free_gpu(dSeed1);
            cuda::free_gpu(dMaskedX);
            cuda::free_gpu(dSharedOut);
            cuda::free_gpu(dPoint);
        }

        for (int i = 0; i < elementNum; i++)
        {
            for (std::size_t j = 0; j < pointNum; j++)
            {
                GroupElement v =
                    sharedOut0[i * pointNum + j] + sharedOut1[i * pointNum + j];
                v = mod_bits<GroupElement>(v, bitWidthOut);

                bool cmp0 = ((x[i] == point[j]) && (v == beta[i]));
                bool cmp1 = ((x[i] != point[j]) && (v == 0));
                if (!(cmp0 || cmp1))
                {
                    std::printf("\n[%d] alpha = %lld, beta = %lld ", __LINE__,
                                (long long)alpha[i], (long long)beta[i]);
                    std::printf("maskedX = %lld v = %lld",
                                (long long)maskedX[i], (long long)v);
                    std::exit(-1);
                }
            }
        }
        cuda::free_gpu(dpfKey);
        cuda::free_gpu(dpfCache);

        std::puts("  pass");
    }
};

int main()
{
    rng.reseed(7);
    // uint8
    TestDpf<std::uint8_t>::run(1, 8, 1024 - 1);
    TestDpf<std::uint8_t>::run(2, 8, 1024 - 1);
    TestDpf<std::uint8_t>::run(3, 8, 1024 - 1);
    TestDpf<std::uint8_t>::run(4, 8, 1024 - 1);
    TestDpf<std::uint8_t>::run(5, 8, 1024 - 1);
    TestDpf<std::uint8_t>::run(6, 8, 1024 - 1);
    TestDpf<std::uint8_t>::run(7, 8, 1024 - 1);
    TestDpf<std::uint8_t>::run(8, 8, 1024 - 1);
    // uint16
    TestDpf<std::uint16_t>::run(12, 8, 1024 - 1);
    TestDpf<std::uint16_t>::run(16, 8, 1024 - 1);
    // uint32
    TestDpf<std::uint32_t>::run(18, 16, 1024 - 1);
    TestDpf<std::uint32_t>::run(18, 8, 1024 - 1);
    // uint64
    TestDpf<std::uint64_t>::run(63, 16, 1024 - 1);
    // uint128
    TestDpf<uint128_t>::run(127, 128, 1024 - 1);
    TestDpf<uint128_t>::run(128, 127, 1024 - 1);

    // uint64
    TestDpfMulti<std::uint16_t>::run(8, 8, 1024 - 1, {0, 2, 4, 8});

    TestDpfMulti<std::uint64_t>::run(18, 18, 1024 - 1, {0, 2, 4, 8});

    // uint128
    TestDpfMulti<uint128_t>::run(18, 18, 1024 - 1, {0, 2, 4, 8});
    return 0;
}