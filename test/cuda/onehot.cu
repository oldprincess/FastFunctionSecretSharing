// clang-format off
// nvcc -I include src/cuda/onehot.cu test/cuda/onehot.cu -o cuda_onehot.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/onehot.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mt19937.hpp"
#include "uint128_t.h"
#include "utils.cuh"

using namespace std::chrono;

MT19937Rng rng;

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
static inline GroupElement mod_bits(GroupElement x, int bitWidth) noexcept
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
class TestOnehotLutEval
{
public:
    static void run(std::size_t bitWidthIn,
                    std::size_t bitWidthOut,
                    std::size_t elementNum,
                    bool        testSpeed = false)
    {
        std::printf("[TEST CUDA ONEHOT] bitWidthIn=%3zu, bitWidthOut=%3zu, "
                    "elementNum=%8zu\n",
                    bitWidthIn, bitWidthOut, elementNum);

        using namespace FastFss;

        int   ret;
        void* dKey0 = nullptr;
        void* dKey1 = nullptr;
        void* dLut  = nullptr;

        high_resolution_clock::time_point st;
        high_resolution_clock::time_point et;

        std::size_t keyDataSize;
        ret = FastFss_cuda_onehotGetKeyDataSize(&keyDataSize, bitWidthIn,
                                                elementNum);
        CHECK(ret);

        std::size_t xDataSize       = elementNum * sizeof(GroupElement);
        std::size_t maskedXDataSize = elementNum * sizeof(GroupElement);
        std::size_t alphaDataSize   = elementNum * sizeof(GroupElement);
        std::size_t lutDataSize     = (1 << bitWidthIn) * sizeof(GroupElement);
        std::size_t sharedOutDataSize = elementNum * sizeof(GroupElement);

        std::vector<GroupElement> x(elementNum);
        std::vector<GroupElement> maskedX(elementNum);
        std::vector<GroupElement> alpha(elementNum);
        std::vector<GroupElement> sharedOutE0(elementNum);
        std::vector<GroupElement> sharedOutE1(elementNum);
        std::vector<GroupElement> sharedOutT0(elementNum);
        std::vector<GroupElement> sharedOutT1(elementNum);
        std::vector<GroupElement> lut(1 << bitWidthIn);
        std::vector<std::uint8_t> key(keyDataSize);

        rng.gen(x.data(), xDataSize);
        rng.gen(alpha.data(), alphaDataSize);
        rng.gen(lut.data(), lutDataSize);
        rng.gen(key.data(), keyDataSize);
        for (int i = 0; i < elementNum; i++)
        {
            x[i]     = 97;
            alpha[i] = 113;

            maskedX[i] = x[i] + alpha[i];

            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
            x[i]       = mod_bits<GroupElement>(x[i], bitWidthIn);
            alpha[i]   = mod_bits<GroupElement>(alpha[i], bitWidthIn);
        }
        for (int i = 0; i < lut.size(); i++)
        {
            lut[i] = i;
        }

        dKey0 = cuda::malloc_gpu(keyDataSize);
        dKey1 = cuda::malloc_gpu(keyDataSize);
        dLut  = cuda::malloc_gpu(lutDataSize);
        cuda::memcpy_cpu2gpu(dKey0, key.data(), keyDataSize);
        cuda::memcpy_cpu2gpu(dKey1, key.data(), keyDataSize);
        cuda::memcpy_cpu2gpu(dLut, lut.data(), lutDataSize);

        {
            void* dAlpha = cuda::malloc_gpu(alphaDataSize);
            cuda::memcpy_cpu2gpu(dAlpha, alpha.data(), alphaDataSize);

            if (testSpeed)
            {
                st = high_resolution_clock::now();
            }

            ret = FastFss_cuda_onehotKeyGen(
                dKey1, keyDataSize, dAlpha, alphaDataSize, bitWidthIn,
                sizeof(GroupElement), elementNum, nullptr);
            CHECK(ret);

            if (testSpeed)
            {
                et        = high_resolution_clock::now();
                auto diff = duration_cast<microseconds>(et - st);
                std::printf("[CUDA ONEHOT KEYGEN] %f ms,  %f us/element\n",
                            (double)diff.count() / 1e3,
                            (double)diff.count() / elementNum);
            }

            cuda::free_gpu(dAlpha);
        }

        {
            void* dSharedOutE0 = cuda::malloc_gpu(sharedOutDataSize);
            void* dSharedOutT0 = cuda::malloc_gpu(sharedOutDataSize);
            void* dMaskedX     = cuda::malloc_gpu(maskedXDataSize);

            cuda::memcpy_cpu2gpu(dMaskedX, maskedX.data(), maskedXDataSize);

            if (testSpeed)
            {
                st = high_resolution_clock::now();
            }

            ret = FastFss_cuda_onehotLutEval(
                dSharedOutE0, dSharedOutT0, dMaskedX, maskedXDataSize, dKey0,
                keyDataSize, 0, dLut, lutDataSize, bitWidthIn,
                sizeof(GroupElement), elementNum, nullptr);
            CHECK(ret);

            if (testSpeed)
            {
                et        = high_resolution_clock::now();
                auto diff = duration_cast<microseconds>(et - st);
                std::printf("[CUDA ONEHOT   EVAL] %f ms,  %f us/element\n",
                            (double)diff.count() / 1e3,
                            (double)diff.count() / elementNum);
            }

            cuda::memcpy_gpu2cpu(sharedOutE0.data(), dSharedOutE0,
                                 sharedOutDataSize);
            cuda::memcpy_gpu2cpu(sharedOutT0.data(), dSharedOutT0,
                                 sharedOutDataSize);

            cuda::free_gpu(dSharedOutE0);
            cuda::free_gpu(dSharedOutT0);
            cuda::free_gpu(dMaskedX);
        }
        {
            void* dSharedOutE1 = cuda::malloc_gpu(sharedOutDataSize);
            void* dSharedOutT1 = cuda::malloc_gpu(sharedOutDataSize);
            void* dMaskedX     = cuda::malloc_gpu(maskedXDataSize);

            cuda::memcpy_cpu2gpu(dMaskedX, maskedX.data(), maskedXDataSize);

            ret = FastFss_cuda_onehotLutEval(
                dSharedOutE1, dSharedOutT1, dMaskedX, maskedXDataSize, dKey1,
                keyDataSize, 1, dLut, lutDataSize, bitWidthIn,
                sizeof(GroupElement), elementNum, nullptr);
            CHECK(ret);

            cuda::memcpy_gpu2cpu(sharedOutE1.data(), dSharedOutE1,
                                 sharedOutDataSize);
            cuda::memcpy_gpu2cpu(sharedOutT1.data(), dSharedOutT1,
                                 sharedOutDataSize);

            cuda::free_gpu(dSharedOutE1);
            cuda::free_gpu(dSharedOutT1);
            cuda::free_gpu(dMaskedX);
        }

        for (std::size_t i = 0; i < elementNum; i++)
        {
            GroupElement outE    = sharedOutE0[i] + sharedOutE1[i];
            GroupElement outT    = sharedOutT0[i] + sharedOutT1[i];
            GroupElement needOut = lut[x[i]];

            outE = mod_bits<GroupElement>(outE, 1);
            outT = mod_bits<GroupElement>(outT, bitWidthOut);

            GroupElement out = (outE == 0) ? outT : (0 - outT);
            out              = mod_bits<GroupElement>(out, bitWidthOut);
            if (out != needOut)
            {
                std::printf("E = %lld\n", (long long)outE);
                std::printf("T = %lld\n", (long long)outT);
                std::printf("X = %lld\n", (long long)x[i]);
                std::printf("A = %lld\n", (long long)alpha[i]);
                LOG("error. idx = %zu, out = %lld need = %lld", i,
                    (long long)out, (long long)needOut);
                std::exit(-1);
            }
        }

        cuda::free_gpu(dKey0);
        cuda::free_gpu(dKey1);
        cuda::free_gpu(dLut);
    }
};

int main()
{
    rng.reseed(7);
    TestOnehotLutEval<std::uint8_t>::run(7, 7, 1023);
    TestOnehotLutEval<std::uint8_t>::run(8, 8, 1023);
    TestOnehotLutEval<std::uint16_t>::run(9, 16, 1023);
    TestOnehotLutEval<std::uint32_t>::run(9, 32, 1023);
    TestOnehotLutEval<std::uint64_t>::run(9, 64, 1023);

    TestOnehotLutEval<std::uint64_t>::run(8, 64, 12 * 128 * 128, true);
    TestOnehotLutEval<std::uint64_t>::run(9, 64, 12 * 128 * 128, true);
    TestOnehotLutEval<std::uint64_t>::run(9, 64, 12 * 128, true);

    TestOnehotLutEval<uint128_t>::run(9, 127, 12 * 128, true);
    TestOnehotLutEval<uint128_t>::run(9, 128, 12 * 128, true);
    return 0;
}