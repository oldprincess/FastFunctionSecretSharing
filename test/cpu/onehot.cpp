// clang-format off
// g++ -I include src/cpu/onehot.cpp src/cpu/config.cpp test/cpu/onehot.cpp -o cpu_onehot.exe -std=c++17 -fopenmp
// clang-format on
#include <FastFss/cpu/onehot.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mt19937.hpp"

using namespace std::chrono;

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

        int   ret;
        void* dKey0 = nullptr;
        void* dKey1 = nullptr;
        void* dLut  = nullptr;

        high_resolution_clock::time_point st;
        high_resolution_clock::time_point et;

        std::size_t keyDataSize;
        ret = FastFss_cpu_onehotGetKeyDataSize(&keyDataSize, bitWidthIn,
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
            maskedX[i] = x[i] + alpha[i];

            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
            x[i]       = mod_bits<GroupElement>(x[i], bitWidthIn);
            alpha[i]   = mod_bits<GroupElement>(alpha[i], bitWidthIn);
        }
        for (int i = 0; i < lut.size(); i++)
        {
            lut[i] = i;
        }

        dKey0 = std::malloc(keyDataSize);
        dKey1 = std::malloc(keyDataSize);
        dLut  = std::malloc(lutDataSize);
        std::memcpy(dKey0, key.data(), keyDataSize);
        std::memcpy(dKey1, key.data(), keyDataSize);
        std::memcpy(dLut, lut.data(), lutDataSize);

        {
            void* dAlpha = std::malloc(alphaDataSize);
            std::memcpy(dAlpha, alpha.data(), alphaDataSize);

            if (testSpeed)
            {
                st = high_resolution_clock::now();
            }

            ret = FastFss_cpu_onehotKeyGen(dKey1, keyDataSize, dAlpha,
                                           alphaDataSize, bitWidthIn,
                                           sizeof(GroupElement), elementNum);
            CHECK(ret);

            if (testSpeed)
            {
                et        = high_resolution_clock::now();
                auto diff = duration_cast<microseconds>(et - st);
                std::printf("[CUDA ONEHOT KEYGEN] %f ms,  %f us/element\n",
                            (double)diff.count() / 1e3,
                            (double)diff.count() / elementNum);
            }

            std::free(dAlpha);
        }

        {
            void* dSharedOutE0 = std::malloc(sharedOutDataSize);
            void* dSharedOutT0 = std::malloc(sharedOutDataSize);
            void* dMaskedX     = std::malloc(maskedXDataSize);

            std::memcpy(dMaskedX, maskedX.data(), maskedXDataSize);

            if (testSpeed)
            {
                st = high_resolution_clock::now();
            }

            ret = FastFss_cpu_onehotLutEval(
                dSharedOutE0, dSharedOutT0, dMaskedX, maskedXDataSize, dKey0,
                keyDataSize, 0, dLut, lutDataSize, bitWidthIn,
                sizeof(GroupElement), elementNum);
            CHECK(ret);

            if (testSpeed)
            {
                et        = high_resolution_clock::now();
                auto diff = duration_cast<microseconds>(et - st);
                std::printf("[CUDA ONEHOT   EVAL] %f ms,  %f us/element\n",
                            (double)diff.count() / 1e3,
                            (double)diff.count() / elementNum);
            }

            std::memcpy(sharedOutE0.data(), dSharedOutE0, sharedOutDataSize);
            std::memcpy(sharedOutT0.data(), dSharedOutT0, sharedOutDataSize);

            std::free(dSharedOutE0);
            std::free(dSharedOutT0);
            std::free(dMaskedX);
        }
        {
            void* dSharedOutE1 = std::malloc(sharedOutDataSize);
            void* dSharedOutT1 = std::malloc(sharedOutDataSize);
            void* dMaskedX     = std::malloc(maskedXDataSize);

            std::memcpy(dMaskedX, maskedX.data(), maskedXDataSize);

            ret = FastFss_cpu_onehotLutEval(
                dSharedOutE1, dSharedOutT1, dMaskedX, maskedXDataSize, dKey1,
                keyDataSize, 1, dLut, lutDataSize, bitWidthIn,
                sizeof(GroupElement), elementNum);
            CHECK(ret);

            std::memcpy(sharedOutE1.data(), dSharedOutE1, sharedOutDataSize);
            std::memcpy(sharedOutT1.data(), dSharedOutT1, sharedOutDataSize);

            std::free(dSharedOutE1);
            std::free(dSharedOutT1);
            std::free(dMaskedX);
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

        std::free(dKey0);
        std::free(dKey1);
        std::free(dLut);
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
    return 0;
}