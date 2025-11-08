// clang-format off
// g++ -I include src/cpu/config.cpp src/cpu/dcf.cpp test/cpu/dcf.cpp -o cpu_dcf.exe -std=c++17 -maes -fopenmp
// clang-format on
#include <FastFss/cpu/dcf.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "mt19937.hpp"
#include "uint128_t.h"

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

template <typename T>
class TestDcf
{
public:
    static void run(std::size_t bitWidthIn,
                    std::size_t bitWidthOut,
                    std::size_t groupSize,
                    std::size_t elementNum)
    {
        std::printf("[cpu test] "
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
        std::size_t          outDataSize   = sizeof(T) * elementNum * groupSize;

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
        std::unique_ptr<T[]> y(new T[elementNum]);
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
        int         ret;
        void       *dcfKey = nullptr;
        std::size_t dcfKeyDataSize;

        ret = FastFss_cpu_dcfGetKeyDataSize(&dcfKeyDataSize, bitWidthIn,
                                            bitWidthOut, groupSize, sizeof(T),
                                            elementNum);
        CHECK(ret);
        dcfKey = std::malloc(dcfKeyDataSize);
        CHECK(dcfKey == nullptr);

        {
            ret = FastFss_cpu_dcfKeyGen(dcfKey, dcfKeyDataSize, alpha.get(),
                                        alphaDataSize, beta.get(), betaDataSize,
                                        seed0.get(), seedDataSize0, seed1.get(),
                                        seedDataSize1, bitWidthIn, bitWidthOut,
                                        groupSize, sizeof(T), elementNum);
            CHECK(ret);
        }

        {
            ret = FastFss_cpu_dcfEval(
                sharedOut0.get(), outDataSize, maskedX.get(), maskedXDataSize,
                dcfKey, dcfKeyDataSize, seed0.get(), seedDataSize0, 0,
                bitWidthIn, bitWidthOut, groupSize, sizeof(T), elementNum,
                nullptr, 0);
            CHECK(ret);
        }

        {
            ret = FastFss_cpu_dcfEval(
                sharedOut1.get(), outDataSize, maskedX.get(), maskedXDataSize,
                dcfKey, dcfKeyDataSize, seed1.get(), seedDataSize1, 1,
                bitWidthIn, bitWidthOut, groupSize, sizeof(T), elementNum,
                nullptr, 0);
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
        std::free(dcfKey);

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