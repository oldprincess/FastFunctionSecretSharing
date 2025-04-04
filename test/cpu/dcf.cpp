// clang-format off
// g++ -I include src/cpu/dcf.cpp test/cpu/dcf.cpp -o cpu_dcf.exe -std=c++17 -maes
// clang-format on
#include <FastFss/cpu/dcf.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "mt19937.hpp"

MT19937Rng rng;

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
class TestDcf
{
public:
    static void run(std::size_t bitWidthIn,
                    std::size_t bitWidthOut,
                    std::size_t elementNum)
    {
        std::printf(
            "[cpu test] elementSize = %2d bitWidthIn = %3d bitWidthOut = "
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
            x[i]       = rng.rand<GroupElement>();
            x[i]       = mod_bits<GroupElement>(x[i], bitWidthIn);
            maskedX[i] = x[i] + alpha[i];
            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
        }

        void*       dcfKey = nullptr;
        std::size_t dcfKeyDataSize;

        {
            int ret1 = FastFss_cpu_dcfKeyGen(
                &dcfKey, &dcfKeyDataSize, alpha.get(), alphaDataSize,
                beta.get(), betaDataSize, seed0.get(), seedDataSize0,
                seed1.get(), seedDataSize1, bitWidthIn, bitWidthOut,
                sizeof(GroupElement), elementNum);
            if (ret1 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_dcfKeyGen ret = %d\n",
                            __LINE__, ret1);
                return;
            }
        }

        {
            int ret2 = FastFss_cpu_dcfEval(
                sharedOut0.get(), maskedX.get(), maskedXDataSize, dcfKey,
                dcfKeyDataSize, seed0.get(), seedDataSize0, 0, bitWidthIn,
                bitWidthOut, sizeof(GroupElement), elementNum);
            if (ret2 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_dcfEval ret = %d\n",
                            __LINE__, ret2);
                std::free(dcfKey);
                return;
            }
        }

        {
            int ret3 = FastFss_cpu_dcfEval(
                sharedOut1.get(), maskedX.get(), maskedXDataSize, dcfKey,
                dcfKeyDataSize, seed1.get(), seedDataSize1, 1, bitWidthIn,
                bitWidthOut, sizeof(GroupElement), elementNum);
            if (ret3 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_dcfEval ret = %d\n",
                            __LINE__, ret3);
                std::free(dcfKey);
                return;
            }
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
        std::free(dcfKey);

        std::puts("  pass");
    }
};

int main()
{
    rng.reseed(7);
    // uint8
    TestDcf<std::uint8_t>::run(1, 8, 1024 - 1);
    TestDcf<std::uint8_t>::run(2, 8, 1024 - 1);
    TestDcf<std::uint8_t>::run(3, 8, 1024 - 1);
    TestDcf<std::uint8_t>::run(4, 8, 1024 - 1);
    TestDcf<std::uint8_t>::run(5, 8, 1024 - 1);
    TestDcf<std::uint8_t>::run(6, 8, 1024 - 1);
    TestDcf<std::uint8_t>::run(7, 8, 1024 - 1);
    TestDcf<std::uint8_t>::run(8, 8, 1024 - 1);
    // uint16
    TestDcf<std::uint16_t>::run(12, 8, 1024 - 1);
    TestDcf<std::uint16_t>::run(16, 8, 1024 - 1);
    // uint32
    TestDcf<std::uint32_t>::run(18, 16, 1024 - 1);
    TestDcf<std::uint32_t>::run(18, 8, 1024 - 1);
    // uint64
    TestDcf<std::uint64_t>::run(63, 16, 1024 - 1);
    return 0;
}