#ifndef TEST_CPU_GROTTO_TEST_GROTTO_HPP
#define TEST_CPU_GROTTO_TEST_GROTTO_HPP

#include <FastFss/cpu/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mt19937.hpp"

template <typename GroupElement>
class TestGrotto
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
    static void run(std::size_t bitWidthIn,
                    std::size_t elementNum,
                    MT19937Rng& rng)
    {
        std::printf("[cpu test GrottoEval] elementSize = %2d bitWidthIn = %3d "
                    "elementNum = %5d",
                    (int)(sizeof(GroupElement)), (int)bitWidthIn,
                    (int)elementNum);

        std::unique_ptr<GroupElement[]> alpha(new GroupElement[elementNum]);
        std::size_t alphaDataSize = sizeof(GroupElement) * elementNum;

        for (std::size_t i = 0; i < elementNum; i++)
        {
            alpha[i] = rng.rand<GroupElement>();
            alpha[i] = mod_bits(alpha[i], bitWidthIn);
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
            x[i]       = mod_bits(x[i], bitWidthIn);
            maskedX[i] = x[i] + alpha[i];
            maskedX[i] = mod_bits(maskedX[i], bitWidthIn);
        }

        void*       grottoKey = nullptr;
        std::size_t grottoKeyDataSize;
        FastFss_cpu_grottoGetKeyDataSize(&grottoKeyDataSize, bitWidthIn,
                                         sizeof(GroupElement), elementNum);
        grottoKey = malloc(grottoKeyDataSize);

        {
            int ret1 = FastFss_cpu_grottoKeyGen(
                grottoKey, grottoKeyDataSize, alpha.get(), alphaDataSize,
                seed0.get(), seedDataSize0, seed1.get(), seedDataSize1,
                bitWidthIn, sizeof(GroupElement), elementNum);
            if (ret1 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_grottoKeyGen ret = %d\n",
                            __LINE__, ret1);
                return;
            }
        }

        {
            int ret2 = FastFss_cpu_grottoEval(
                sharedOut0.get(), maskedX.get(), maskedXDataSize, grottoKey,
                grottoKeyDataSize, seed0.get(), seedDataSize0, false, 0,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr, 0);
            if (ret2 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_grottoEval ret = %d\n",
                            __LINE__, ret2);
                std::free(grottoKey);
                return;
            }
        }

        {
            int ret3 = FastFss_cpu_grottoEval(
                sharedOut1.get(), maskedX.get(), maskedXDataSize, grottoKey,
                grottoKeyDataSize, seed1.get(), seedDataSize1, false, 1,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr, 0);
            if (ret3 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_grottoEval ret = %d\n",
                            __LINE__, ret3);
                std::free(grottoKey);
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
            int ret2 = FastFss_cpu_grottoEval(
                sharedOut0.get(), maskedX.get(), maskedXDataSize, grottoKey,
                grottoKeyDataSize, seed0.get(), seedDataSize0, true, 0,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr, 0);
            if (ret2 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_grottoEval ret = %d\n",
                            __LINE__, ret2);
                std::free(grottoKey);
                return;
            }
        }

        {
            int ret3 = FastFss_cpu_grottoEval(
                sharedOut1.get(), maskedX.get(), maskedXDataSize, grottoKey,
                grottoKeyDataSize, seed1.get(), seedDataSize1, true, 1,
                bitWidthIn, sizeof(GroupElement), elementNum, nullptr, 0);
            if (ret3 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_grottoEval ret = %d\n",
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

        std::free(grottoKey);

        std::puts("  pass");
    }
};

#endif