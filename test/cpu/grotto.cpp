// clang-format off
// g++ -I include src/cpu/grotto.cpp test/cpu/grotto.cpp -o cpu_grotto.exe -std=c++17 -maes
// clang-format on
#include <FastFss/cpu/grotto.h>

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
class TestGrottoEq
{
public:
    static void run(std::size_t bitWidthIn, std::size_t elementNum)
    {
        std::printf("[cpu test GrottoEq] elementSize = %2d bitWidthIn = %3d "
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
            int ret2 = FastFss_cpu_grottoEvalEq(
                sharedOut0.get(), maskedX.get(), maskedXDataSize, grottoKey,
                grottoKeyDataSize, seed0.get(), seedDataSize0, 0, bitWidthIn,
                sizeof(GroupElement), elementNum, nullptr, 0);
            if (ret2 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_grottoEvalEq ret = %d\n",
                            __LINE__, ret2);
                std::free(grottoKey);
                return;
            }
        }

        {
            int ret3 = FastFss_cpu_grottoEvalEq(
                sharedOut1.get(), maskedX.get(), maskedXDataSize, grottoKey,
                grottoKeyDataSize, seed1.get(), seedDataSize1, 1, bitWidthIn,
                sizeof(GroupElement), elementNum, nullptr, 0);
            if (ret3 != 0)
            {
                std::printf("\n[%d] err. FastFss_cpu_grottoEvalEq ret = %d\n",
                            __LINE__, ret3);
                std::free(grottoKey);
                return;
            }
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
        std::free(grottoKey);

        std::puts("  pass");
    }
};

template <typename GroupElement>
class TestGrotto
{
public:
    static void run(std::size_t bitWidthIn, std::size_t elementNum)
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

template <typename GroupElement>
class TestGrottoMIC
{
public:
    static void run(std::size_t                     bitWidthIn,
                    std::size_t                     elementNum,
                    const std::vector<GroupElement> leftBoundary,
                    const std::vector<GroupElement> rightBoundary)
    {
        std::printf("[cpu test GrottoMIC] elementSize = %2d bitWidthIn = %3d "
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

        void*       grottoMICKey = nullptr;
        std::size_t grottoMICKeyDataSize;
        FastFss_cpu_grottoGetKeyDataSize(&grottoMICKeyDataSize, bitWidthIn,
                                         sizeof(GroupElement), elementNum);
        grottoMICKey = malloc(grottoMICKeyDataSize);

        void*       cache = nullptr;
        std::size_t cacheDataSize;
        FastFss_cpu_grottoGetCacheDataSize(&cacheDataSize, bitWidthIn,
                                           sizeof(GroupElement), elementNum);
        cache = malloc(cacheDataSize);

        rng.gen(seed0.data(), seed0.size());
        rng.gen(seed1.data(), seed1.size());

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            x[i]       = rng.rand<GroupElement>();
            alpha[i]   = rng.rand<GroupElement>();
            maskedX[i] = x[i] + alpha[i];

            x[i]       = mod_bits<GroupElement>(x[i], bitWidthIn);
            alpha[i]   = mod_bits<GroupElement>(alpha[i], bitWidthIn);
            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
        }

        int ret0 = FastFss_cpu_grottoKeyGen(
            grottoMICKey, grottoMICKeyDataSize,  //
            alpha.data(),                        //
            alpha.size() * sizeof(GroupElement), //
            seed0.data(),                        //
            seed0.size(),                        //
            seed1.data(),                        //
            seed1.size(),                        //
            bitWidthIn, sizeof(GroupElement), elementNum);
        if (ret0 != 0)
        {
            std::printf("[err] FastFss_cpu_grottoKeyGen failed ret = %d\n",
                        ret0);
            return;
        }

        int ret1 = FastFss_cpu_grottoMICEval(
            sharedOut0.data(),                           //
            sharedOut0.size() * sizeof(GroupElement),    //
            maskedX.data(),                              //
            maskedX.size() * sizeof(GroupElement),       //
            grottoMICKey,                                //
            grottoMICKeyDataSize,                        //
            seed0.data(),                                //
            seed0.size(),                                //
            0,                                           //
            leftBoundary.data(),                         //
            leftBoundary.size() * sizeof(GroupElement),  //
            rightBoundary.data(),                        //
            rightBoundary.size() * sizeof(GroupElement), //
            bitWidthIn, sizeof(GroupElement), elementNum, cache, cacheDataSize);
        if (ret1 != 0)
        {
            std::free(grottoMICKey);
            std::printf("[err] FastFss_cpu_grottoMICEval failed ret = %d\n",
                        ret1);
            return;
        }

        int ret2 = FastFss_cpu_grottoMICEval(
            sharedOut1.data(),                           //
            sharedOut1.size() * sizeof(GroupElement),    //
            maskedX.data(),                              //
            maskedX.size() * sizeof(GroupElement),       //
            grottoMICKey,                                //
            grottoMICKeyDataSize,                        //
            seed1.data(),                                //
            seed1.size(),                                //
            1,                                           //
            leftBoundary.data(),                         //
            leftBoundary.size() * sizeof(GroupElement),  //
            rightBoundary.data(),                        //
            rightBoundary.size() * sizeof(GroupElement), //
            bitWidthIn, sizeof(GroupElement), elementNum, cache, cacheDataSize);
        if (ret2 != 0)
        {
            std::free(grottoMICKey);
            std::printf("[err] FastFss_cpu_grottoMICEval failed ret = %d\n",
                        ret2);
            return;
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

        std::free(grottoMICKey);
        std::free(cache);
        std::puts("  pass");
    }
};

int main()
{
    rng.reseed(time(NULL));
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
    return 0;
}