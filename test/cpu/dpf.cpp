// clang-format off
// g++ -I include src/cpu/config.cpp src/cpu/dpf.cpp test/cpu/dpf.cpp -o cpu_dpf.exe -std=c++17 -maes -fopenmp
// clang-format on
#include <FastFss/cpu/dpf.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "mt19937.hpp"

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

        ret = FastFss_cpu_dpfGetKeyDataSize(&dpfKeyDataSize, bitWidthIn,
                                            bitWidthOut, sizeof(GroupElement),
                                            elementNum);
        CHECK(ret);
        dpfKey = std::malloc(dpfKeyDataSize);
        CHECK(dpfKey == nullptr);

        {
            ret = FastFss_cpu_dpfKeyGen(dpfKey, dpfKeyDataSize, alpha.get(),
                                        alphaDataSize, beta.get(), betaDataSize,
                                        seed0.get(), seedDataSize0, seed1.get(),
                                        seedDataSize1, bitWidthIn, bitWidthOut,
                                        sizeof(GroupElement), elementNum);
            CHECK(ret);
        }

        {
            ret = FastFss_cpu_dpfEval(
                sharedOut0.get(), maskedX.get(), maskedXDataSize, dpfKey,
                dpfKeyDataSize, seed0.get(), seedDataSize0, 0, bitWidthIn,
                bitWidthOut, sizeof(GroupElement), elementNum, nullptr, 0);
            CHECK(ret);
        }

        {
            ret = FastFss_cpu_dpfEval(
                sharedOut1.get(), maskedX.get(), maskedXDataSize, dpfKey,
                dpfKeyDataSize, seed1.get(), seedDataSize1, 1, bitWidthIn,
                bitWidthOut, sizeof(GroupElement), elementNum, nullptr, 0);
            CHECK(ret);
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
        std::free(dpfKey);

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
        std::printf("[cpu test] [TestDpfMulti] elementSize = %2d bitWidthIn = "
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

        ret = FastFss_cpu_dpfGetCacheDataSize(&cacheDataSize, bitWidthIn,
                                              bitWidthOut, sizeof(GroupElement),
                                              elementNum);
        CHECK(ret);
        dpfCache = std::malloc(cacheDataSize);
        CHECK(dpfCache == nullptr);

        ret = FastFss_cpu_dpfGetKeyDataSize(&dpfKeyDataSize, bitWidthIn,
                                            bitWidthOut, sizeof(GroupElement),
                                            elementNum);
        CHECK(ret);
        dpfKey = std::malloc(dpfKeyDataSize);
        CHECK(dpfKey == nullptr);

        {
            ret = FastFss_cpu_dpfKeyGen(dpfKey, dpfKeyDataSize, alpha.get(),
                                        alphaDataSize, beta.get(), betaDataSize,
                                        seed0.get(), seedDataSize0, seed1.get(),
                                        seedDataSize1, bitWidthIn, bitWidthOut,
                                        sizeof(GroupElement), elementNum);
            CHECK(ret);
        }

        {
            ret = FastFss_cpu_dpfEvalMulti(
                sharedOut0.get(), sharedOutDataSize, maskedX.get(),
                maskedXDataSize, dpfKey, dpfKeyDataSize, seed0.get(),
                seedDataSize0, 0, point.data(), pointDataSize, bitWidthIn,
                bitWidthOut, sizeof(GroupElement), elementNum, dpfCache, cacheDataSize);
            CHECK(ret);
        }

        {
            ret = FastFss_cpu_dpfEvalMulti(
                sharedOut1.get(), sharedOutDataSize, maskedX.get(),
                maskedXDataSize, dpfKey, dpfKeyDataSize, seed1.get(),
                seedDataSize1, 1, point.data(), pointDataSize, bitWidthIn,
                bitWidthOut, sizeof(GroupElement), elementNum, nullptr, 0);
            CHECK(ret);
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
        std::free(dpfKey);
        std::free(dpfCache);

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

    // uint64
    TestDpfMulti<std::uint16_t>::run(8, 8, 1024 - 1, {0, 2, 4, 8});

    TestDpfMulti<std::uint64_t>::run(18, 18, 1024 - 1, {0, 2, 4, 8});
    return 0;
}