// clang-format off
// g++ -I include src/cpu/dcf.cpp src/cpu/mic.cpp test/cpu/mic.cpp -o cpu_mic.exe -std=c++20 -maes
// clang-format on
#include <FastFss/cpu/mic.h>

#include <initializer_list>
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
class Test
{
public:
    static void run(std::size_t                     bitWidthIn,
                    std::size_t                     bitWidthOut,
                    std::size_t                     elementNum,
                    const std::vector<GroupElement> leftBoundary,
                    const std::vector<GroupElement> rightBoundary)
    {
        std::printf(
            "[cpu test] elementSize = %2d bitWidthIn = %3d bitWidthOut = "
            "%3d elementNum = %5d",
            (int)(sizeof(GroupElement)), (int)bitWidthIn, (int)bitWidthOut,
            (int)elementNum);

        std::size_t intervalNum = leftBoundary.size();

        std::vector<GroupElement> x(elementNum);
        std::vector<GroupElement> maskedX(elementNum);
        std::vector<GroupElement> alpha(elementNum);
        std::vector<GroupElement> z(elementNum * intervalNum);
        std::vector<GroupElement> sharedZ0(elementNum * intervalNum);
        std::vector<GroupElement> sharedZ1(elementNum * intervalNum);
        std::vector<GroupElement> sharedOut0(elementNum * intervalNum);
        std::vector<GroupElement> sharedOut1(elementNum * intervalNum);

        std::vector<std::uint8_t> seed0(elementNum * 16);
        std::vector<std::uint8_t> seed1(elementNum * 16);

        void*       dcfMICKey         = nullptr;
        std::size_t dcfMICKeyDataSize = 0;

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            x[i]       = rng.rand<GroupElement>();
            alpha[i]   = rng.rand<GroupElement>();
            maskedX[i] = x[i] + alpha[i];

            x[i]       = mod_bits<GroupElement>(x[i], bitWidthIn);
            alpha[i]   = mod_bits<GroupElement>(alpha[i], bitWidthIn);
            maskedX[i] = mod_bits<GroupElement>(maskedX[i], bitWidthIn);
        }

        int ret0 = FastFss_cpu_dcfMICKeyGen(
            &dcfMICKey, &dcfMICKeyDataSize,              //
            z.data(),                                    //
            z.size() * sizeof(GroupElement),             //
            alpha.data(),                                //
            alpha.size() * sizeof(GroupElement),         //
            seed0.data(),                                //
            seed0.size(),                                //
            seed1.data(),                                //
            seed1.size(),                                //
            leftBoundary.data(),                         //
            leftBoundary.size() * sizeof(GroupElement),  //
            rightBoundary.data(),                        //
            rightBoundary.size() * sizeof(GroupElement), //
            bitWidthIn, bitWidthOut, sizeof(GroupElement), elementNum);
        if (ret0 != 0)
        {
            std::printf("[err] FastFss_cpu_dcfMICKeyGen failed ret = %d\n",
                        ret0);
            return;
        }
        // split share
        for (std::size_t i = 0; i < z.size(); ++i)
        {
            sharedZ0[i] = rng.rand<GroupElement>();
            sharedZ1[i] = z[i] - sharedZ0[i];
        }

        int ret1 = FastFss_cpu_dcfMICEval(
            sharedOut0.data(),                           //
            sharedOut0.size() * sizeof(GroupElement),    //
            maskedX.data(),                              //
            maskedX.size() * sizeof(GroupElement),       //
            dcfMICKey,                                   //
            dcfMICKeyDataSize,                           //
            sharedZ0.data(),                             //
            sharedZ0.size() * sizeof(GroupElement),      //
            seed0.data(),                                //
            seed0.size(),                                //
            0,                                           //
            leftBoundary.data(),                         //
            leftBoundary.size() * sizeof(GroupElement),  //
            rightBoundary.data(),                        //
            rightBoundary.size() * sizeof(GroupElement), //
            bitWidthIn, bitWidthOut, sizeof(GroupElement), elementNum);
        if (ret1 != 0)
        {
            std::free(dcfMICKey);
            std::printf("[err] FastFss_cpu_dcfMICEval failed ret = %d\n", ret1);
            return;
        }

        int ret2 = FastFss_cpu_dcfMICEval(
            sharedOut1.data(),                           //
            sharedOut1.size() * sizeof(GroupElement),    //
            maskedX.data(),                              //
            maskedX.size() * sizeof(GroupElement),       //
            dcfMICKey,                                   //
            dcfMICKeyDataSize,                           //
            sharedZ1.data(),                             //
            sharedZ1.size() * sizeof(GroupElement),      //
            seed1.data(),                                //
            seed1.size(),                                //
            1,                                           //
            leftBoundary.data(),                         //
            leftBoundary.size() * sizeof(GroupElement),  //
            rightBoundary.data(),                        //
            rightBoundary.size() * sizeof(GroupElement), //
            bitWidthIn, bitWidthOut, sizeof(GroupElement), elementNum);
        if (ret2 != 0)
        {
            std::free(dcfMICKey);
            std::printf("[err] FastFss_cpu_dcfMICEval failed ret = %d\n", ret2);
            return;
        }

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            for (std::size_t k = 0; k < intervalNum; ++k)
            {
                GroupElement v = sharedOut0[i * intervalNum + k] +
                                 sharedOut1[i * intervalNum + k];
                v = mod_bits<GroupElement>(v, bitWidthOut);

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
                    std::printf("[err] x = %lld, left = "
                                "%lld, right = %lld, v = %lld\n",
                                (long long)x[i], (long long)leftBoundary[k],
                                (long long)rightBoundary[k], (long long)v);
                    std::exit(-1);
                }
            }
        }

        std::free(dcfMICKey);
        std::puts("  pass");
    }
};

int main()
{
    rng.reseed(7);
    Test<std::uint8_t>::run(4, 8, 1024 - 1, {1, 2, 3, 4}, {2, 3, 4, 5});
    Test<std::uint8_t>::run(8, 8, 1024 - 1, {1, 2, 3, 4}, {2, 3, 4, 5});
    Test<std::uint16_t>::run(12, 8, 1024 - 1,              //
                             {10, 20, 30, 40, 50, 60, 70}, //
                             {20, 30, 40, 50, 60, 70, 80}  //
    );
    Test<std::uint16_t>::run(16, 8, 1024 - 1,              //
                             {10, 20, 30, 40, 50, 60, 70}, //
                             {20, 30, 40, 50, 60, 70, 80}  //
    );
    Test<std::uint32_t>::run(24, 8, 1024 - 1,                     //
                             {100, 200, 300, 400, 500, 600, 700}, //
                             {200, 300, 400, 500, 600, 700, 800}  //
    );
    Test<std::uint32_t>::run(32, 8, 1024 - 1,                     //
                             {100, 200, 300, 400, 500, 600, 700}, //
                             {200, 300, 400, 500, 600, 700, 800}  //
    );

    Test<std::uint64_t>::run(32, 8, 1024 - 1,                            //
                             {1000, 2000, 3000, 4000, 5000, 6000, 7000}, //
                             {2000, 3000, 4000, 5000, 6000, 7000, 8000}  //
    );

    Test<std::uint64_t>::run(
        48, 8, 1024 - 1,                                   //
        {10000, 20000, 30000, 40000, 50000, 60000, 70000}, //
        {20000, 30000, 40000, 50000, 60000, 70000, 80000}  //
    );

    return 0;
}