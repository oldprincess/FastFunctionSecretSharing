// clang-format off
// g++ -I include src/cpu/config.cpp src/cpu/grotto.cpp test/cpu/grotto.cpp -o cpu_grotto.exe -std=c++17 -maes -fopenmp
// clang-format on
#include <FastFss/cpu/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "grotto/TestGrottoEqEval.hpp"
#include "grotto/TestGrottoEqMultiEval.hpp"
#include "grotto/TestGrottoEval.hpp"
#include "grotto/TestGrottoLutEval.hpp"
#include "grotto/TestGrottoLutEval_ex.hpp"
#include "grotto/TestGrottoLutEval_ex2.hpp"
#include "grotto/TestGrottoMICEval.hpp"
#include "mt19937.hpp"
#include "uint128_t.h"

MT19937Rng rng;

int main()
{
    rng.reseed(time(NULL));
    {
        // uint8
        TestGrottoEval<std::uint8_t>::run(7, 1024 - 1, rng);
        TestGrottoEval<std::uint8_t>::run(8, 1024 - 1, rng);
        // uint16
        TestGrottoEval<std::uint16_t>::run(12, 1024 - 1, rng);
        TestGrottoEval<std::uint16_t>::run(16, 1024 - 1, rng);
        // uint32
        TestGrottoEval<std::uint32_t>::run(18, 1024 - 1, rng);
        TestGrottoEval<std::uint32_t>::run(18, 1024 - 1, rng);
        // uint64
        TestGrottoEval<std::uint64_t>::run(63, 1024 - 1, rng);
        // uint128
        TestGrottoEval<uint128_t>::run(127, 1024 - 1, rng);
        TestGrottoEval<uint128_t>::run(128, 1024 - 1, rng);
    }
    {
        // uint8
        TestGrottoEqEval<std::uint8_t>::run(7, 1024 - 1, rng);
        TestGrottoEqEval<std::uint8_t>::run(8, 1024 - 1, rng);
        // uint16
        TestGrottoEqEval<std::uint16_t>::run(12, 1024 - 1, rng);
        TestGrottoEqEval<std::uint16_t>::run(16, 1024 - 1, rng);
        // uint32
        TestGrottoEqEval<std::uint32_t>::run(18, 1024 - 1, rng);
        TestGrottoEqEval<std::uint32_t>::run(18, 1024 - 1, rng);
        // uint64
        TestGrottoEqEval<std::uint64_t>::run(63, 1024 - 1, rng);
        // uint128
        TestGrottoEqEval<uint128_t>::run(127, 1024 - 1, rng);
        TestGrottoEqEval<uint128_t>::run(128, 1024 - 1, rng);
    }

    {
        // uint8
        TestGrottoEqMultiEval<std::uint8_t>::run(31, 7, 1024 - 1, rng);
        TestGrottoEqMultiEval<std::uint8_t>::run(31, 8, 1024 - 1, rng);
        // uint16
        TestGrottoEqMultiEval<std::uint16_t>::run(31, 12, 1024 - 1, rng);
        TestGrottoEqMultiEval<std::uint16_t>::run(31, 16, 1024 - 1, rng);
        // uint32
        TestGrottoEqMultiEval<std::uint32_t>::run(31, 18, 1024 - 1, rng);
        TestGrottoEqMultiEval<std::uint32_t>::run(31, 18, 1024 - 1, rng);
        // uint64
        TestGrottoEqMultiEval<std::uint64_t>::run(31, 63, 1024 - 1, rng);
        // uint128
        TestGrottoEqMultiEval<uint128_t>::run(31, 127, 1024 - 1, rng);
        TestGrottoEqMultiEval<uint128_t>::run(31, 128, 1024 - 1, rng);
    }
    {
        constexpr int elementNum = 1024 - 1;
        TestGrottoMICEval<std::uint8_t>::run(8, elementNum, {1, 2, 3, 4},
                                             {2, 3, 4, 5}, rng);
        TestGrottoMICEval<std::uint16_t>::run(12, elementNum,               //
                                              {10, 20, 30, 40, 50, 60, 70}, //
                                              {20, 30, 40, 50, 60, 70, 80}, //
                                              rng                           //
        );
        TestGrottoMICEval<std::uint16_t>::run(16, elementNum,               //
                                              {10, 20, 30, 40, 50, 60, 70}, //
                                              {20, 30, 40, 50, 60, 70, 80}, //
                                              rng                           //
        );
        TestGrottoMICEval<std::uint32_t>::run(
            24, elementNum,                      //
            {100, 200, 300, 400, 500, 600, 700}, //
            {200, 300, 400, 500, 600, 700, 800}, //
            rng                                  //
        );
        TestGrottoMICEval<std::uint32_t>::run(
            32, elementNum,                      //
            {100, 200, 300, 400, 500, 600, 700}, //
            {200, 300, 400, 500, 600, 700, 800}, //
            rng                                  //
        );

        TestGrottoMICEval<std::uint64_t>::run(
            32, elementNum,                             //
            {1000, 2000, 3000, 4000, 5000, 6000, 7000}, //
            {2000, 3000, 4000, 5000, 6000, 7000, 8000}, //
            rng                                         //
        );

        TestGrottoMICEval<std::uint64_t>::run(
            48, elementNum,                                    //
            {10000, 20000, 30000, 40000, 50000, 60000, 70000}, //
            {20000, 30000, 40000, 50000, 60000, 70000, 80000}, //
            rng                                                //
        );
    }
    {
        // uint8
        TestGrottoLutEval<std::uint8_t>::run(7, 1024 - 1, rng);
        TestGrottoLutEval<std::uint8_t>::run(8, 1024 - 1, rng);
        // uint16
        TestGrottoLutEval<std::uint16_t>::run(8, 1024 - 1, rng);
        TestGrottoLutEval<std::uint16_t>::run(12, 1024 - 1, rng);
        // uint32
        TestGrottoLutEval<std::uint32_t>::run(8, 1024 - 1, rng);
        TestGrottoLutEval<std::uint32_t>::run(12, 1024 - 1, rng);
        // uint64
        TestGrottoLutEval<std::uint64_t>::run(8, 1024 - 1, rng);
        TestGrottoLutEval<std::uint64_t>::run(12, 1024 - 1, rng);
        // uint64
        TestGrottoLutEval<uint128_t>::run(8, 1024 - 1, rng);
        TestGrottoLutEval<uint128_t>::run(12, 1024 - 1, rng);
    }
    {
        TestGrottoLutEval_ex<std::uint8_t>::run(7, 7, 1024 - 1, rng);
        TestGrottoLutEval_ex<std::uint8_t>::run(7, 8, 1024 - 1, rng);
        TestGrottoLutEval_ex<std::uint16_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLutEval_ex<std::uint32_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLutEval_ex<std::uint64_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLutEval_ex<uint128_t>::run(9, 11, 1024 - 1, rng);
    }

    {
        TestGrottoLutEval_ex2<std::uint8_t>::run(7, 7, 1024 - 1, rng);
        TestGrottoLutEval_ex2<std::uint8_t>::run(7, 8, 1024 - 1, rng);
        TestGrottoLutEval_ex2<std::uint16_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLutEval_ex2<std::uint32_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLutEval_ex2<std::uint64_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLutEval_ex2<uint128_t>::run(9, 11, 1024 - 1, rng);
    }

    return 0;
}