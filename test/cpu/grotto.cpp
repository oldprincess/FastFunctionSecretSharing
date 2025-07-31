// clang-format off
// g++ -I include src/cpu/config.cpp src/cpu/grotto.cpp test/cpu/grotto.cpp -o cpu_grotto.exe -std=c++17 -maes -fopenmp
// clang-format on
#include <FastFss/cpu/grotto.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "grotto/TestGrotto.hpp"
#include "grotto/TestGrottoEq.hpp"
#include "grotto/TestGrottoEqMulti.hpp"
#include "grotto/TestGrottoEvalAllLut.hpp"
#include "grotto/TestGrottoLut_ex.hpp"
#include "grotto/TestGrottoMIC.hpp"
#include "mt19937.hpp"
#include "uint128_t.h"

MT19937Rng rng;

int main()
{
    rng.reseed(time(NULL));
    {
        // uint8
        TestGrottoEq<std::uint8_t>::run(7, 1024 - 1, rng);
        TestGrottoEq<std::uint8_t>::run(8, 1024 - 1, rng);
        // uint16
        TestGrottoEq<std::uint16_t>::run(12, 1024 - 1, rng);
        TestGrottoEq<std::uint16_t>::run(16, 1024 - 1, rng);
        // uint32
        TestGrottoEq<std::uint32_t>::run(18, 1024 - 1, rng);
        TestGrottoEq<std::uint32_t>::run(18, 1024 - 1, rng);
        // uint64
        TestGrottoEq<std::uint64_t>::run(63, 1024 - 1, rng);
        // uint128
        TestGrottoEq<uint128_t>::run(127, 1024 - 1, rng);
        TestGrottoEq<uint128_t>::run(128, 1024 - 1, rng);
    }
    {
        // uint8
        TestGrotto<std::uint8_t>::run(7, 1024 - 1, rng);
        TestGrotto<std::uint8_t>::run(8, 1024 - 1, rng);
        // uint16
        TestGrotto<std::uint16_t>::run(12, 1024 - 1, rng);
        TestGrotto<std::uint16_t>::run(16, 1024 - 1, rng);
        // uint32
        TestGrotto<std::uint32_t>::run(18, 1024 - 1, rng);
        TestGrotto<std::uint32_t>::run(18, 1024 - 1, rng);
        // uint64
        TestGrotto<std::uint64_t>::run(63, 1024 - 1, rng);
        // uint128
        TestGrotto<uint128_t>::run(127, 1024 - 1, rng);
        TestGrotto<uint128_t>::run(128, 1024 - 1, rng);
    }
    {
        constexpr int elementNum = 1024 - 1;
        TestGrottoMIC<std::uint8_t>::run(8, elementNum, {1, 2, 3, 4},
                                         {2, 3, 4, 5}, rng);
        TestGrottoMIC<std::uint16_t>::run(12, elementNum,               //
                                          {10, 20, 30, 40, 50, 60, 70}, //
                                          {20, 30, 40, 50, 60, 70, 80}, //
                                          rng);
        TestGrottoMIC<std::uint16_t>::run(16, elementNum,               //
                                          {10, 20, 30, 40, 50, 60, 70}, //
                                          {20, 30, 40, 50, 60, 70, 80}, //
                                          rng);
        TestGrottoMIC<std::uint32_t>::run(
            24, elementNum,                      //
            {100, 200, 300, 400, 500, 600, 700}, //
            {200, 300, 400, 500, 600, 700, 800}, //
            rng);
        TestGrottoMIC<std::uint32_t>::run(
            32, elementNum,                      //
            {100, 200, 300, 400, 500, 600, 700}, //
            {200, 300, 400, 500, 600, 700, 800}, //
            rng);

        TestGrottoMIC<std::uint64_t>::run(
            32, elementNum,                             //
            {1000, 2000, 3000, 4000, 5000, 6000, 7000}, //
            {2000, 3000, 4000, 5000, 6000, 7000, 8000}, //
            rng);

        TestGrottoMIC<std::uint64_t>::run(
            48, elementNum,                                    //
            {10000, 20000, 30000, 40000, 50000, 60000, 70000}, //
            {20000, 30000, 40000, 50000, 60000, 70000, 80000}, //
            rng);
    }

    {
        constexpr int elementNum = 1024 - 1;
        TestGrottoEqMulti<std::uint8_t>::run(                               //
            8, elementNum, {1, 2, 3, 4}, rng                                //
        );                                                                  //
        TestGrottoEqMulti<std::uint16_t>::run(                              //
            12, elementNum, {10, 20, 30, 40, 50, 60, 70}, rng               //
        );                                                                  //
        TestGrottoEqMulti<std::uint16_t>::run(                              //
            16, elementNum, {10, 20, 30, 40, 50, 60, 70}, rng               //
        );                                                                  //
        TestGrottoEqMulti<std::uint32_t>::run(                              //
            24, elementNum, {100, 200, 300, 400, 500, 600, 700}, rng        //
        );                                                                  //
        TestGrottoEqMulti<std::uint32_t>::run(                              //
            32, elementNum, {100, 200, 300, 400, 500, 600, 700}, rng        //
        );                                                                  //
        TestGrottoEqMulti<std::uint64_t>::run(                              //
            32, elementNum, {1000, 2000, 3000, 4000, 5000, 6000, 7000}, rng //
        );                                                                  //
        TestGrottoEqMulti<std::uint64_t>::run(                              //
            48, elementNum,                                                 //
            {10000, 20000, 30000, 40000, 50000, 60000, 70000}, rng          //
        );                                                                  //
    }

    {
        TestGrottoLutEval<std::uint8_t>::run(7, 1024 - 1, rng);
        TestGrottoLutEval<std::uint8_t>::run(8, 1024 - 1, rng);
        TestGrottoLutEval<std::uint16_t>::run(11, 1024 - 1, rng);
        TestGrottoLutEval<std::uint32_t>::run(11, 1024 - 1, rng);
        TestGrottoLutEval<std::uint64_t>::run(11, 1024 - 1, rng);
        TestGrottoLutEval<uint128_t>::run(11, 1024 - 1, rng);
    }

    {
        TestGrottoLut_ex<std::uint8_t>::run(7, 7, 1024 - 1, rng);
        TestGrottoLut_ex<std::uint8_t>::run(7, 8, 1024 - 1, rng);
        TestGrottoLut_ex<std::uint16_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLut_ex<std::uint32_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLut_ex<std::uint64_t>::run(9, 11, 1024 - 1, rng);
        TestGrottoLut_ex<uint128_t>::run(9, 11, 1024 - 1, rng);

        TestGrottoLut_ex<std::uint64_t>::run(9, 30, 128 * 3072, rng);
    }

    return 0;
}