// clang-format off
// nvcc -I include src/cuda/config.cpp src/cuda/dpf.cu test/cuda/dpf.cu -o cuda_dpf.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/config.h>
#include <FastFss/cuda/dpf.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "dpf/TestDpfEval.cuh"
#include "dpf/TestDpfEvalAll.cuh"
#include "dpf/TestDpfMultiEval.cuh"
#include "mt19937.hpp"
#include "uint128_t.h"
#include "utils.cuh"

MT19937Rng rng;

int main()
{
    std::printf("=============================================\n");
    std::printf("============== Test DPF =====================\n");
    std::printf("=============================================\n");

    std::printf("numGridDim = %d\n", FastFss_cuda_getGridDim());

    rng.reseed(7);
    {
        // uint8
        TestDpfEval<std::uint8_t>::run(1, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint8_t>::run(2, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint8_t>::run(3, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint8_t>::run(4, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint8_t>::run(5, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint8_t>::run(6, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint8_t>::run(7, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint8_t>::run(8, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint8_t>::run(8, 8, 4, 1024 - 1, rng);
        TestDpfEval<std::uint8_t>::run(8, 8, 30, 1024 - 1, rng);
        // uint16
        TestDpfEval<std::uint16_t>::run(12, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint16_t>::run(16, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint16_t>::run(16, 8, 30, 1024 - 1, rng);
        // uint32
        TestDpfEval<std::uint32_t>::run(18, 16, 1, 1024 - 1, rng);
        TestDpfEval<std::uint32_t>::run(18, 8, 1, 1024 - 1, rng);
        TestDpfEval<std::uint32_t>::run(18, 8, 30, 1024 - 1, rng);
        // uint64
        TestDpfEval<std::uint64_t>::run(63, 16, 1, 1024 - 1, rng);
        TestDpfEval<std::uint64_t>::run(63, 16, 30, 1024 - 1, rng);
        // uint128
        TestDpfEval<uint128_t>::run(127, 128, 1, 1024 - 1, rng);
        TestDpfEval<uint128_t>::run(128, 127, 1, 1024 - 1, rng);
        TestDpfEval<uint128_t>::run(128, 127, 30, 1024 - 1, rng);
    }
    {
        // uint8
        TestDpfEvalAll<std::uint8_t>::run(1, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint8_t>::run(2, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint8_t>::run(3, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint8_t>::run(4, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint8_t>::run(5, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint8_t>::run(6, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint8_t>::run(7, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint8_t>::run(8, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint8_t>::run(8, 8, 30, 1024 - 1, rng);
        // uint16
        TestDpfEvalAll<std::uint16_t>::run(12, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint16_t>::run(12, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint16_t>::run(12, 8, 30, 1024 - 1, rng);

        // uint32
        TestDpfEvalAll<std::uint32_t>::run(12, 16, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint32_t>::run(12, 8, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint32_t>::run(12, 8, 30, 1024 - 1, rng);
        // uint64
        TestDpfEvalAll<std::uint64_t>::run(12, 16, 1, 1024 - 1, rng);
        TestDpfEvalAll<std::uint64_t>::run(12, 16, 30, 1024 - 1, rng);
        // uint128
        TestDpfEvalAll<uint128_t>::run(12, 128, 1, 1024 - 1, rng);
        TestDpfEvalAll<uint128_t>::run(12, 127, 1, 1024 - 1, rng);
        TestDpfEvalAll<uint128_t>::run(12, 127, 30, 1024 - 1, rng);
    }
    {
        // uint8
        TestDpfMultiEval<std::uint8_t>::run(31, 1, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint8_t>::run(31, 2, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint8_t>::run(31, 3, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint8_t>::run(31, 4, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint8_t>::run(31, 5, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint8_t>::run(31, 6, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint8_t>::run(31, 7, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint8_t>::run(31, 8, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint8_t>::run(31, 8, 8, 4, 1024 - 1, rng);
        TestDpfMultiEval<std::uint8_t>::run(31, 8, 8, 30, 1024 - 1, rng);
        // uint16
        TestDpfMultiEval<std::uint16_t>::run(31, 12, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint16_t>::run(31, 12, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint16_t>::run(31, 12, 8, 4, 1024 - 1, rng);
        TestDpfMultiEval<std::uint16_t>::run(31, 12, 8, 30, 1024 - 1, rng);
        // uint32
        TestDpfMultiEval<std::uint32_t>::run(31, 12, 16, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint32_t>::run(31, 12, 8, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint32_t>::run(31, 12, 8, 4, 1024 - 1, rng);
        TestDpfMultiEval<std::uint32_t>::run(31, 12, 8, 30, 1024 - 1, rng);
        // uint64
        TestDpfMultiEval<std::uint64_t>::run(31, 12, 16, 1, 1024 - 1, rng);
        TestDpfMultiEval<std::uint64_t>::run(31, 12, 16, 4, 1024 - 1, rng);
        TestDpfMultiEval<std::uint64_t>::run(31, 12, 16, 30, 1024 - 1, rng);
        // uint128
        TestDpfMultiEval<uint128_t>::run(31, 12, 128, 1, 1024 - 1, rng);
        TestDpfMultiEval<uint128_t>::run(31, 12, 127, 1, 1024 - 1, rng);
        TestDpfMultiEval<uint128_t>::run(31, 12, 127, 4, 1024 - 1, rng);
        TestDpfMultiEval<uint128_t>::run(31, 12, 127, 30, 1024 - 1, rng);
    }
    return 0;
}