// clang-format off
// g++ -I include test/cpu/prng.cpp src/cpu/prng.cpp -o cpu_prng.exe -std=c++17 -maes
// clang-format on
#include <FastFss/cpu/prng.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main()
{
    int ret;
    // PRNG Init
    void* prng = FastFss_cpu_prngInit();
    if (prng == nullptr)
    {
        std::printf("[%d] [FastFss Error]. %s:%d\n", __LINE__, __FILE__,
                    __LINE__);
        std::exit(-1);
    }
    // PRNG Seed
    std::uint8_t seed[16], counter[16];
    std::memset(seed, 1, 16);
    std::memset(counter, 2, 16);

    ret = FastFss_cpu_prngSetCurrentSeed(prng, seed, counter);
    if (ret != 0)
    {
        std::printf("[%d] [FastFss Error] FastFss_cpu_prngSetCurrentSeed ret = "
                    "%d. %s:%d\n",
                    __LINE__, ret, __FILE__, __LINE__);
        std::exit(-1);
    }

    // Gen
    std::uint8_t buffer1[163], buffer2[163];
    ret = FastFss_cpu_prngGen(prng, buffer1, 8, 1, 163);
    if (ret != 0)
    {
        std::printf(
            "[%d] [FastFss Error] FastFss_cpu_prngGen ret = %d. %s:%d\n",
            __LINE__, ret, __FILE__, __LINE__);
        std::exit(-1);
    }

    ret = FastFss_cpu_prngSetCurrentSeed(prng, seed, counter);
    if (ret != 0)
    {
        std::printf("[%d] [FastFss Error] FastFss_cpu_prngSetCurrentSeed ret = "
                    "%d. %s:%d\n",
                    __LINE__, ret, __FILE__, __LINE__);
        std::exit(-1);
    }

    ret = FastFss_cpu_prngGen(prng, buffer2, 8, 1, 163);
    if (ret != 0)
    {
        std::printf(
            "[%d] [FastFss Error] FastFss_cpu_prngGen ret = %d. %s:%d\n",
            __LINE__, ret, __FILE__, __LINE__);
        std::exit(-1);
    }

    std::printf("[%d] Buffer1[: 32] ", __LINE__);
    for (int i = 0; i < 32; i++)
    {
        std::printf("%02x ", buffer1[i]);
    }
    std::printf("\n");
    std::printf("[%d] Buffer2[: 32] ", __LINE__);
    for (int i = 0; i < 32; i++)
    {
        std::printf("%02x ", buffer2[i]);
    }
    std::printf("\n");

    if (std::memcmp(buffer1, buffer2, 163) != 0)
    {
        std::printf("[%d] [FastFss Error] buffer1 != buffer2. %s:%d\n",
                    __LINE__, __FILE__, __LINE__);
        std::exit(-1);
    }

    // speed
    int speedElementNum  = 1024 * 1024 * 128;
    int speedElementSize = 4;

    auto buffer = std::vector<std::uint8_t>(speedElementNum * speedElementSize);
    auto start  = std::chrono::high_resolution_clock::now();
    ret         = FastFss_cpu_prngGen(prng,                 //
                                      buffer.data(),        //
                                      speedElementSize * 8, //
                                      speedElementSize,     //
                                      speedElementNum);
    auto stop   = std::chrono::high_resolution_clock::now();
    if (ret != 0)
    {
        std::printf(
            "[%d] [FastFss Error] FastFss_cpu_prngGen ret = %d. %s:%d\n",
            __LINE__, ret, __FILE__, __LINE__);
        std::exit(-1);
    }
    double timeSeconds =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count() /
        1e6;
    std::size_t processBytes = speedElementNum * speedElementSize;
    std::printf("[%d] [FastFss Info] "                                   //
                "ElementNum = %d, ElementSize = %d Speed: %.2f GB/s.\n", //
                __LINE__,                                                //
                speedElementNum,                                         //
                speedElementSize,                                        //
                processBytes / timeSeconds / 1024.0 / 1024.0 / 1024.0    //
    );

    // Release
    FastFss_cpu_prngRelease(prng);

    std::printf("[%d] [FastFss Info] Cpu Test Passed.\n", __LINE__);
    return 0;
}