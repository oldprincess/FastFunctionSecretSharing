// clang-format off
// nvcc -I include test/cuda/prng.cu src/cuda/prng.cu -o cuda_prng.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/prng.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "utils.cuh"

int main()
{
    int ret;
    // PRNG Init
    void* prng = FastFss_cuda_prngInit();
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

    ret = FastFss_cuda_prngSetCurrentSeed(prng, seed, counter);
    if (ret != 0)
    {
        std::printf(
            "[%d] [FastFss Error] FastFss_cuda_prngSetCurrentSeed ret = "
            "%d. %s:%d\n",
            __LINE__, ret, __FILE__, __LINE__);
        std::exit(-1);
    }

    // Gen
    int   bufferSize    = 163;
    void* deviceBuffer1 = FastFss::cuda::malloc_gpu(bufferSize);
    void* deviceBuffer2 = FastFss::cuda::malloc_gpu(bufferSize);
    ret = FastFss_cuda_prngGen(prng, deviceBuffer1, 8, 1, bufferSize, nullptr);
    if (ret != 0)
    {
        std::printf(
            "[%d] [FastFss Error] FastFss_cuda_prngGen ret = %d. %s:%d\n",
            __LINE__, ret, __FILE__, __LINE__);
        std::exit(-1);
    }

    ret = FastFss_cuda_prngSetCurrentSeed(prng, seed, counter);
    if (ret != 0)
    {
        std::printf(
            "[%d] [FastFss Error] FastFss_cuda_prngSetCurrentSeed ret = "
            "%d. %s:%d\n",
            __LINE__, ret, __FILE__, __LINE__);
        std::exit(-1);
    }

    ret = FastFss_cuda_prngGen(prng, deviceBuffer2, 8, 1, 163, nullptr);
    if (ret != 0)
    {
        std::printf(
            "[%d] [FastFss Error] FastFss_cuda_prngGen ret = %d. %s:%d\n",
            __LINE__, ret, __FILE__, __LINE__);
        std::exit(-1);
    }

    std::vector<std::uint8_t> buffer1(bufferSize);
    std::vector<std::uint8_t> buffer2(bufferSize);

    FastFss::cuda::memcpy_gpu2cpu(buffer1.data(), deviceBuffer1, bufferSize);
    FastFss::cuda::memcpy_gpu2cpu(buffer2.data(), deviceBuffer2, bufferSize);

    if (std::memcmp(buffer1.data(), buffer2.data(), bufferSize) != 0)
    {
        std::printf("[%d] [FastFss Error] buffer1 != buffer2. %s:%d\n",
                    __LINE__, __FILE__, __LINE__);
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

    // speed
    int speedElementNum  = 1024 * 1024 * 128;
    int speedElementSize = 4;

    auto buffer = FastFss::cuda::malloc_gpu(speedElementNum * speedElementSize);
    auto start  = std::chrono::high_resolution_clock::now();
    ret         = FastFss_cuda_prngGen(prng,                 //
                                       buffer,               //
                                       speedElementSize * 8, //
                                       speedElementSize,     //
                                       speedElementNum, nullptr);
    auto stop   = std::chrono::high_resolution_clock::now();
    FastFss::cuda::free_gpu(buffer);
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

    FastFss::cuda::free_gpu(deviceBuffer1);
    FastFss::cuda::free_gpu(deviceBuffer2);
    FastFss_cuda_prngRelease(prng);

    std::printf("[%d] [FastFss Info] Cuda Test Passed.\n", __LINE__);
    return 0;
}