// clang-format off
// nvcc benchmark/cuda/grotto_lut_ex2.cu libFastFss.so -o cuda_benchmark_grotto_lut_ex2.exe -std=c++17 -I include
// clang-format on
#include <FastFss/cuda/grotto.h>
#include <FastFss/cuda/prng.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "../ExpressionEvaluator.hpp"
#include "utils.cuh"

using namespace FastFss::cuda;
using namespace std;
using namespace std::chrono;

#define BENCHMARK_CHECK_ERR(ret)                                    \
    if (ret)                                                        \
    {                                                               \
        printf("[ERR] ret = %d. %s:%d\n", ret, __FILE__, __LINE__); \
        exit(-1);                                                   \
    }

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        printf("Usage: %s <N> <bw> <lutBits> <loop>\n", argv[0]);
        return -1;
    }

    ExpressionEvaluator evaluator;

    int N       = evaluator.evaluate(argv[1]);
    int bw      = evaluator.evaluate(argv[2]);
    int lutBits = evaluator.evaluate(argv[3]);
    int loop    = evaluator.evaluate(argv[4]);
    int lutSize = 1 << lutBits;

    printf("N = %d, bw = %d, lutBits = %d, lutSize = %d\n", //
           N, bw, lutBits, lutSize);

    uint8_t prngSeed[16]    = {0};
    uint8_t prngCounter[16] = {0};

    int ret;

    size_t keySize = 0, cacheSize = 0;

    ret = FastFss_cuda_grottoGetKeyDataSize(&keySize, bw, 8, N);
    BENCHMARK_CHECK_ERR(ret);
    ret = FastFss_cuda_grottoGetCacheDataSize(&cacheSize, bw, 8, N);
    BENCHMARK_CHECK_ERR(ret);

    void *d_prng        = FastFss_cuda_prngInit();
    auto  d_alpha       = make_unique_gpu_ptr(sizeof(uint64_t) * N);
    auto  d_maskedX     = make_unique_gpu_ptr(sizeof(uint64_t) * N);
    auto  d_outE        = make_unique_gpu_ptr(sizeof(uint64_t) * N);
    auto  d_outT        = make_unique_gpu_ptr(sizeof(uint64_t) * N);
    auto  d_seed0       = make_unique_gpu_ptr(sizeof(uint8_t) * 16 * N);
    auto  d_seed1       = make_unique_gpu_ptr(sizeof(uint8_t) * 16 * N);
    auto  d_grottoKey   = make_unique_gpu_ptr(keySize);
    auto  d_grottoCache = make_unique_gpu_ptr(cacheSize);
    auto  d_luts        = make_unique_gpu_ptr(sizeof(uint64_t) * lutSize);
    auto  d_points      = make_unique_gpu_ptr(sizeof(uint64_t) * N);

    auto points = unique_ptr<uint64_t[]>(new uint64_t[lutSize]);
    for (int i = 0; i < lutSize; i++)
    {
        points[i] = i;
    }
    memcpy_cpu2gpu(d_points.get(), points.get(), sizeof(uint64_t) * lutSize);

    ret = FastFss_cuda_prngGetCurrentSeed(d_prng, prngSeed, prngCounter);
    BENCHMARK_CHECK_ERR(ret);
    ret = FastFss_cuda_prngGen(d_prng, d_alpha.get(), bw, 4, N, nullptr);
    BENCHMARK_CHECK_ERR(ret);
    ret = FastFss_cuda_prngGen(d_prng, d_maskedX.get(), bw, 4, N, nullptr);
    BENCHMARK_CHECK_ERR(ret);
    ret = FastFss_cuda_prngGen(d_prng, d_seed0.get(), 8, 1, 16 * N, nullptr);
    BENCHMARK_CHECK_ERR(ret);
    ret = FastFss_cuda_prngGen(d_prng, d_seed1.get(), 8, 1, 16 * N, nullptr);
    BENCHMARK_CHECK_ERR(ret);
    ret = FastFss_cuda_prngGen(d_prng, d_luts.get(), bw, 8, lutSize, nullptr);
    BENCHMARK_CHECK_ERR(ret);

    ret = FastFss_cuda_grottoKeyGen(d_grottoKey.get(),        //
                                    keySize,                  //
                                    d_alpha.get(),            //
                                    sizeof(uint64_t) * N,     //
                                    d_seed0.get(),            //
                                    sizeof(uint8_t) * 16 * N, //
                                    d_seed1.get(),            //
                                    sizeof(uint8_t) * 16 * N, //
                                    bw,                       //
                                    8,                        //
                                    N,                        //
                                    nullptr                   //
    );
    BENCHMARK_CHECK_ERR(ret);

    puts("random inputs");
    for (int i = 0; i < loop; i++)
    {
        cudaDeviceSynchronize();
        auto start_time = high_resolution_clock::now();

        ret = FastFss_cuda_grottoLutEval_ex2(d_outE.get(),               //
                                             d_outT.get(),               //
                                             d_maskedX.get(),            //
                                             sizeof(uint64_t) * N,       //
                                             d_grottoKey.get(),          //
                                             keySize,                    //
                                             d_seed0.get(),              //
                                             sizeof(uint8_t) * 16 * N,   //
                                             0,                          //
                                             d_points.get(),             //
                                             sizeof(uint64_t) * lutSize, //
                                             d_luts.get(),               //
                                             sizeof(uint64_t) * lutSize, //
                                             bw,                         //
                                             64,                         //
                                             8,                          //
                                             N,                          //
                                             d_grottoCache.get(),        //
                                             cacheSize,                  //
                                             nullptr);
        cudaDeviceSynchronize();
        auto stop_time = high_resolution_clock::now();
        auto diff      = duration_cast<nanoseconds>(stop_time - start_time);
        printf("[%d/%d] Time: %g s\n", i + 1, loop, diff.count() / 1e9);
    }

    puts("ideal inputs");
    memset_gpu(d_maskedX.get(), 0, sizeof(uint64_t) * N);
    for (int i = 0; i < loop; i++)
    {
        cudaDeviceSynchronize();
        auto start_time = high_resolution_clock::now();

        ret = FastFss_cuda_grottoLutEval_ex2(d_outE.get(),               //
                                             d_outT.get(),               //
                                             d_maskedX.get(),            //
                                             sizeof(uint64_t) * N,       //
                                             d_grottoKey.get(),          //
                                             keySize,                    //
                                             d_seed0.get(),              //
                                             sizeof(uint8_t) * 16 * N,   //
                                             0,                          //
                                             d_points.get(),             //
                                             sizeof(uint64_t) * lutSize, //
                                             d_luts.get(),               //
                                             sizeof(uint64_t) * lutSize, //
                                             bw,                         //
                                             64,                         //
                                             8,                          //
                                             N,                          //
                                             d_grottoCache.get(),        //
                                             cacheSize,                  //
                                             nullptr);
        cudaDeviceSynchronize();
        auto stop_time = high_resolution_clock::now();
        auto diff      = duration_cast<nanoseconds>(stop_time - start_time);
        printf("[%d/%d] Time: %g s\n", i + 1, loop, diff.count() / 1e9);
    }
    FastFss_cuda_prngRelease(d_prng);
    return 0;
}