#ifndef FAST_FSS_CUDA_PRNG_H
#define FAST_FSS_CUDA_PRNG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* FastFss_cuda_prngInit();

void FastFss_cuda_prngRelease(void* prng);

int FastFss_cuda_prngSetCurrentSeed(void*       prng,
                                    const void* hostSeed128bit,
                                    const void* hostCounter128bit);

int FastFss_cuda_prngGetCurrentSeed(const void* prng,
                                    void*       hostSeed128bit,
                                    void*       hostCounter128bit);

int FastFss_cuda_prngGen(void*  prng,
                         void*  deviceDst,
                         size_t bitWidth,
                         size_t elementSize,
                         size_t elementNum,
                         void*  cudaStreamPtr);

#ifdef __cplusplus
}
#endif

#endif