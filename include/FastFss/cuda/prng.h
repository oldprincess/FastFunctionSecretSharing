#ifndef FAST_FSS_CUDA_PRNG_H
#define FAST_FSS_CUDA_PRNG_H

#include <FastFss/api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

FAST_FSS_API void* FastFss_cuda_prngInit();

FAST_FSS_API void FastFss_cuda_prngRelease(void* prng);

FAST_FSS_API int FastFss_cuda_prngSetCurrentSeed(void* prng, const void* hostSeed128bit, const void* hostCounter128bit);

FAST_FSS_API int FastFss_cuda_prngGetCurrentSeed(const void* prng, void* hostSeed128bit, void* hostCounter128bit);

FAST_FSS_API int FastFss_cuda_prngGen(void*  prng,
                                      void*  deviceDst,
                                      size_t bitWidth,
                                      size_t elementSize,
                                      size_t elementNum,
                                      void*  cudaStreamPtr);

#ifdef __cplusplus
}
#endif

#endif
