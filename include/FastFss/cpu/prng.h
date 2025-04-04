#ifndef FAST_FSS_CPU_PRNG_H
#define FAST_FSS_CPU_PRNG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* FastFss_cpu_prngInit();

void FastFss_cpu_prngRelease(void* prng);

int FastFss_cpu_prngSetCurrentSeed(void*       prng,
                                   const void* seed128bit,
                                   const void* counter128bit);

int FastFss_cpu_prngGetCurrentSeed(const void* prng,
                                   void*       seed128bit,
                                   void*       counter128bit);

int FastFss_cpu_prngGen(void*  prng,
                        void*  dst,
                        size_t bitWidth,
                        size_t elementSize,
                        size_t elementNum);

#ifdef __cplusplus
}
#endif

#endif