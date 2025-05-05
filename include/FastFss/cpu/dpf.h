// Function secret sharing: Improvements and extensions
#ifndef FAST_FSS_CPU_DPF_H
#define FAST_FSS_CPU_DPF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int FastFss_cpu_dpfKeyGen(void*       key,
                          size_t      keyDataSize,
                          const void* alpha,
                          size_t      alphaDataSize,
                          const void* beta,
                          size_t      betaDataSize,
                          const void* seed0,
                          size_t      seedDataSize0,
                          const void* seed1,
                          size_t      seedDataSize1,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      elementSize,
                          size_t      elementNum);

int FastFss_cpu_dpfEval(void*       sharedOut,
                        const void* maskedX,
                        size_t      maskedXDataSize,
                        const void* key,
                        size_t      keyDataSize,
                        const void* seed,
                        size_t      seedDataSize,
                        int         partyId,
                        size_t      bitWidthIn,
                        size_t      bitWidthOut,
                        size_t      elementSize,
                        size_t      elementNum,
                        void*       cache,
                        size_t      cacheDataSize);

int FastFss_cpu_dpfEvalMulti(void*       sharedOut,
                             size_t      sharedOutDataSize,
                             const void* maskedX,
                             size_t      maskedXDataSize,
                             const void* key,
                             size_t      keyDataSize,
                             const void* seed,
                             size_t      seedDataSize,
                             int         partyId,
                             const void* point,
                             size_t      pointDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      elementSize,
                             size_t      elementNum,
                             void*       cache,
                             size_t      cacheDataSize);

int FastFss_cpu_dpfKeyZip(void*       zippedKey,
                          size_t      zippedKeyDataSize,
                          const void* key,
                          size_t      keyDataSize,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      elementSize,
                          size_t      elementNum);

int FastFss_cpu_dpfKeyUnzip(void*       key,
                            size_t      keyDataSize,
                            const void* zippedKey,
                            size_t      zippedKeyDataSize,
                            size_t      bitWidthIn,
                            size_t      bitWidthOut,
                            size_t      elementSize,
                            size_t      elementNum);

int FastFss_cpu_dpfGetKeyDataSize(size_t* keyDataSize,
                                  size_t  bitWidthIn,
                                  size_t  bitWidthOut,
                                  size_t  elementSize,
                                  size_t  elementNum);

int FastFss_cpu_dpfGetZippedKeyDataSize(size_t* keyDataSize,
                                        size_t  bitWidthIn,
                                        size_t  bitWidthOut,
                                        size_t  elementSize,
                                        size_t  elementNum);

int FastFss_cpu_dpfGetCacheDataSize(size_t* cacheDataSize,
                                    size_t  bitWidthIn,
                                    size_t  bitWidthOut,
                                    size_t  elementSize,
                                    size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif