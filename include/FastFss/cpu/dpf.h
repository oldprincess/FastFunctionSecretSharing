#ifndef FAST_FSS_CPU_DPF_H
#define FAST_FSS_CPU_DPF_H

#include <FastFss/api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

FAST_FSS_API int FastFss_cpu_dpfKeyGen(void       *key,
                                       size_t      keyDataSize,
                                       const void *alpha,
                                       size_t      alphaDataSize,
                                       const void *beta,
                                       size_t      betaDataSize,
                                       const void *seed0,
                                       size_t      seedDataSize0,
                                       const void *seed1,
                                       size_t      seedDataSize1,
                                       size_t      bitWidthIn,
                                       size_t      bitWidthOut,
                                       size_t      groupSize,
                                       size_t      elementSize,
                                       size_t      elementNum);

FAST_FSS_API int FastFss_cpu_dpfEval(void       *sharedOut,
                                     size_t      sharedOutDataSize,
                                     const void *maskedX,
                                     size_t      maskedXDataSize,
                                     const void *key,
                                     size_t      keyDataSize,
                                     const void *seed,
                                     size_t      seedDataSize,
                                     int         partyId,
                                     size_t      bitWidthIn,
                                     size_t      bitWidthOut,
                                     size_t      groupSize,
                                     size_t      elementSize,
                                     size_t      elementNum,
                                     void       *cache,
                                     size_t      cacheDataSize);

FAST_FSS_API int FastFss_cpu_dpfEvalAll(void       *sharedOut,
                                        size_t      sharedOutDataSize,
                                        const void *maskedX,
                                        size_t      maskedXDataSize,
                                        const void *key,
                                        size_t      keyDataSize,
                                        const void *seed,
                                        size_t      seedDataSize,
                                        int         partyId,
                                        size_t      bitWidthIn,
                                        size_t      bitWidthOut,
                                        size_t      groupSize,
                                        size_t      elementSize,
                                        size_t      elementNum,
                                        void       *cache,
                                        size_t      cacheDataSize);

FAST_FSS_API int FastFss_cpu_dpfEvalMulti(void       *sharedOut,
                                          size_t      sharedOutDataSize,
                                          const void *maskedX,
                                          size_t      maskedXDataSize,
                                          const void *key,
                                          size_t      keyDataSize,
                                          const void *seed,
                                          size_t      seedDataSize,
                                          int         partyId,
                                          const void *point,
                                          size_t      pointDataSize,
                                          size_t      bitWidthIn,
                                          size_t      bitWidthOut,
                                          size_t      groupSize,
                                          size_t      elementSize,
                                          size_t      elementNum,
                                          void       *cache,
                                          size_t      cacheDataSize);

#ifdef __cplusplus
}
#endif

#endif
