#ifndef FAST_FSS_CPU_SPLINE_H
#define FAST_FSS_CPU_SPLINE_H

#include <FastFss/api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

FAST_FSS_API int FastFss_cpu_dcfSplineKeyGen(void       *key,
                                             size_t      keyDataSize,
                                             void       *e,
                                             size_t      eDataSize,
                                             void       *beta,
                                             size_t      betaDataSize,
                                             const void *alpha,
                                             size_t      alphaDataSize,
                                             const void *seed0,
                                             size_t      seedDataSize0,
                                             const void *seed1,
                                             size_t      seedDataSize1,
                                             const void *coefficients,
                                             size_t      coefficientsDataSize,
                                             size_t      degree,
                                             const void *leftEndpoints,
                                             size_t      leftEndpointsDataSize,
                                             const void *rightEndpoints,
                                             size_t      rightEndpointsDataSize,
                                             size_t      intervalNum,
                                             size_t      bitWidthIn,
                                             size_t      bitWidthOut,
                                             size_t      elementSize,
                                             size_t      elementNum);

FAST_FSS_API int FastFss_cpu_dcfSplineEval(void       *sharedOut,
                                           size_t      sharedOutDataSize,
                                           const void *maskedX,
                                           size_t      maskedXDataSize,
                                           const void *key,
                                           size_t      keyDataSize,
                                           const void *sharedE,
                                           size_t      sharedEDataSize,
                                           const void *sharedBeta,
                                           size_t      sharedBetaDataSize,
                                           const void *seed,
                                           size_t      seedDataSize,
                                           int         partyId,
                                           const void *leftEndpoints,
                                           size_t      leftEndpointsDataSize,
                                           const void *rightEndpoints,
                                           size_t      rightEndpointsDataSize,
                                           size_t      intervalNum,
                                           size_t      degree,
                                           size_t      bitWidthIn,
                                           size_t      bitWidthOut,
                                           size_t      elementSize,
                                           size_t      elementNum,
                                           void       *cache,
                                           size_t      cacheDataSize);

#ifdef __cplusplus
}
#endif

#endif
