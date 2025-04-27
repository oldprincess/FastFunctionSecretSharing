// Distributed Comparison Function,
// Function secret sharing for mixed-mode and fixed-point secure computation
#ifndef FAST_FSS_CPU_MIC_H
#define FAST_FSS_CPU_MIC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int FastFss_cpu_dcfMICKeyGen(void*       key,
                             size_t      keyDataSize,
                             void*       z,
                             size_t      zDataSize,
                             const void* alpha,
                             size_t      alphaDataSize,
                             const void* seed0,
                             size_t      seedDataSize0,
                             const void* seed1,
                             size_t      seedDataSize1,
                             const void* leftBoundary,
                             size_t      leftBoundaryDataSize,
                             const void* rightBoundary,
                             size_t      rightBoundaryDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      elementSize,
                             size_t      elementNum);

int FastFss_cpu_dcfMICEval(void*       sharedOut,
                           size_t      sharedOutDataSize,
                           const void* maskedX,
                           size_t      maskedXDataSize,
                           const void* key,
                           size_t      keyDataSize,
                           const void* sharedZ,
                           size_t      sharedZDataSize,
                           const void* seed,
                           size_t      seedDataSize,
                           int         partyId,
                           const void* leftBoundary,
                           size_t      leftBoundaryDataSize,
                           const void* rightBoundary,
                           size_t      rightBoundaryDataSize,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      elementSize,
                           size_t      elementNum,
                           void*       cache,
                           size_t      cacheDataSize);

int FastFss_cpu_dcfMICKeyZip(void*       zippedKey,
                             size_t      zippedKeyDataSize,
                             const void* key,
                             size_t      keyDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      elementSize,
                             size_t      elementNum);

int FastFss_cpu_dcfMICKeyUnzip(void*       key,
                               size_t      keyDataSize,
                               const void* zippedKey,
                               size_t      zippedKeyDataSize,
                               size_t      bitWidthIn,
                               size_t      bitWidthOut,
                               size_t      elementSize,
                               size_t      elementNum);

int FastFss_cpu_dcfMICGetCacheDataSize(size_t* cacheDataSize,
                                       size_t  bitWidthIn,
                                       size_t  bitWidthOut,
                                       size_t  elementSize,
                                       size_t  elementNum);

int FastFss_cpu_dcfMICGetKeyDataSize(size_t* keyDataSize,
                                     size_t  bitWidthIn,
                                     size_t  bitWidthOut,
                                     size_t  elementSize,
                                     size_t  elementNum);

int FastFss_cpu_dcfMICGetZippedKeyDataSize(size_t* keyDataSize,
                                           size_t  bitWidthIn,
                                           size_t  bitWidthOut,
                                           size_t  elementSize,
                                           size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif