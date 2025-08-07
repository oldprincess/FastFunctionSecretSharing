
#ifndef FAST_FSS_CPU_GROTTO_H
#define FAST_FSS_CPU_GROTTO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int FastFss_cpu_grottoKeyGen(void*       key,
                             size_t      keyDataSize,
                             const void* alpha,
                             size_t      alphaDataSize,
                             const void* seed0,
                             size_t      seedDataSize0,
                             const void* seed1,
                             size_t      seedDataSize1,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum);

int FastFss_cpu_grottoEval(void*       sharedBooleanOut,
                           const void* maskedX,
                           size_t      maskedXDataSize,
                           const void* key,
                           size_t      keyDataSize,
                           const void* seed,
                           size_t      seedDataSize,
                           bool        equalBound,
                           int         partyId,
                           size_t      bitWidthIn,
                           size_t      elementSize,
                           size_t      elementNum,
                           void*       cache,
                           size_t      cacheDataSize);

int FastFss_cpu_grottoEqEval(void*       sharedBooleanOut,
                             const void* maskedX,
                             size_t      maskedXDataSize,
                             const void* key,
                             size_t      keyDataSize,
                             const void* seed,
                             size_t      seedDataSize,
                             int         partyId,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum,
                             void*       cache,
                             size_t      cacheDataSize);

int FastFss_cpu_grottoEqMultiEval(void*       sharedBooleanOut,
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
                                  size_t      elementSize,
                                  size_t      elementNum,
                                  void*       cache,
                                  size_t      cacheDataSize);

int FastFss_cpu_grottoMICEval(void*       sharedBooleanOut,
                              size_t      sharedBooleanOutDataSize,
                              const void* maskedX,
                              size_t      maskedXDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              const void* seed,
                              size_t      seedDataSize,
                              int         partyId,
                              const void* leftBoundary,
                              size_t      leftBoundaryDataSize,
                              const void* rightBoundary,
                              size_t      rightBoundaryDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cache,
                              size_t      cacheDataSize);

int FastFss_cpu_grottoLutEval(void*       sharedOutE,
                              void*       sharedOutT,
                              const void* maskedX,
                              size_t      maskedXDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              const void* seed,
                              size_t      seedDataSize,
                              int         partyId,
                              const void* lookUpTable,
                              size_t      lookUpTableDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cache,
                              size_t      cacheDataSize);

int FastFss_cpu_grottoLutEval_ex(void*       sharedOutE,
                                 void*       sharedOutT,
                                 const void* maskedX,
                                 size_t      maskedXDataSize,
                                 const void* key,
                                 size_t      keyDataSize,
                                 const void* seed,
                                 size_t      seedDataSize,
                                 int         partyId,
                                 const void* lookUpTable,
                                 size_t      lookUpTableDataSize,
                                 size_t      lutBitWidth,
                                 size_t      bitWidthIn,
                                 size_t      bitWidthOut,
                                 size_t      elementSize,
                                 size_t      elementNum,
                                 void*       cache,
                                 size_t      cacheDataSize);

int FastFss_cpu_grottoLutEval_ex2(void*       sharedOutE,
                                  void*       sharedOutT,
                                  const void* maskedX,
                                  size_t      maskedXDataSize,
                                  const void* key,
                                  size_t      keyDataSize,
                                  const void* seed,
                                  size_t      seedDataSize,
                                  int         partyId,
                                  const void* points,
                                  size_t      pointsDataSize,
                                  const void* lookUpTable,
                                  size_t      lookUpTableDataSize,
                                  size_t      bitWidthIn,
                                  size_t      bitWidthOut,
                                  size_t      elementSize,
                                  size_t      elementNum,
                                  void*       cache,
                                  size_t      cacheDataSize);

int FastFss_cpu_grottoIntervalLutEval(void*       sharedOutE,
                                      void*       sharedOutT,
                                      const void* maskedX,
                                      size_t      maskedXDataSize,
                                      const void* key,
                                      size_t      keyDataSize,
                                      const void* seed,
                                      size_t      seedDataSize,
                                      int         partyId,
                                      const void* leftBoundary,
                                      size_t      leftBoundaryDataSize,
                                      const void* rightBoundary,
                                      size_t      rightBoundaryDataSize,
                                      const void* lookUpTable,
                                      size_t      lookUpTableDataSize,
                                      size_t      bitWidthIn,
                                      size_t      bitWidthOut,
                                      size_t      elementSize,
                                      size_t      elementNum,
                                      void*       cache,
                                      size_t      cacheDataSize);

int FastFss_cpu_grottoKeyZip(void*       zippedKey,
                             size_t      zippedKeyDataSize,
                             const void* key,
                             size_t      keyDataSize,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum);

int FastFss_cpu_grottoKeyUnzip(void*       key,
                               size_t      keyDataSize,
                               const void* zippedKey,
                               size_t      zippedKeyDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum);

int FastFss_cpu_grottoGetKeyDataSize(size_t* keyDataSize,
                                     size_t  bitWidthIn,
                                     size_t  elementSize,
                                     size_t  elementNum);

int FastFss_cpu_grottoGetZippedKeyDataSize(size_t* keyDataSize,
                                           size_t  bitWidthIn,
                                           size_t  elementSize,
                                           size_t  elementNum);

int FastFss_cpu_grottoGetCacheDataSize(size_t* cacheDataSize,
                                       size_t  bitWidthIn,
                                       size_t  elementSize,
                                       size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif