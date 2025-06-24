
#ifndef FAST_FSS_CUDA_GROTTO_H
#define FAST_FSS_CUDA_GROTTO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int FastFss_cuda_grottoKeyGen(void*       key,
                              size_t      keyDataSize,
                              const void* alpha,
                              size_t      alphaDataSize,
                              const void* seed0,
                              size_t      seedDataSize0,
                              const void* seed1,
                              size_t      seedDataSize1,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cudaStreamPtr);

int FastFss_cuda_grottoEval(void*       sharedBooleanOut,
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
                            size_t      cacheDataSize,
                            void*       cudaStreamPtr);

int FastFss_cuda_grottoEvalEq(void*       sharedBooleanOut,
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
                              size_t      cacheDataSize,
                              void*       cudaStreamPtr);

int FastFss_cuda_grottoEvalEqMulti(void*       sharedBooleanOut,
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
                                   size_t      cacheDataSize,
                                   void*       cudaStreamPtr);

int FastFss_cuda_grottoMICEval(void*       sharedBooleanOut,
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
                               size_t      cacheDataSize,
                               void*       cudaStreamPtr);

int FastFss_cuda_grottoLutEval(void*       sharedOutE,
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
                               size_t      cacheDataSize,
                               void*       cudaStreamPtr);

int FastFss_cuda_grottoIntervalLutEval(void*       sharedOutE,
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
                                       size_t      cacheDataSize,
                                       void*       cudaStreamPtr);

int FastFss_cuda_grottoKeyZip(void*       zippedKey,
                              size_t      zippedKeyDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cudaStreamPtr);

int FastFss_cuda_grottoKeyUnzip(void*       key,
                                size_t      keyDataSize,
                                const void* zippedKey,
                                size_t      zippedKeyDataSize,
                                size_t      bitWidthIn,
                                size_t      elementSize,
                                size_t      elementNum,
                                void*       cudaStreamPtr);

int FastFss_cuda_grottoGetKeyDataSize(size_t* keyDataSize,
                                      size_t  bitWidthIn,
                                      size_t  elementSize,
                                      size_t  elementNum);

int FastFss_cuda_grottoGetZippedKeyDataSize(size_t* keyDataSize,
                                            size_t  bitWidthIn,
                                            size_t  elementSize,
                                            size_t  elementNum);

int FastFss_cuda_grottoGetCacheDataSize(size_t* cacheDataSize,
                                        size_t  bitWidthIn,
                                        size_t  elementSize,
                                        size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif