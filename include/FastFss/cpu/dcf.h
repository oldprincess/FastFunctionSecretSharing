// Distributed Comparison Function,
// Function secret sharing for mixed-mode and fixed-point secure computation
#ifndef FAST_FSS_CPU_DCF_H
#define FAST_FSS_CPU_DCF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int FastFss_cpu_dcfKeyGen(void**      key,
                          size_t*     keyDataSize,
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

int FastFss_cpu_dcfEval(void*       sharedOut,
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
                        size_t      elementNum);

int FastFss_cpu_dcfKeyZip(void**      zippedKey,
                          size_t*     zippedKeyDataSize,
                          const void* key,
                          size_t      keyDataSize,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      elementSize,
                          size_t      elementNum);

int FastFss_cpu_dcfKeyUnzip(void**      key,
                            size_t*     keyDataSize,
                            const void* zippedKey,
                            size_t      zippedKeyDataSize,
                            size_t      bitWidthIn,
                            size_t      bitWidthOut,
                            size_t      elementSize,
                            size_t      elementNum);

int FastFss_cpu_dcfGetKeyDataSize(size_t bitWidthIn,
                                  size_t bitWidthOut,
                                  size_t elementSize,
                                  size_t elementNum);

int FastFss_cpu_dcfGetZippedKeyDataSize(size_t bitWidthIn,
                                        size_t bitWidthOut,
                                        size_t elementSize,
                                        size_t elementNum);

#ifdef __cplusplus
}
#endif

#endif