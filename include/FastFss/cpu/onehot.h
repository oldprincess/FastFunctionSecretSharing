#ifndef FAST_FSS_CPU_ONEHOT_H
#define FAST_FSS_CPU_ONEHOT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int FastFss_cpu_onehotKeyGen(void*       key,
                             size_t      keyDataSize,
                             const void* alpha,
                             size_t      alphaDataSize,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum);

int FastFss_cpu_onehotLutEval(void*       sharedOutE,
                              void*       sharedOutT,
                              const void* maskedX,
                              size_t      maskedXDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              int         partyId,
                              const void* lookUpTable,
                              size_t      lookUpTableDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum);

int FastFss_cpu_onehotGetKeyDataSize(size_t* keyDataSize,
                                     size_t  bitWidthIn,
                                     size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif