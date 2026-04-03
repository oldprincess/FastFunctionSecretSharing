#ifndef FAST_FSS_CUDA_OTTT_H
#define FAST_FSS_CUDA_OTTT_H

#include <FastFss/api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

FAST_FSS_API int FastFss_cuda_otttKeyGen(void       *key,
                                         size_t      keyDataSize,
                                         const void *alpha,
                                         size_t      alphaDataSize,
                                         size_t      bitWidthIn,
                                         size_t      elementSize,
                                         size_t      elementNum,
                                         void       *cudaStreamPtr); // cudaStream_t*

FAST_FSS_API int FastFss_cuda_otttLutEval(void       *sharedOutE,
                                          size_t      sharedOutEDataSize,
                                          void       *sharedOutT,
                                          size_t      sharedOutTDataSize,
                                          const void *maskedX,
                                          size_t      maskedXDataSize,
                                          const void *key,
                                          size_t      keyDataSize,
                                          int         partyId,
                                          const void *lookUpTable,
                                          size_t      lookUpTableDataSize,
                                          size_t      bitWidthIn,
                                          size_t      elementSize,
                                          size_t      elementNum,
                                          void       *cudaStreamPtr); // cudaStream_t*

#ifdef __cplusplus
}
#endif

#endif
