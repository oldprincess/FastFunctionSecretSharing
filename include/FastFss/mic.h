// clang-format off
/*
 * BibTeX:
 * @inproceedings{boyle2021function,
 *   title        = {Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation},
 *   author       = {Boyle, Elette and Chandran, Nishanth and Gilboa, Niv and
 *                   Gupta, Divya and Ishai, Yuval and Kumar, Nishant and Rathee, Mayank},
 *   booktitle    = {EUROCRYPT},
 *   pages        = {871--900},
 *   year         = {2021},
 *   organization = {Springer}
 * }
 * 
 * Cite: https://eprint.iacr.org/2020/1392
 */
// clang-format on

#ifndef FAST_FSS_MIC_H
#define FAST_FSS_MIC_H

#include <FastFss/api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

FAST_FSS_API int FastFss_dcfMICKeyZip(void       *zippedKey,
                                      size_t      zippedKeyDataSize,
                                      const void *key,
                                      size_t      keyDataSize,
                                      size_t      bitWidthIn,
                                      size_t      bitWidthOut,
                                      size_t      elementSize,
                                      size_t      elementNum);

FAST_FSS_API int FastFss_dcfMICKeyUnzip(void       *key,
                                        size_t      keyDataSize,
                                        const void *zippedKey,
                                        size_t      zippedKeyDataSize,
                                        size_t      bitWidthIn,
                                        size_t      bitWidthOut,
                                        size_t      elementSize,
                                        size_t      elementNum);

FAST_FSS_API int FastFss_dcfMICGetCacheDataSize(size_t *cacheDataSize,
                                                size_t  bitWidthIn,
                                                size_t  bitWidthOut,
                                                size_t  elementSize,
                                                size_t  elementNum);

FAST_FSS_API int FastFss_dcfMICGetKeyDataSize(size_t *keyDataSize,
                                              size_t  bitWidthIn,
                                              size_t  bitWidthOut,
                                              size_t  elementSize,
                                              size_t  elementNum);

FAST_FSS_API int FastFss_dcfMICGetZippedKeyDataSize(size_t *keyDataSize,
                                                    size_t  bitWidthIn,
                                                    size_t  bitWidthOut,
                                                    size_t  elementSize,
                                                    size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif
