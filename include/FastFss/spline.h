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

#ifndef FAST_FSS_SPLINE_H
#define FAST_FSS_SPLINE_H

#include <FastFss/api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

FAST_FSS_API int FastFss_dcfSplineKeyZip(void       *zippedKey,
                                         size_t      zippedKeyDataSize,
                                         const void *key,
                                         size_t      keyDataSize,
                                         size_t      degree,
                                         size_t      intervalNum,
                                         size_t      bitWidthIn,
                                         size_t      bitWidthOut,
                                         size_t      elementSize,
                                         size_t      elementNum);

FAST_FSS_API int FastFss_dcfSplineKeyUnzip(void       *key,
                                           size_t      keyDataSize,
                                           const void *zippedKey,
                                           size_t      zippedKeyDataSize,
                                           size_t      degree,
                                           size_t      intervalNum,
                                           size_t      bitWidthIn,
                                           size_t      bitWidthOut,
                                           size_t      elementSize,
                                           size_t      elementNum);

FAST_FSS_API int FastFss_dcfSplineGetCacheDataSize(size_t *cacheDataSize,
                                                   size_t  degree,
                                                   size_t  intervalNum,
                                                   size_t  bitWidthIn,
                                                   size_t  bitWidthOut,
                                                   size_t  elementSize,
                                                   size_t  elementNum);

FAST_FSS_API int FastFss_dcfSplineGetKeyDataSize(size_t *keyDataSize,
                                                 size_t  degree,
                                                 size_t  intervalNum,
                                                 size_t  bitWidthIn,
                                                 size_t  bitWidthOut,
                                                 size_t  elementSize,
                                                 size_t  elementNum);

FAST_FSS_API int FastFss_dcfSplineGetZippedKeyDataSize(size_t *keyDataSize,
                                                       size_t  degree,
                                                       size_t  intervalNum,
                                                       size_t  bitWidthIn,
                                                       size_t  bitWidthOut,
                                                       size_t  elementSize,
                                                       size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif
