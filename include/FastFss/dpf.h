// clang-format off
/*
 * BibTeX:
 * @inproceedings{boyle2016function,
 *   title     = {Function Secret Sharing: Improvements and Extensions},
 *   author    = {Boyle, Elette and Gilboa, Niv and Ishai, Yuval},
 *   booktitle = {Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security},
 *   pages     = {1292--1303},
 *   year      = {2016}
 * }
 * 
 * Cite: https://eprint.iacr.org/2018/707
 */
// clang-format on

#ifndef FAST_FSS_DPF_H
#define FAST_FSS_DPF_H

#include <FastFss/api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

FAST_FSS_API int FastFss_dpfKeyZip(void       *zippedKey,
                                   size_t      zippedKeyDataSize,
                                   const void *key,
                                   size_t      keyDataSize,
                                   size_t      bitWidthIn,
                                   size_t      bitWidthOut,
                                   size_t      groupSize,
                                   size_t      elementSize,
                                   size_t      elementNum);

FAST_FSS_API int FastFss_dpfKeyUnzip(void       *key,
                                     size_t      keyDataSize,
                                     const void *zippedKey,
                                     size_t      zippedKeyDataSize,
                                     size_t      bitWidthIn,
                                     size_t      bitWidthOut,
                                     size_t      groupSize,
                                     size_t      elementSize,
                                     size_t      elementNum);

FAST_FSS_API int FastFss_dpfGetKeyDataSize(size_t *keyDataSize,
                                           size_t  bitWidthIn,
                                           size_t  bitWidthOut,
                                           size_t  groupSize,
                                           size_t  elementSize,
                                           size_t  elementNum);

FAST_FSS_API int FastFss_dpfGetZippedKeyDataSize(size_t *keyDataSize,
                                                 size_t  bitWidthIn,
                                                 size_t  bitWidthOut,
                                                 size_t  groupSize,
                                                 size_t  elementSize,
                                                 size_t  elementNum);

FAST_FSS_API int FastFss_dpfGetCacheDataSize(size_t *cacheDataSize,
                                             size_t  bitWidthIn,
                                             size_t  elementSize,
                                             size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif
