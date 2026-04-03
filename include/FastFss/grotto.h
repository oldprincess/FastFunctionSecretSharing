// clang-format off
/*
 * BibTeX:
 * @inproceedings{storrier2023grotto,
 *   title={Grotto: Screaming fast (2+ 1)-PC over $\mathbb{Z}_{2^n}$ via (2, 2)-DPFs},
 *   author={Storrier, Kyle and Vadapalli, Adithya and Lyons, Allan and Henry, Ryan},
 *   booktitle={Proceedings of the 2023 ACM SIGSAC conference on computer and communications security},
 *   pages={2143--2157},
 *   year={2023}
 * }
 * 
 * Cite: https://eprint.iacr.org/2023/108
 */
// clang-format on

#ifndef FAST_FSS_GROTTO_H
#define FAST_FSS_GROTTO_H

#include <FastFss/api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

FAST_FSS_API int FastFss_grottoKeyZip(void       *zippedKey,
                                      size_t      zippedKeyDataSize,
                                      const void *key,
                                      size_t      keyDataSize,
                                      size_t      bitWidthIn,
                                      size_t      elementSize,
                                      size_t      elementNum);

FAST_FSS_API int FastFss_grottoKeyUnzip(void       *key,
                                        size_t      keyDataSize,
                                        const void *zippedKey,
                                        size_t      zippedKeyDataSize,
                                        size_t      bitWidthIn,
                                        size_t      elementSize,
                                        size_t      elementNum);

FAST_FSS_API int FastFss_grottoGetKeyDataSize(size_t *keyDataSize,
                                              size_t  bitWidthIn,
                                              size_t  elementSize,
                                              size_t  elementNum);

FAST_FSS_API int FastFss_grottoGetZippedKeyDataSize(size_t *keyDataSize,
                                                    size_t  bitWidthIn,
                                                    size_t  elementSize,
                                                    size_t  elementNum);

FAST_FSS_API int FastFss_grottoGetCacheDataSize(size_t *cacheDataSize,
                                                size_t  bitWidthIn,
                                                size_t  elementSize,
                                                size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif
