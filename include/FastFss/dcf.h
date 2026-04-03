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

#ifndef FAST_FSS_DCF_H
#define FAST_FSS_DCF_H

#include <FastFss/api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compress DCF keys into a zipped representation.
 *
 * @param[out] zippedKey
 *     Pointer to the output buffer for the compressed key representation.
 *
 * @param[in] zippedKeyDataSize
 *     Size (in bytes) of the compressed key buffer.
 *
 * @param[in] key
 *     Pointer to the input DCF key buffer.
 *
 * @param[in] keyDataSize
 *     Size (in bytes) of the input key buffer.
 *
 * @param[in] bitWidthIn
 *     Bit width of the input domain.
 *
 * @param[in] bitWidthOut
 *     Bit width of each output element in beta.
 *
 * @param[in] groupSize
 *     Length of each beta vector. Use 1 when beta is a scalar.
 *
 * @param[in] elementSize
 *     Size (in bytes) of each individual element.
 *
 * @param[in] elementNum
 *     Total number of DCF instances represented in the buffer.
 *
 * @return
 *     Returns 0 on success; non-zero value indicates an error.
 */
FAST_FSS_API int FastFss_dcfKeyZip(void       *zippedKey,
                                   size_t      zippedKeyDataSize,
                                   const void *key,
                                   size_t      keyDataSize,
                                   size_t      bitWidthIn,
                                   size_t      bitWidthOut,
                                   size_t      groupSize,
                                   size_t      elementSize,
                                   size_t      elementNum);

/**
 * @brief Restore DCF keys from a zipped representation.
 *
 * @param[out] key
 *     Pointer to the output buffer for the restored DCF keys.
 *
 * @param[in] keyDataSize
 *     Size (in bytes) of the output key buffer.
 *
 * @param[in] zippedKey
 *     Pointer to the compressed key representation.
 *
 * @param[in] zippedKeyDataSize
 *     Size (in bytes) of the compressed key buffer.
 *
 * @param[in] bitWidthIn
 *     Bit width of the input domain.
 *
 * @param[in] bitWidthOut
 *     Bit width of each output element in beta.
 *
 * @param[in] groupSize
 *     Length of each beta vector. Use 1 when beta is a scalar.
 *
 * @param[in] elementSize
 *     Size (in bytes) of each individual element.
 *
 * @param[in] elementNum
 *     Total number of DCF instances represented in the buffer.
 *
 * @return
 *     Returns 0 on success; non-zero value indicates an error.
 */
FAST_FSS_API int FastFss_dcfKeyUnzip(void       *key,
                                     size_t      keyDataSize,
                                     const void *zippedKey,
                                     size_t      zippedKeyDataSize,
                                     size_t      bitWidthIn,
                                     size_t      bitWidthOut,
                                     size_t      groupSize,
                                     size_t      elementSize,
                                     size_t      elementNum);

/**
 * @brief Compute the required buffer size for DCF keys.
 *
 * @param[out] keyDataSize
 *     Pointer to the output location for the required key buffer size.
 *
 * @param[in] bitWidthIn
 *     Bit width of the input domain.
 *
 * @param[in] bitWidthOut
 *     Bit width of each output element in beta.
 *
 * @param[in] groupSize
 *     Length of each beta vector. Use 1 when beta is a scalar.
 *
 * @param[in] elementSize
 *     Size (in bytes) of each individual element.
 *
 * @param[in] elementNum
 *     Total number of DCF instances.
 *
 * @return
 *     Returns 0 on success; non-zero value indicates an error.
 */
FAST_FSS_API int FastFss_dcfGetKeyDataSize(size_t *keyDataSize,
                                           size_t  bitWidthIn,
                                           size_t  bitWidthOut,
                                           size_t  groupSize,
                                           size_t  elementSize,
                                           size_t  elementNum);

/**
 * @brief Compute the required buffer size for zipped DCF keys.
 *
 * @param[out] keyDataSize
 *     Pointer to the output location for the required zipped key buffer size.
 *
 * @param[in] bitWidthIn
 *     Bit width of the input domain.
 *
 * @param[in] bitWidthOut
 *     Bit width of each output element in beta.
 *
 * @param[in] groupSize
 *     Length of each beta vector. Use 1 when beta is a scalar.
 *
 * @param[in] elementSize
 *     Size (in bytes) of each individual element.
 *
 * @param[in] elementNum
 *     Total number of DCF instances.
 *
 * @return
 *     Returns 0 on success; non-zero value indicates an error.
 */
FAST_FSS_API int FastFss_dcfGetZippedKeyDataSize(size_t *keyDataSize,
                                                 size_t  bitWidthIn,
                                                 size_t  bitWidthOut,
                                                 size_t  groupSize,
                                                 size_t  elementSize,
                                                 size_t  elementNum);

/**
 * @brief Compute the required cache size for DCF evaluation.
 *
 * @param[out] cacheDataSize
 *     Pointer to the output location for the required cache buffer size.
 *
 * @param[in] bitWidthIn
 *     Bit width of the input domain.
 *
 * @param[in] bitWidthOut
 *     Bit width of each output element in beta.
 *
 * @param[in] groupSize
 *     Length of each beta vector. Use 1 when beta is a scalar.
 *
 * @param[in] elementSize
 *     Size (in bytes) of each individual element.
 *
 * @param[in] elementNum
 *     Total number of DCF instances.
 *
 * @return
 *     Returns 0 on success; non-zero value indicates an error.
 */
FAST_FSS_API int FastFss_dcfGetCacheDataSize(size_t *cacheDataSize,
                                             size_t  bitWidthIn,
                                             size_t  bitWidthOut,
                                             size_t  groupSize,
                                             size_t  elementSize,
                                             size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif
