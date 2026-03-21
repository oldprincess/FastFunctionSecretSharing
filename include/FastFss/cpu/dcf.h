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

#ifndef FAST_FSS_CPU_DCF_H
#define FAST_FSS_CPU_DCF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate Distributed Comparison Function (DCF) keys on CPU.
 *
 * This function generates a pair of DCF keys for secure two-party computation.
 * The generated keys enable two parties to jointly evaluate a comparison
 * function f(x) = beta if x < alpha, and 0 otherwise, without revealing their
 * inputs. Here beta may be either a scalar or a vector.
 *
 * @note
 * - If beta == NULL, the function defaults to beta = 1.
 *
 * @param[out] key
 *     Pointer to the output buffer where the generated DCF keys will be stored.
 *
 * @param[in] keyDataSize
 *     Size (in bytes) of the output key buffer.
 *
 * @param[in] alpha
 *     Pointer to the threshold value alpha used in the comparison function.
 *
 * @param[in] alphaDataSize
 *     Size (in bytes) of the alpha input.
 *
 * @param[in] beta
 *     Pointer to the output value beta. The function output is beta when the
 *     condition is satisfied. beta may be a scalar or a vector. If NULL, beta
 *     is implicitly set to 1.
 *
 * @param[in] betaDataSize
 *     Size (in bytes) of the beta input. Ignored if beta == NULL.
 *
 * @param[in] seed0
 *     Pointer to the PRG seed for party 0.
 *
 * @param[in] seedDataSize0
 *     Size (in bytes) of seed0.
 *
 * @param[in] seed1
 *     Pointer to the PRG seed for party 1.
 *
 * @param[in] seedDataSize1
 *     Size (in bytes) of seed1.
 *
 * @param[in] bitWidthIn
 *     Bit width of the input domain (i.e., representation size of x and alpha).
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
 *     Total number of DCF instances to generate.
 *
 * @return
 *     Returns 0 on success; non-zero value indicates an error.
 */
int FastFss_cpu_dcfKeyGen(void       *key,
                          size_t      keyDataSize,
                          const void *alpha,
                          size_t      alphaDataSize,
                          const void *beta,
                          size_t      betaDataSize,
                          const void *seed0,
                          size_t      seedDataSize0,
                          const void *seed1,
                          size_t      seedDataSize1,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      groupSize,
                          size_t      elementSize,
                          size_t      elementNum);

/**
 * @brief Evaluate DCF keys on CPU.
 *
 * This function evaluates DCF keys on masked inputs and produces a
 * secret-shared output. The output for each instance is either 0 or beta,
 * where beta may be a scalar or a vector.
 *
 * @param[out] sharedOut
 *     Pointer to the output buffer for the secret-shared result.
 *
 * @param[in] sharedOutSize
 *     Size (in bytes) of the output buffer.
 *
 * @param[in] maskedX
 *     Pointer to the masked input values.
 *
 * @param[in] maskedXDataSize
 *     Size (in bytes) of the masked input buffer.
 *
 * @param[in] key
 *     Pointer to the DCF key for the selected party.
 *
 * @param[in] keyDataSize
 *     Size (in bytes) of the key buffer.
 *
 * @param[in] seed
 *     Pointer to the PRG seed for the selected party.
 *
 * @param[in] seedDataSize
 *     Size (in bytes) of the seed buffer.
 *
 * @param[in] partyId
 *     Party identifier. Expected values are 0 or 1.
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
 *     Size (in bytes) of each input or output element.
 *
 * @param[in] elementNum
 *     Total number of DCF instances to evaluate.
 *
 * @param[in,out] cache
 *     Temporary workspace buffer used during evaluation.
 *
 * @param[in] cacheDataSize
 *     Size (in bytes) of the workspace buffer.
 *
 * @return
 *     Returns 0 on success; non-zero value indicates an error.
 */
int FastFss_cpu_dcfEval(void       *sharedOut,
                        size_t      sharedOutSize,
                        const void *maskedX,
                        size_t      maskedXDataSize,
                        const void *key,
                        size_t      keyDataSize,
                        const void *seed,
                        size_t      seedDataSize,
                        int         partyId,
                        size_t      bitWidthIn,
                        size_t      bitWidthOut,
                        size_t      groupSize,
                        size_t      elementSize,
                        size_t      elementNum,
                        void       *cache,
                        size_t      cacheDataSize);

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
int FastFss_cpu_dcfKeyZip(void       *zippedKey,
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
int FastFss_cpu_dcfKeyUnzip(void       *key,
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
int FastFss_cpu_dcfGetKeyDataSize(size_t *keyDataSize,
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
int FastFss_cpu_dcfGetZippedKeyDataSize(size_t *keyDataSize,
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
int FastFss_cpu_dcfGetCacheDataSize(size_t *cacheDataSize,
                                    size_t  bitWidthIn,
                                    size_t  bitWidthOut,
                                    size_t  groupSize,
                                    size_t  elementSize,
                                    size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif
