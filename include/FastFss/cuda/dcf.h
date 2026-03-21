// Distributed Comparison Function,
// Function secret sharing for mixed-mode and fixed-point secure computation
#ifndef FAST_FSS_CUDA_DCF_H
#define FAST_FSS_CUDA_DCF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate Distributed Comparison Function (DCF) keys using CUDA
 * acceleration.
 *
 * This function generates a pair of DCF keys for secure two-party computation.
 * The generated keys enable two parties to jointly evaluate a comparison
 * function f(x) = beta if x < alpha, and 0 otherwise, without revealing their
 * inputs. Here beta may be either a scalar or a vector.
 *
 * The implementation leverages GPU (CUDA) acceleration to improve key
 * generation efficiency.
 *
 * @note
 * - The output key buffer must be 16-byte aligned for correct and efficient
 * execution.
 * - If beta == NULL, the function defaults to beta = 1.
 * - If cudaStreamPtr == NULL, the default CUDA stream is used.
 *
 * @param[out] key
 *     Pointer to the output buffer where the generated DCF keys will be stored.
 *     This pointer must be 16-byte aligned.
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
 * @param[in] cudaStreamPtr
 *     Pointer to a CUDA stream used for asynchronous GPU execution.
 *     If NULL, the default CUDA stream is used.
 *
 * @return
 *     Returns 0 on success; non-zero value indicates an error.
 */
int FastFss_cuda_dcfKeyGen(void       *key,
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
                           size_t      elementNum,
                           void       *cudaStreamPtr);

/**
 * @brief Evaluate DCF keys using CUDA acceleration.
 *
 * This function evaluates DCF keys on masked inputs and produces a
 * secret-shared output. The output for each instance is either 0 or beta,
 * where beta may be a scalar or a vector.
 *
 * @note
 * - If cudaStreamPtr == NULL, the default CUDA stream is used.
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
 * @param[in] cudaStreamPtr
 *     Pointer to a CUDA stream used for asynchronous GPU execution.
 *     If NULL, the default CUDA stream is used.
 *
 * @return
 *     Returns 0 on success; non-zero value indicates an error.
 */
int FastFss_cuda_dcfEval(void       *sharedOut,
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
                         size_t      cacheDataSize,
                         void       *cudaStreamPtr);

int FastFss_cuda_dcfKeyZip(void       *zippedKey,
                           size_t      zippedKeyDataSize,
                           const void *key,
                           size_t      keyDataSize,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      groupSize,
                           size_t      elementSize,
                           size_t      elementNum);

int FastFss_cuda_dcfKeyUnzip(void       *key,
                             size_t      keyDataSize,
                             const void *zippedKey,
                             size_t      zippedKeyDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      groupSize,
                             size_t      elementSize,
                             size_t      elementNum);

int FastFss_cuda_dcfGetKeyDataSize(size_t *keyDataSize,
                                   size_t  bitWidthIn,
                                   size_t  bitWidthOut,
                                   size_t  groupSize,
                                   size_t  elementSize,
                                   size_t  elementNum);

int FastFss_cuda_dcfGetZippedKeyDataSize(size_t *keyDataSize,
                                         size_t  bitWidthIn,
                                         size_t  bitWidthOut,
                                         size_t  groupSize,
                                         size_t  elementSize,
                                         size_t  elementNum);

int FastFss_cuda_dcfGetCacheDataSize(size_t *cacheDataSize,
                                     size_t  bitWidthIn,
                                     size_t  bitWidthOut,
                                     size_t  groupSize,
                                     size_t  elementSize,
                                     size_t  elementNum);

int FastFss_cuda_dcfKeyZip(void       *zippedKey,
                           size_t      zippedKeyDataSize,
                           const void *key,
                           size_t      keyDataSize,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      groupSize,
                           size_t      elementSize,
                           size_t      elementNum);

int FastFss_cuda_dcfKeyUnzip(void       *key,
                             size_t      keyDataSize,
                             const void *zippedKey,
                             size_t      zippedKeyDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      groupSize,
                             size_t      elementSize,
                             size_t      elementNum);

int FastFss_cuda_dcfGetKeyDataSize(size_t *keyDataSize,
                                   size_t  bitWidthIn,
                                   size_t  bitWidthOut,
                                   size_t  groupSize,
                                   size_t  elementSize,
                                   size_t  elementNum);

int FastFss_cuda_dcfGetZippedKeyDataSize(size_t *keyDataSize,
                                         size_t  bitWidthIn,
                                         size_t  bitWidthOut,
                                         size_t  groupSize,
                                         size_t  elementSize,
                                         size_t  elementNum);

int FastFss_cuda_dcfGetCacheDataSize(size_t *cacheDataSize,
                                     size_t  bitWidthIn,
                                     size_t  bitWidthOut,
                                     size_t  groupSize,
                                     size_t  elementSize,
                                     size_t  elementNum);

#ifdef __cplusplus
}
#endif

#endif
