#include <FastFss/cuda/grotto.h>
#include <cuda_runtime.h>

#include <cassert>
#include <memory>

#include "grotto.cuh"

#define CUDA_CHECK(expression, do_something)                          \
    if ((expression) != cudaSuccess)                                  \
    {                                                                 \
        std::printf("[error] %s in %s:%d\n",                          \
                    cudaGetErrorString(cudaGetLastError()), __FILE__, \
                    __LINE__);                                        \
        do_something                                                  \
    }

#define CUDA_ERR_CHECK(do_something)                                  \
    if (cudaDeviceSynchronize() != cudaSuccess)                       \
    {                                                                 \
        std::printf("[error] %s in %s:%d\n",                          \
                    cudaGetErrorString(cudaGetLastError()), __FILE__, \
                    __LINE__);                                        \
        do_something                                                  \
    }

// #define CUDA_ERR_CHECK(do_something)                                  \
//     if (cudaPeekAtLastError() != cudaSuccess)                         \
//     {                                                                 \
//         std::printf("[error] %s in %s:%d\n",                          \
//                     cudaGetErrorString(cudaGetLastError()), __FILE__, \
//                     __LINE__);                                        \
//         do_something                                                  \
//     }

enum GROTTO_ERROR_CODE
{
    GROTTO_SUCCESS                            = 0,
    GROTTO_RUNTIME_ERROR                      = -1,
    GROTTO_INVALID_KEY_DATA_SIZE_ERROR        = -2,
    GROTTO_INVALID_ALPHA_DATA_SIZE_ERROR      = -3,
    GROTTO_INVALID_SEED_DATA_SIZE_ERROR       = -5,
    GROTTO_INVALID_BOUNDARY_DATA_SIZE_ERROR   = -6,
    GROTTO_INVALID_Z_DATA_SIZE_ERROR          = -7,
    GROTTO_INVALID_SHARED_OUT_DATA_SIZE_ERROR = -8,
    GROTTO_INVLIAD_MASKED_X_DATA_SIZE_ERROR   = -9,
    GROTTO_INVALID_BITWIDTH_ERROR             = -10,
    GROTTO_INVALID_ELEMENT_SIZE_ERROR         = -11,
    GROTTO_INVALID_PARTY_ID_ERROR             = -12,
};

namespace FastFss::cuda {

template <typename GroupElement>
static int IMPL_FastFss_cuda_grottoKeyGen(void**      key,
                                          size_t*     keyDataSize,
                                          const void* alpha,
                                          size_t      alphaDataSize,
                                          const void* seed0,
                                          size_t      seedDataSize0,
                                          const void* seed1,
                                          size_t      seedDataSize1,
                                          size_t      bitWidthIn,
                                          size_t      elementSize,
                                          size_t      elementNum) noexcept
{
    bool mallocKey = false;
    assert(sizeof(GroupElement) == elementSize);

    if (alphaDataSize != sizeof(GroupElement) * elementNum)
    {
        return GROTTO_INVALID_ALPHA_DATA_SIZE_ERROR;
    }
    if (!(seedDataSize0 == 16 * elementNum && seedDataSize1 == 16 * elementNum))
    {
        return GROTTO_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = grottoGetKeyDataSize<GroupElement>( //
        bitWidthIn, elementNum                                        //
    );                                                                //
    if (*key != nullptr)
    {
        if (*keyDataSize != needKeyDataSize)
        {
            return GROTTO_INVALID_KEY_DATA_SIZE_ERROR;
        }
    }
    else
    {
        CUDA_CHECK(cudaMalloc(key, needKeyDataSize),
                   { return GROTTO_RUNTIME_ERROR; });
        *keyDataSize = needKeyDataSize;
        mallocKey    = true;
    }

    // ===============================================
    int BLOCK_SIZE = 512;
    int GRID_SIZE  = (elementNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (GRID_SIZE > 128 * 32)
    {
        GRID_SIZE = 128 * 32;
    }

    grottoKeyGenKernel<GroupElement><<<GRID_SIZE, BLOCK_SIZE>>>( //
        *key, alpha, seed0, seed1, bitWidthIn, elementNum        //
    );                                                           //
    CUDA_ERR_CHECK({
        if (mallocKey)
        {
            cudaFree(*key);
            *key = nullptr;
        }
        return GROTTO_RUNTIME_ERROR;
    });

    return GROTTO_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cuda_grottoEval(void*       sharedOut,
                                        const void* maskedX,
                                        size_t      maskedXDataSize,
                                        const void* key,
                                        size_t      keyDataSize,
                                        const void* seed,
                                        size_t      seedDataSize,
                                        bool        equalBound,
                                        int         partyId,
                                        size_t      bitWidthIn,
                                        size_t      elementSize,
                                        size_t      elementNum)
{
    assert(sizeof(GroupElement) == elementSize);

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return GROTTO_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return GROTTO_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = grottoGetKeyDataSize<GroupElement>( //
        bitWidthIn, elementNum                                        //
    );                                                                //
    if (keyDataSize != needKeyDataSize)
    {
        return GROTTO_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return GROTTO_INVALID_PARTY_ID_ERROR;
    }

    // ===============================================
    int BLOCK_SIZE = 512;
    int GRID_SIZE  = (elementNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (GRID_SIZE > 128 * 32)
    {
        GRID_SIZE = 128 * 32;
    }

    void*       deviceCache   = nullptr;
    std::size_t needCacheSize = grottoCacheDataSize<GroupElement>( //
        bitWidthIn, GRID_SIZE * BLOCK_SIZE                         //
    );
    CUDA_CHECK(cudaMalloc(&deviceCache, needCacheSize),
               { return GROTTO_RUNTIME_ERROR; });

    grottoEvalKernel<GroupElement><<<GRID_SIZE, BLOCK_SIZE>>>(
        sharedOut, key, maskedX, seed, partyId, bitWidthIn, equalBound,
        deviceCache, elementNum);
    CUDA_ERR_CHECK({
        cudaFree(deviceCache);
        return GROTTO_RUNTIME_ERROR;
    });
    cudaFree(deviceCache);
    return 0;
}

template <typename GroupElement>
static int IMPL_FastFss_cuda_grottoEvalEq(void*       sharedOut,
                                          const void* maskedX,
                                          size_t      maskedXDataSize,
                                          const void* key,
                                          size_t      keyDataSize,
                                          const void* seed,
                                          size_t      seedDataSize,
                                          int         partyId,
                                          size_t      bitWidthIn,
                                          size_t      elementSize,
                                          size_t      elementNum)
{
    assert(sizeof(GroupElement) == elementSize);

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return GROTTO_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return GROTTO_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = grottoGetKeyDataSize<GroupElement>( //
        bitWidthIn, elementNum                                        //
    );                                                                //
    if (keyDataSize != needKeyDataSize)
    {
        return GROTTO_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return GROTTO_INVALID_PARTY_ID_ERROR;
    }

    // ===============================================
    int BLOCK_SIZE = 512;
    int GRID_SIZE  = (elementNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (GRID_SIZE > 128 * 32)
    {
        GRID_SIZE = 128 * 32;
    }

    grottoEvalEqKernel<GroupElement><<<GRID_SIZE, BLOCK_SIZE>>>(       //
        sharedOut, key, maskedX, seed, partyId, bitWidthIn, elementNum //
    );                                                                 //
    CUDA_ERR_CHECK({ return GROTTO_RUNTIME_ERROR; });
    return 0;
}

template <typename GroupElement>
static int IMPL_FastFss_cuda_grottoMICEval(void*       sharedBooleanOut,
                                           size_t      sharedBooleanOutDataSize,
                                           const void* maskedX,
                                           size_t      maskedXDataSize,
                                           const void* key,
                                           size_t      keyDataSize,
                                           const void* seed,
                                           size_t      seedDataSize,
                                           int         partyId,
                                           const void* leftBoundary,
                                           size_t      leftBoundaryDataSize,
                                           const void* rightBoundary,
                                           size_t      rightBoundaryDataSize,
                                           size_t      bitWidthIn,
                                           size_t      elementSize,
                                           size_t      elementNum) noexcept
{
    assert(sizeof(GroupElement) == elementSize);
    if (leftBoundaryDataSize != rightBoundaryDataSize ||
        leftBoundaryDataSize % sizeof(GroupElement) != 0)
    {
        return GROTTO_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }
    std::size_t intervalNum = leftBoundaryDataSize / sizeof(GroupElement);
    if (sharedBooleanOutDataSize !=
        sizeof(GroupElement) * elementNum * intervalNum)
    {
        return GROTTO_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
    }

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return GROTTO_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return GROTTO_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = grottoGetKeyDataSize<GroupElement>( //
        bitWidthIn, elementNum                                        //
    );                                                                //
    if (keyDataSize != needKeyDataSize)
    {
        return GROTTO_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return GROTTO_INVALID_PARTY_ID_ERROR;
    }
    // ===============================
    int BLOCK_SIZE = 512;
    int GRID_SIZE  = (elementNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (GRID_SIZE > 128 * 32)
    {
        GRID_SIZE = 128 * 32;
    }

    void*       deviceCache   = nullptr;
    std::size_t needCacheSize = grottoCacheDataSize<GroupElement>( //
        bitWidthIn, GRID_SIZE * BLOCK_SIZE                         //
    );
    CUDA_CHECK(cudaMalloc(&deviceCache, needCacheSize),
               { return GROTTO_RUNTIME_ERROR; });

    grottoMICEvalKernel<GroupElement><<<GRID_SIZE, BLOCK_SIZE>>>(
        sharedBooleanOut, maskedX, key, seed, partyId, leftBoundary,
        rightBoundary, intervalNum, bitWidthIn, deviceCache, elementNum);
    CUDA_ERR_CHECK({
        cudaFree(deviceCache);
        return GROTTO_RUNTIME_ERROR;
    });
    cudaFree(deviceCache);
    return GROTTO_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cuda_grottoIntervalLutEval( //
    void*       sharedOutE,
    void*       sharedOutT,
    const void* maskedX,
    size_t      maskedXDataSize,
    const void* key,
    size_t      keyDataSize,
    const void* seed,
    size_t      seedDataSize,
    int         partyId,
    const void* leftBoundary,
    size_t      leftBoundaryDataSize,
    const void* rightBoundary,
    size_t      rightBoundaryDataSize,
    const void* lookUpTable,
    size_t      lookUpTableDataSize,
    size_t      bitWidthIn,
    size_t      bitWidthOut,
    size_t      elementSize,
    size_t      elementNum) noexcept
{
    assert(sizeof(GroupElement) == elementSize);
    if (leftBoundaryDataSize != rightBoundaryDataSize ||
        lookUpTableDataSize != rightBoundaryDataSize ||
        leftBoundaryDataSize % sizeof(GroupElement) != 0)
    {
        return GROTTO_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }
    std::size_t intervalNum = leftBoundaryDataSize / sizeof(GroupElement);

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return GROTTO_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return GROTTO_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6 &&
          bitWidthOut <= elementSize * 8))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = grottoGetKeyDataSize<GroupElement>( //
        bitWidthIn, elementNum                                        //
    );                                                                //
    if (keyDataSize != needKeyDataSize)
    {
        return GROTTO_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return GROTTO_INVALID_PARTY_ID_ERROR;
    }

    // ===============================
    int BLOCK_SIZE = 512;
    int GRID_SIZE  = (elementNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (GRID_SIZE > 128 * 32)
    {
        GRID_SIZE = 128 * 32;
    }

    void*       deviceCache   = nullptr;
    std::size_t needCacheSize = grottoCacheDataSize<GroupElement>( //
        bitWidthIn, BLOCK_SIZE * GRID_SIZE                         //
    );
    CUDA_CHECK(cudaMalloc(&deviceCache, needCacheSize),
               { return GROTTO_RUNTIME_ERROR; });

    grottoIntervalLutEvalKernel<GroupElement><<<GRID_SIZE, BLOCK_SIZE>>>(
        sharedOutE, sharedOutT, maskedX, key, seed, partyId, leftBoundary,
        rightBoundary, lookUpTable, intervalNum, bitWidthIn, deviceCache,
        elementNum);
    CUDA_ERR_CHECK({
        cudaFree(deviceCache);
        return GROTTO_RUNTIME_ERROR;
    });
    cudaFree(deviceCache);
    return GROTTO_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cuda_grottoKeyZip(void**      zippedKey,
                                          size_t*     zippedKeyDataSize,
                                          const void* key,
                                          size_t      keyDataSize,
                                          size_t      bitWidthIn,
                                          size_t      elementSize,
                                          size_t      elementNum)
{
    return GROTTO_RUNTIME_ERROR;
}

template <typename GroupElement>
static int IMPL_FastFss_cuda_grottoKeyUnzip(void**      key,
                                            size_t*     keyDataSize,
                                            const void* zippedKey,
                                            size_t      zippedKeyDataSize,
                                            size_t      bitWidthIn,
                                            size_t      elementSize,
                                            size_t      elementNum)
{
    return GROTTO_RUNTIME_ERROR;
}

}; // namespace FastFss::cuda

int FastFss_cuda_grottoKeyGen(void**      key,
                              size_t*     keyDataSize,
                              const void* alpha,
                              size_t      alphaDataSize,
                              const void* seed0,
                              size_t      seedDataSize0,
                              const void* seed1,
                              size_t      seedDataSize1,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyGen<std::uint8_t>(
                key, keyDataSize, alpha, alphaDataSize, seed0, seedDataSize0,
                seed1, seedDataSize1, bitWidthIn, elementSize, elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyGen<std::uint16_t>(
                key, keyDataSize, alpha, alphaDataSize, seed0, seedDataSize0,
                seed1, seedDataSize1, bitWidthIn, elementSize, elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyGen<std::uint32_t>(
                key, keyDataSize, alpha, alphaDataSize, seed0, seedDataSize0,
                seed1, seedDataSize1, bitWidthIn, elementSize, elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyGen<std::uint64_t>(
                key, keyDataSize, alpha, alphaDataSize, seed0, seedDataSize0,
                seed1, seedDataSize1, bitWidthIn, elementSize, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_grottoEval(void*       sharedBooleanOut,
                            const void* maskedX,
                            size_t      maskedXDataSize,
                            const void* key,
                            size_t      keyDataSize,
                            const void* seed,
                            size_t      seedDataSize,
                            bool        equalBound,
                            int         partyId,
                            size_t      bitWidthIn,
                            size_t      elementSize,
                            size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoEval<std::uint8_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, equalBound, partyId, bitWidthIn,
                elementSize, elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoEval<std::uint16_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, equalBound, partyId, bitWidthIn,
                elementSize, elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoEval<std::uint32_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, equalBound, partyId, bitWidthIn,
                elementSize, elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoEval<std::uint64_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, equalBound, partyId, bitWidthIn,
                elementSize, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_grottoEvalEq(void*       sharedBooleanOut,
                              const void* maskedX,
                              size_t      maskedXDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              const void* seed,
                              size_t      seedDataSize,
                              int         partyId,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoEvalEq<std::uint8_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, partyId, bitWidthIn, elementSize,
                elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoEvalEq<std::uint16_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, partyId, bitWidthIn, elementSize,
                elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoEvalEq<std::uint32_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, partyId, bitWidthIn, elementSize,
                elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoEvalEq<std::uint64_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, partyId, bitWidthIn, elementSize,
                elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_grottoMICEval(void*       sharedBooleanOut,
                               size_t      sharedBooleanOutDataSize,
                               const void* maskedX,
                               size_t      maskedXDataSize,
                               const void* key,
                               size_t      keyDataSize,
                               const void* seed,
                               size_t      seedDataSize,
                               int         partyId,
                               const void* leftBoundary,
                               size_t      leftBoundaryDataSize,
                               const void* rightBoundary,
                               size_t      rightBoundaryDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoMICEval<std::uint8_t>(
                sharedBooleanOut, sharedBooleanOutDataSize, maskedX,
                maskedXDataSize, key, keyDataSize, seed, seedDataSize, partyId,
                leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, elementSize, elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoMICEval<
                std::uint16_t>(
                sharedBooleanOut, sharedBooleanOutDataSize, maskedX,
                maskedXDataSize, key, keyDataSize, seed, seedDataSize, partyId,
                leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, elementSize, elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoMICEval<
                std::uint32_t>(
                sharedBooleanOut, sharedBooleanOutDataSize, maskedX,
                maskedXDataSize, key, keyDataSize, seed, seedDataSize, partyId,
                leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, elementSize, elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoMICEval<
                std::uint64_t>(
                sharedBooleanOut, sharedBooleanOutDataSize, maskedX,
                maskedXDataSize, key, keyDataSize, seed, seedDataSize, partyId,
                leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, elementSize, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_grottoIntervalLutEval(void*       sharedOutE,
                                       void*       sharedOutT,
                                       const void* maskedX,
                                       size_t      maskedXDataSize,
                                       const void* key,
                                       size_t      keyDataSize,
                                       const void* seed,
                                       size_t      seedDataSize,
                                       int         partyId,
                                       const void* leftBoundary,
                                       size_t      leftBoundaryDataSize,
                                       const void* rightBoundary,
                                       size_t      rightBoundaryDataSize,
                                       const void* lookUpTable,
                                       size_t      lookUpTableDataSize,
                                       size_t      bitWidthIn,
                                       size_t      bitWidthOut,
                                       size_t      elementSize,
                                       size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoIntervalLutEval<
                std::uint8_t>(sharedOutE, sharedOutT, maskedX, maskedXDataSize,
                              key, keyDataSize, seed, seedDataSize, partyId,
                              leftBoundary, leftBoundaryDataSize, rightBoundary,
                              rightBoundaryDataSize, lookUpTable,
                              lookUpTableDataSize, bitWidthIn, bitWidthOut,
                              elementSize, elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoIntervalLutEval<
                std::uint16_t>(sharedOutE, sharedOutT, maskedX, maskedXDataSize,
                               key, keyDataSize, seed, seedDataSize, partyId,
                               leftBoundary, leftBoundaryDataSize,
                               rightBoundary, rightBoundaryDataSize,
                               lookUpTable, lookUpTableDataSize, bitWidthIn,
                               bitWidthOut, elementSize, elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoIntervalLutEval<
                std::uint32_t>(sharedOutE, sharedOutT, maskedX, maskedXDataSize,
                               key, keyDataSize, seed, seedDataSize, partyId,
                               leftBoundary, leftBoundaryDataSize,
                               rightBoundary, rightBoundaryDataSize,
                               lookUpTable, lookUpTableDataSize, bitWidthIn,
                               bitWidthOut, elementSize, elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoIntervalLutEval<
                std::uint64_t>(sharedOutE, sharedOutT, maskedX, maskedXDataSize,
                               key, keyDataSize, seed, seedDataSize, partyId,
                               leftBoundary, leftBoundaryDataSize,
                               rightBoundary, rightBoundaryDataSize,
                               lookUpTable, lookUpTableDataSize, bitWidthIn,
                               bitWidthOut, elementSize, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_grottoKeyZip(void**      zippedKey,
                              size_t*     zippedKeyDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyZip<std::uint8_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                elementSize, elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyZip<std::uint16_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                elementSize, elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyZip<std::uint32_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                elementSize, elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyZip<std::uint64_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                elementSize, elementNum);

        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_grottoKeyUnzip(void**      key,
                                size_t*     keyDataSize,
                                const void* zippedKey,
                                size_t      zippedKeyDataSize,
                                size_t      bitWidthIn,
                                size_t      elementSize,
                                size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyUnzip<
                std::uint8_t>(key, keyDataSize, zippedKey, zippedKeyDataSize,
                              bitWidthIn, elementSize, elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyUnzip<
                std::uint16_t>(key, keyDataSize, zippedKey, zippedKeyDataSize,
                               bitWidthIn, elementSize, elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyUnzip<
                std::uint32_t>(key, keyDataSize, zippedKey, zippedKeyDataSize,
                               bitWidthIn, elementSize, elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_grottoKeyUnzip<
                std::uint64_t>(key, keyDataSize, zippedKey, zippedKeyDataSize,
                               bitWidthIn, elementSize, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_grottoGetKeyDataSize(size_t bitWidthIn,
                                      size_t elementSize,
                                      size_t elementNum)
{
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    switch (elementSize)
    {
        case 1:
            return (int)FastFss::cuda::grottoGetKeyDataSize<std::uint8_t>(
                bitWidthIn, elementNum);
        case 2:
            return (int)FastFss::cuda::grottoGetKeyDataSize<std::uint16_t>(
                bitWidthIn, elementNum);
        case 4:
            return (int)FastFss::cuda::grottoGetKeyDataSize<std::uint32_t>(
                bitWidthIn, elementNum);
        case 8:
            return (int)FastFss::cuda::grottoGetKeyDataSize<std::uint64_t>(
                bitWidthIn, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_grottoGetZippedKeyDataSize(size_t bitWidthIn,
                                            size_t elementSize,
                                            size_t elementNum)
{
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    switch (elementSize)
    {
        case 1:
            return (int)FastFss::cuda::grottoGetZippedKeyDataSize<std::uint8_t>(
                bitWidthIn, elementNum);
        case 2:
            return (int)
                FastFss::cuda::grottoGetZippedKeyDataSize<std::uint16_t>(
                    bitWidthIn, elementNum);
        case 4:
            return (int)
                FastFss::cuda::grottoGetZippedKeyDataSize<std::uint32_t>(
                    bitWidthIn, elementNum);
        case 8:
            return (int)
                FastFss::cuda::grottoGetZippedKeyDataSize<std::uint64_t>(
                    bitWidthIn, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}