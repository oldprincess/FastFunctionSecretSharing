#include <FastFss/cuda/dcf.h>
#include <FastFss/cuda/mic.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "mic.cuh"

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

enum MIC_ERROR_CODE
{
    MIC_SUCCESS                            = 0,
    MIC_RUNTIME_ERROR                      = -1,
    MIC_INVALID_KEY_DATA_SIZE_ERROR        = -2,
    MIC_INVALID_ALPHA_DATA_SIZE_ERROR      = -3,
    MIC_INVALID_SEED_DATA_SIZE_ERROR       = -4,
    MIC_INVALID_BOUNDARY_DATA_SIZE_ERROR   = -5,
    MIC_INVALID_Z_DATA_SIZE_ERROR          = -6,
    MIC_INVALID_SHARED_OUT_DATA_SIZE_ERROR = -7,
    MIC_INVLIAD_MASKED_X_DATA_SIZE_ERROR   = -8,
    MIC_INVALID_BITWIDTH_ERROR             = -9,
    MIC_INVALID_ELEMENT_SIZE_ERROR         = -10,
    MIC_INVALID_PARTY_ID_ERROR             = -11,
};

namespace FastFss::cuda {

template <typename GroupElement>
static int IMPL_FastFss_cuda_dcfMICKeyGen(void**      key,
                                          size_t*     keyDataSize,
                                          void*       z,
                                          size_t      zDataSize,
                                          const void* alpha,
                                          size_t      alphaDataSize,
                                          const void* seed0,
                                          size_t      seedDataSize0,
                                          const void* seed1,
                                          size_t      seedDataSize1,
                                          const void* leftBoundary,
                                          size_t      leftBoundaryDataSize,
                                          const void* rightBoundary,
                                          size_t      rightBoundaryDataSize,
                                          size_t      bitWidthIn,
                                          size_t      bitWidthOut,
                                          size_t      elementSize,
                                          size_t      elementNum) noexcept
{
    assert(sizeof(GroupElement) == elementSize);

    if (alphaDataSize != sizeof(GroupElement) * elementNum)
    {
        return MIC_INVALID_ALPHA_DATA_SIZE_ERROR;
    }
    if (leftBoundaryDataSize != rightBoundaryDataSize ||
        leftBoundaryDataSize % sizeof(GroupElement) != 0)
    {
        return MIC_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }
    std::size_t intervalNum = leftBoundaryDataSize / sizeof(GroupElement);
    if (!(seedDataSize0 == 16 * elementNum && seedDataSize1 == 16 * elementNum))
    {
        return MIC_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthOut <= elementSize * 8))
    {
        return MIC_INVALID_BITWIDTH_ERROR;
    }
    if (zDataSize != intervalNum * elementNum * sizeof(GroupElement))
    {
        return MIC_INVALID_Z_DATA_SIZE_ERROR;
    }
    std::size_t needKeyDataSize = dcfGetKeyDataSize<GroupElement>( //
        bitWidthIn, bitWidthOut, elementNum                        //
    );                                                             //

    bool mallocKey = false;
    if (*key != nullptr)
    {
        if (*keyDataSize != needKeyDataSize)
        {
            return MIC_INVALID_KEY_DATA_SIZE_ERROR;
        }
    }
    else
    {
        CUDA_CHECK(cudaMalloc(key, needKeyDataSize),
                   { return MIC_RUNTIME_ERROR; });
        *keyDataSize = needKeyDataSize;
        mallocKey    = true;
    }

    dcfMICKeyGenKernel<GroupElement><<<256, 512>>>(
        *key, z, alpha, seed0, seed1, leftBoundary, rightBoundary, intervalNum,
        bitWidthIn, bitWidthOut, elementNum);
    CUDA_ERR_CHECK({
        if (mallocKey)
        {
            cudaFree(*key);
            *key = nullptr;
        }
        return MIC_RUNTIME_ERROR;
    });

    return MIC_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cuda_dcfMICEval(void*       sharedOut,
                                        size_t      sharedOutDataSize,
                                        const void* maskedX,
                                        size_t      maskedXDataSize,
                                        const void* key,
                                        size_t      keyDataSize,
                                        const void* sharedZ,
                                        size_t      sharedZDataSize,
                                        const void* seed,
                                        size_t      seedDataSize,
                                        int         partyId,
                                        const void* leftBoundary,
                                        size_t      leftBoundaryDataSize,
                                        const void* rightBoundary,
                                        size_t      rightBoundaryDataSize,
                                        size_t      bitWidthIn,
                                        size_t      bitWidthOut,
                                        size_t      elementSize,
                                        size_t      elementNum) noexcept
{
    assert(sizeof(GroupElement) == elementSize);
    if (leftBoundaryDataSize != rightBoundaryDataSize ||
        leftBoundaryDataSize % sizeof(GroupElement) != 0)
    {
        return MIC_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }
    std::size_t intervalNum = leftBoundaryDataSize / sizeof(GroupElement);
    if (sharedZDataSize != sizeof(GroupElement) * elementNum * intervalNum)
    {
        return MIC_INVALID_Z_DATA_SIZE_ERROR;
    }
    if (sharedOutDataSize != sizeof(GroupElement) * elementNum * intervalNum)
    {
        return MIC_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
    }

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return MIC_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return MIC_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthOut <= elementSize * 8))
    {
        return MIC_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = dcfGetKeyDataSize<GroupElement>( //
        bitWidthIn, bitWidthOut, elementNum                        //
    );                                                             //
    if (keyDataSize != needKeyDataSize)
    {
        return MIC_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return MIC_INVALID_PARTY_ID_ERROR;
    }

    dcfMICEvalKernel<GroupElement><<<256, 512>>>(
        sharedOut, maskedX, key, sharedZ, seed, partyId, leftBoundary,
        rightBoundary, intervalNum, bitWidthIn, bitWidthOut, elementNum);

    CUDA_ERR_CHECK({ return MIC_RUNTIME_ERROR; });
    return MIC_SUCCESS;
}

}; // namespace FastFss::cuda

int FastFss_cuda_dcfMICKeyGen(void**      key,
                              size_t*     keyDataSize,
                              void*       z,
                              size_t      zDataSize,
                              const void* alpha,
                              size_t      alphaDataSize,
                              const void* seed0,
                              size_t      seedDataSize0,
                              const void* seed1,
                              size_t      seedDataSize1,
                              const void* leftBoundary,
                              size_t      leftBoundaryDataSize,
                              const void* rightBoundary,
                              size_t      rightBoundaryDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      elementSize,
                              size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfMICKeyGen<std::uint8_t>(
                key, keyDataSize, z, zDataSize, alpha, alphaDataSize, seed0,
                seedDataSize0, seed1, seedDataSize1, leftBoundary,
                leftBoundaryDataSize, rightBoundary, rightBoundaryDataSize,
                bitWidthIn, bitWidthOut, elementSize, elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfMICKeyGen<std::uint16_t>(
                key, keyDataSize, z, zDataSize, alpha, alphaDataSize, seed0,
                seedDataSize0, seed1, seedDataSize1, leftBoundary,
                leftBoundaryDataSize, rightBoundary, rightBoundaryDataSize,
                bitWidthIn, bitWidthOut, elementSize, elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfMICKeyGen<std::uint32_t>(
                key, keyDataSize, z, zDataSize, alpha, alphaDataSize, seed0,
                seedDataSize0, seed1, seedDataSize1, leftBoundary,
                leftBoundaryDataSize, rightBoundary, rightBoundaryDataSize,
                bitWidthIn, bitWidthOut, elementSize, elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfMICKeyGen<std::uint64_t>(
                key, keyDataSize, z, zDataSize, alpha, alphaDataSize, seed0,
                seedDataSize0, seed1, seedDataSize1, leftBoundary,
                leftBoundaryDataSize, rightBoundary, rightBoundaryDataSize,
                bitWidthIn, bitWidthOut, elementSize, elementNum);
        default: return MIC_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_dcfMICEval(void*       sharedOut,
                            size_t      sharedOutDataSize,
                            const void* maskedX,
                            size_t      maskedXDataSize,
                            const void* key,
                            size_t      keyDataSize,
                            const void* sharedZ,
                            size_t      sharedZDataSize,
                            const void* seed,
                            size_t      seedDataSize,
                            int         partyId,
                            const void* leftBoundary,
                            size_t      leftBoundaryDataSize,
                            const void* rightBoundary,
                            size_t      rightBoundaryDataSize,
                            size_t      bitWidthIn,
                            size_t      bitWidthOut,
                            size_t      elementSize,
                            size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfMICEval<std::uint8_t>(
                sharedOut, sharedOutDataSize, maskedX, maskedXDataSize, key,
                keyDataSize, sharedZ, sharedZDataSize, seed, seedDataSize,
                partyId, leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfMICEval<std::uint16_t>(
                sharedOut, sharedOutDataSize, maskedX, maskedXDataSize, key,
                keyDataSize, sharedZ, sharedZDataSize, seed, seedDataSize,
                partyId, leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfMICEval<std::uint32_t>(
                sharedOut, sharedOutDataSize, maskedX, maskedXDataSize, key,
                keyDataSize, sharedZ, sharedZDataSize, seed, seedDataSize,
                partyId, leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfMICEval<std::uint64_t>(
                sharedOut, sharedOutDataSize, maskedX, maskedXDataSize, key,
                keyDataSize, sharedZ, sharedZDataSize, seed, seedDataSize,
                partyId, leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        default: return MIC_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_dcfMICKeyZip(void**      zippedKey,
                              size_t*     zippedKeyDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      elementSize,
                              size_t      elementNum)
{
    return FastFss_cuda_dcfKeyZip(zippedKey, zippedKeyDataSize, key,
                                  keyDataSize, bitWidthIn, bitWidthOut,
                                  elementSize, elementNum);
}

int FastFss_cuda_dcfMICKeyUnzip(void**      key,
                                size_t*     keyDataSize,
                                const void* zippedKey,
                                size_t      zippedKeyDataSize,
                                size_t      bitWidthIn,
                                size_t      bitWidthOut,
                                size_t      elementSize,
                                size_t      elementNum)
{
    return FastFss_cuda_dcfKeyUnzip(key, keyDataSize, zippedKey,
                                    zippedKeyDataSize, bitWidthIn, bitWidthOut,
                                    elementSize, elementNum);
}

int FastFss_cuda_dcfMICGetKeyDataSize(size_t bitWidthIn,
                                      size_t bitWidthOut,
                                      size_t elementSize,
                                      size_t elementNum)
{
    return FastFss_cuda_dcfGetKeyDataSize(bitWidthIn, bitWidthOut, elementSize,
                                          elementNum);
}

int FastFss_cuda_dcfMICGetZippedKeyDataSize(size_t bitWidthIn,
                                            size_t bitWidthOut,
                                            size_t elementSize,
                                            size_t elementNum)
{
    return FastFss_cuda_dcfGetZippedKeyDataSize(bitWidthIn, bitWidthOut,
                                                elementSize, elementNum);
}