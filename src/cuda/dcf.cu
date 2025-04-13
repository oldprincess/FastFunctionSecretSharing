#include <FastFss/cuda/dcf.h>
#include <cuda_runtime.h>

#include <cassert>
#include <memory>

#include "dcf.cuh"

#define CUDA_CHECK(expression, do_something)                          \
    if ((expression) != cudaSuccess)                                  \
    {                                                                 \
        std::printf("[error] %s in %s:%d\n",                          \
                    cudaGetErrorString(cudaGetLastError()), __FILE__, \
                    __LINE__);                                        \
        do_something                                                  \
    }

// #define CUDA_ERR_CHECK(do_something)                                  \
//     if (cudaDeviceSynchronize() != cudaSuccess)                       \
//     {                                                                 \
//         std::printf("[error] %s in %s:%d\n",                          \
//                     cudaGetErrorString(cudaGetLastError()), __FILE__, \
//                     __LINE__);                                        \
//         do_something                                                  \
//     }

#define CUDA_ERR_CHECK(do_something)                                  \
    if (cudaPeekAtLastError() != cudaSuccess)                         \
    {                                                                 \
        std::printf("[error] %s in %s:%d\n",                          \
                    cudaGetErrorString(cudaGetLastError()), __FILE__, \
                    __LINE__);                                        \
        do_something                                                  \
    }

enum DCF_ERROR_CODE
{
    DCF_SUCCESS                            = 0,
    DCF_RUNTIME_ERROR                      = -1,
    DCF_INVALID_KEY_DATA_SIZE_ERROR        = -2,
    DCF_INVALID_ALPHA_DATA_SIZE_ERROR      = -3,
    DCF_INVALID_BETA_DATA_SIZE_ERROR       = -4,
    DCF_INVALID_SEED_DATA_SIZE_ERROR       = -5,
    DCF_INVALID_BOUNDARY_DATA_SIZE_ERROR   = -6,
    DCF_INVALID_Z_DATA_SIZE_ERROR          = -7,
    DCF_INVALID_SHARED_OUT_DATA_SIZE_ERROR = -8,
    DCF_INVLIAD_MASKED_X_DATA_SIZE_ERROR   = -9,
    DCF_INVALID_BITWIDTH_ERROR             = -10,
    DCF_INVALID_ELEMENT_SIZE_ERROR         = -11,
    DCF_INVALID_PARTY_ID_ERROR             = -12,
};

namespace FastFss::cuda {

template <typename GroupElement>
static int IMPL_FastFss_cuda_dcfKeyGen(void**      key,
                                       size_t*     keyDataSize,
                                       const void* alpha,
                                       size_t      alphaDataSize,
                                       const void* beta,
                                       size_t      betaDataSize,
                                       const void* seed0,
                                       size_t      seedDataSize0,
                                       const void* seed1,
                                       size_t      seedDataSize1,
                                       size_t      bitWidthIn,
                                       size_t      bitWidthOut,
                                       size_t      elementSize,
                                       size_t      elementNum) noexcept
{
    assert(sizeof(GroupElement) == elementSize);

    bool mallocKey = false;

    if (alphaDataSize != sizeof(GroupElement) * elementNum)
    {
        return DCF_INVALID_ALPHA_DATA_SIZE_ERROR;
    }
    if (beta != nullptr && betaDataSize != sizeof(GroupElement) * elementNum)
    {
        return DCF_INVALID_BETA_DATA_SIZE_ERROR;
    }
    if (!(seedDataSize0 == 16 * elementNum && seedDataSize1 == 16 * elementNum))
    {
        return DCF_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthOut <= elementSize * 8))
    {
        return DCF_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = dcfGetKeyDataSize<GroupElement>( //
        bitWidthIn, bitWidthOut, elementNum                        //
    );                                                             //
    if (*key != nullptr)
    {
        if (*keyDataSize != needKeyDataSize)
        {
            return DCF_INVALID_KEY_DATA_SIZE_ERROR;
        }
    }
    else
    {
        CUDA_CHECK(cudaMalloc(key, needKeyDataSize),
                   { return DCF_RUNTIME_ERROR; });
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

    dcfKeyGenKernel<GroupElement><<<GRID_SIZE, BLOCK_SIZE>>>(
        *key, alpha, beta, seed0, seed1, bitWidthIn, bitWidthOut, elementNum);
    CUDA_ERR_CHECK({
        if (mallocKey)
        {
            cudaFree(*key);
            *key = nullptr;
        }
        return DCF_RUNTIME_ERROR;
    });

    return DCF_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cuda_dcfEval(void*       sharedOut,
                                     const void* maskedX,
                                     size_t      maskedXDataSize,
                                     const void* key,
                                     size_t      keyDataSize,
                                     const void* seed,
                                     size_t      seedDataSize,
                                     int         partyId,
                                     size_t      bitWidthIn,
                                     size_t      bitWidthOut,
                                     size_t      elementSize,
                                     size_t      elementNum)
{
    assert(sizeof(GroupElement) == elementSize);

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return DCF_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return DCF_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthOut <= elementSize * 8))
    {
        return DCF_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = dcfGetKeyDataSize<GroupElement>( //
        bitWidthIn, bitWidthOut, elementNum                        //
    );                                                             //
    if (keyDataSize != needKeyDataSize)
    {
        return DCF_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return DCF_INVALID_PARTY_ID_ERROR;
    }

    // ===============================================
    int BLOCK_SIZE = 512;
    int GRID_SIZE  = (elementNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (GRID_SIZE > 128 * 32)
    {
        GRID_SIZE = 128 * 32;
    }

    dcfEvalKernel<GroupElement>
        <<<GRID_SIZE, BLOCK_SIZE>>>(sharedOut, maskedX, key, seed, partyId,
                                    bitWidthIn, bitWidthOut, elementNum);
    CUDA_ERR_CHECK({ return DCF_RUNTIME_ERROR; });
    return 0;
}

template <typename GroupElement>
static int IMPL_FastFss_cuda_dcfKeyZip(void**      zippedKey,
                                       size_t*     zippedKeyDataSize,
                                       const void* key,
                                       size_t      keyDataSize,
                                       size_t      bitWidthIn,
                                       size_t      bitWidthOut,
                                       size_t      elementSize,
                                       size_t      elementNum)
{
    return DCF_RUNTIME_ERROR;
}

template <typename GroupElement>
static int IMPL_FastFss_cuda_dcfKeyUnzip(void**      key,
                                         size_t*     keyDataSize,
                                         const void* zippedKey,
                                         size_t      zippedKeyDataSize,
                                         size_t      bitWidthIn,
                                         size_t      bitWidthOut,
                                         size_t      elementSize,
                                         size_t      elementNum)
{
    return DCF_RUNTIME_ERROR;
}

}; // namespace FastFss::cuda

int FastFss_cuda_dcfKeyGen(void**      key,
                           size_t*     keyDataSize,
                           const void* alpha,
                           size_t      alphaDataSize,
                           const void* beta,
                           size_t      betaDataSize,
                           const void* seed0,
                           size_t      seedDataSize0,
                           const void* seed1,
                           size_t      seedDataSize1,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      elementSize,
                           size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyGen<std::uint8_t>(
                key, keyDataSize, alpha, alphaDataSize, beta, betaDataSize,
                seed0, seedDataSize0, seed1, seedDataSize1, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyGen<std::uint16_t>(
                key, keyDataSize, alpha, alphaDataSize, beta, betaDataSize,
                seed0, seedDataSize0, seed1, seedDataSize1, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyGen<std::uint32_t>(
                key, keyDataSize, alpha, alphaDataSize, beta, betaDataSize,
                seed0, seedDataSize0, seed1, seedDataSize1, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyGen<std::uint64_t>(
                key, keyDataSize, alpha, alphaDataSize, beta, betaDataSize,
                seed0, seedDataSize0, seed1, seedDataSize1, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_dcfEval(void*       sharedOut,
                         const void* maskedX,
                         size_t      maskedXDataSize,
                         const void* key,
                         size_t      keyDataSize,
                         const void* seed,
                         size_t      seedDataSize,
                         int         partyId,
                         size_t      bitWidthIn,
                         size_t      bitWidthOut,
                         size_t      elementSize,
                         size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfEval<std::uint8_t>(
                sharedOut, maskedX, maskedXDataSize, key, keyDataSize, seed,
                seedDataSize, partyId, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfEval<std::uint16_t>(
                sharedOut, maskedX, maskedXDataSize, key, keyDataSize, seed,
                seedDataSize, partyId, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfEval<std::uint32_t>(
                sharedOut, maskedX, maskedXDataSize, key, keyDataSize, seed,
                seedDataSize, partyId, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfEval<std::uint64_t>(
                sharedOut, maskedX, maskedXDataSize, key, keyDataSize, seed,
                seedDataSize, partyId, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_dcfKeyZip(void**      zippedKey,
                           size_t*     zippedKeyDataSize,
                           const void* key,
                           size_t      keyDataSize,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      elementSize,
                           size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyZip<std::uint8_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyZip<std::uint16_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyZip<std::uint32_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyZip<std::uint64_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);

        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_dcfKeyUnzip(void**      key,
                             size_t*     keyDataSize,
                             const void* zippedKey,
                             size_t      zippedKeyDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      elementSize,
                             size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyUnzip<std::uint8_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 2:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyUnzip<std::uint16_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 4:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyUnzip<std::uint32_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 8:
            return FastFss::cuda::IMPL_FastFss_cuda_dcfKeyUnzip<std::uint64_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_dcfGetKeyDataSize(size_t bitWidthIn,
                                   size_t bitWidthOut,
                                   size_t elementSize,
                                   size_t elementNum)
{
    switch (elementSize)
    {
        case 1:
            return (int)FastFss::cuda::dcfGetKeyDataSize<std::uint8_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 2:
            return (int)FastFss::cuda::dcfGetKeyDataSize<std::uint16_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 4:
            return (int)FastFss::cuda::dcfGetKeyDataSize<std::uint32_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 8:
            return (int)FastFss::cuda::dcfGetKeyDataSize<std::uint64_t>(
                bitWidthIn, bitWidthOut, elementNum);
        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cuda_dcfGetZippedKeyDataSize(size_t bitWidthIn,
                                         size_t bitWidthOut,
                                         size_t elementSize,
                                         size_t elementNum)
{
    switch (elementSize)
    {
        case 1:
            return (int)FastFss::cuda::dcfGetZippedKeyDataSize<std::uint8_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 2:
            return (int)FastFss::cuda::dcfGetZippedKeyDataSize<std::uint16_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 4:
            return (int)FastFss::cuda::dcfGetZippedKeyDataSize<std::uint32_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 8:
            return (int)FastFss::cuda::dcfGetZippedKeyDataSize<std::uint64_t>(
                bitWidthIn, bitWidthOut, elementNum);
        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}