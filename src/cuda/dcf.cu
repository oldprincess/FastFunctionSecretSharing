#include <FastFss/cuda/dcf.h>

#include "../helper/dcf_helper.h"
#include "../impl/dcf.h"

using namespace FastFss;

template <typename GroupElement>
__global__ static void dcfKeyGenKernel(void       *key,
                                       const void *alpha,
                                       const void *beta,
                                       const void *seed0,
                                       const void *seed1,
                                       std::size_t bitWidthIn,
                                       std::size_t bitWidthOut,
                                       std::size_t groupSize,
                                       std::size_t elementNum)
{
    const GroupElement ONE = 1;

    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement *alphaPtr = (const GroupElement *)alpha;
    const GroupElement *betaPtr  = (const GroupElement *)beta;
    const std::uint8_t *seed0Ptr = (const std::uint8_t *)seed0;
    const std::uint8_t *seed1Ptr = (const std::uint8_t *)seed1;

    impl::DcfKey<GroupElement> keyObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, groupSize, i,
                           elementNum);
        const GroupElement *ptr = &ONE;
        if (betaPtr != nullptr)
        {
            ptr = betaPtr + i * groupSize;
        }
        impl::dcfKeyGen(keyObj,            //
                        alphaPtr[i],       //
                        ptr,               //
                        seed0Ptr + 16 * i, //
                        seed1Ptr + 16 * i, //
                        bitWidthIn,        //
                        bitWidthOut,       //
                        groupSize          //
        );
    }
}

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
                           void       *cudaStreamPtr)
{
    int ret = FastFss_helper_checkDcfKeyGenParams( //
        keyDataSize,                               //
        alphaDataSize,                             //
        betaDataSize,                              //
        seedDataSize0,                             //
        seedDataSize1,                             //
        bitWidthIn,                                //
        bitWidthOut,                               //
        groupSize,                                 //
        elementSize,                               //
        elementNum,                                //
        FastFss_cuda_dcfGetKeyDataSize             //
    );
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t *)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dcfKeyGenKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                key, alpha, (betaDataSize) ? beta : nullptr, seed0, seed1,
                bitWidthIn, bitWidthOut, groupSize, elementNum);

            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
__global__ static void dcfEvalKernel(void       *sharedOut,
                                     const void *maskedX,
                                     const void *key,
                                     const void *seed,
                                     int         partyId,
                                     size_t      bitWidthIn,
                                     size_t      bitWidthOut,
                                     size_t      groupSize,
                                     size_t      elementNum,
                                     void       *cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement       *sharedOutPtr = (GroupElement *)sharedOut;
    const GroupElement *maskedXPtr   = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr      = (const std::uint8_t *)seed;

    impl::DcfKey<GroupElement> keyObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, groupSize, i,
                           elementNum);
        impl::dcfEval(sharedOutPtr + groupSize * i, //
                      keyObj,                       //
                      maskedXPtr[i],                //
                      seedPtr + 16 * i,             //
                      partyId,                      //
                      bitWidthIn,                   //
                      bitWidthOut,                  //
                      groupSize);
    }
}

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
                         void       *cudaStreamPtr)
{
    int ret = FastFss_helper_checkDcfEvalParams( //
        sharedOutSize,                           //
        maskedXDataSize,                         //
        keyDataSize,                             //
        seedDataSize,                            //
        partyId,                                 //
        bitWidthIn,                              //
        bitWidthOut,                             //
        groupSize,                               //
        elementSize,                             //
        elementNum,                              //
        cacheDataSize,                           //
        FastFss_cuda_dcfGetKeyDataSize,          //
        FastFss_cuda_dcfGetCacheDataSize         //
    );

    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t *)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dcfEvalKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                sharedOut, maskedX, key, seed, partyId, bitWidthIn, bitWidthOut,
                groupSize, elementNum, cache);
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_dcfGetKeyDataSize(size_t *keyDataSize,
                                   size_t  bitWidthIn,
                                   size_t  bitWidthOut,
                                   size_t  groupSize,
                                   size_t  elementSize,
                                   size_t  elementNum)
{
    if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dcfGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut,
                                                     groupSize, elementNum);
        });
    if (*keyDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_dcfGetCacheDataSize(size_t *cacheDataSize,
                                     size_t  bitWidthIn,
                                     size_t  bitWidthOut,
                                     size_t  groupSize,
                                     size_t  elementSize,
                                     size_t  elementNum)
{
    if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dcfGetCacheDataSize<scalar_t>(bitWidthIn, groupSize,
                                                       elementNum);
        });
    if (*cacheDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}