#include <FastFss/cuda/onehot.h>

#include "../helper/onehot_helper.h"
#include "../impl/onehot.h"

using namespace FastFss;

template <typename GroupElement>
__global__ static void onehotKeyGenKernel(void       *key,
                                          const void *alpha,
                                          std::size_t bitWidthIn,
                                          std::size_t elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement *alphaPtr = (const GroupElement *)alpha;
    std::uint8_t       *keyPtr   = (std::uint8_t *)key;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        std::size_t keyOffset = i * (1ULL << bitWidthIn) / 8; //
        impl::onehotKeyGen<GroupElement>(                     //
            keyPtr + keyOffset, alphaPtr[i], bitWidthIn       //
        );                                                    //
    }
}

template <typename GroupElement>
__global__ static void onehotLutEvalKernel(void       *sharedOutE,
                                           void       *sharedOutT,
                                           const void *maskedX,
                                           const void *key,
                                           const void *lut,
                                           int         partyId,
                                           std::size_t bitWidthIn,
                                           std::size_t elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement *maskedXPtr    = (const GroupElement *)maskedX;
    const std::uint8_t *keyPtr        = (const std::uint8_t *)key;
    const GroupElement *lutPtr        = (const GroupElement *)lut;
    GroupElement       *sharedOutEPtr = (GroupElement *)sharedOutE;
    GroupElement       *sharedOutTPtr = (GroupElement *)sharedOutT;

    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        std::size_t keyOffset = i * (1ULL << bitWidthIn) / 8; //

        impl::onehotLutEval<GroupElement>( //
            sharedOutEPtr + i,             //
            sharedOutTPtr + i,             //
            maskedXPtr[i],                 //
            keyPtr + keyOffset,            //
            lutPtr,                        //
            partyId,                       //
            bitWidthIn);
    }
}

int FastFss_cuda_onehotKeyGen(void       *key,
                              size_t      keyDataSize,
                              const void *alpha,
                              size_t      alphaDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void       *cudaStreamPtr) // cudaStream_t*
{
    int ret = FastFss_helper_checkOnehotKeyGenParams(
        keyDataSize, alphaDataSize, bitWidthIn, elementSize, elementNum,
        FastFss_cuda_onehotGetKeyDataSize);
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

    cudaStream_t stream = 0;
    if (cudaStreamPtr != nullptr)
    {
        stream = *(cudaStream_t *)cudaStreamPtr;
    }
    ret = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize,                                     //
        { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; }, //
        [&] {
            onehotKeyGenKernel<scalar_t>               //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(  //
                    key, alpha, bitWidthIn, elementNum //
                );                                     //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
    return ret;
}

int FastFss_cuda_onehotLutEval(void       *sharedOutE,
                               size_t      sharedOutEDataSize,
                               void       *sharedOutT,
                               size_t      sharedOutTDataSize,
                               const void *maskedX,
                               size_t      maskedXDataSize,
                               const void *key,
                               size_t      keyDataSize,
                               int         partyId,
                               const void *lookUpTable,
                               size_t      lookUpTableDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum,
                               void       *cudaStreamPtr) // cudaStream_t*
{
    int ret = FastFss_helper_checkOnehotLutEvalParams(
        sharedOutEDataSize, sharedOutTDataSize, maskedXDataSize, keyDataSize,
        partyId, lookUpTableDataSize, bitWidthIn, elementSize, elementNum,
        FastFss_cuda_onehotGetKeyDataSize);
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

    cudaStream_t stream = 0;
    if (cudaStreamPtr != nullptr)
    {
        stream = *(cudaStream_t *)cudaStreamPtr;
    }
    ret = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize,                                     //
        { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; }, //
        [&] {
            onehotLutEvalKernel<scalar_t>             //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                    sharedOutE,                       //
                    sharedOutT,                       //
                    maskedX,                          //
                    key,                              //
                    lookUpTable,                      //
                    partyId,                          //
                    bitWidthIn,                       //
                    elementNum);                      //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
    return ret;
}

int FastFss_cuda_onehotGetKeyDataSize(size_t *keyDataSize,
                                      size_t  bitWidthIn,
                                      size_t  elementNum)
{
    if (bitWidthIn < 3)
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    *keyDataSize = impl::onehotGetKeyDataSize(bitWidthIn, elementNum);
    return FAST_FSS_SUCCESS;
}