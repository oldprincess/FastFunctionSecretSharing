#include <FastFss/cuda/onehot.h>

#include "../impl/onehot.h"

using namespace FastFss;

#define FSS_ASSERT(cond, errCode) \
    if (!(cond)) return errCode

enum ERR_CODE
{
    SUCCESS                        = 0,
    RUNTIME_ERROR                  = -1,
    INVALID_BIT_WIDTH_IN           = -2,
    INVALID_ELEMENT_SIZE           = -3,
    INVALID_KEY_DATA_SIZE          = -4,
    INVALID_ALPHA_DATA_SIZE        = -5,
    INVALID_LOOKUP_TABLE_DATA_SIZE = -6,
    INVALID_MASKED_X_DATA_SIZE     = -7,
};

template <typename GroupElement>
__global__ static void onehotKeyGenKernel(void*       key,
                                          const void* alpha,
                                          std::size_t bitWidthIn,
                                          std::size_t elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* alphaPtr = (const GroupElement*)alpha;
    std::uint8_t*       keyPtr   = (std::uint8_t*)key;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        std::size_t keyOffset = i * (1ULL << bitWidthIn) / 8; //
        impl::onehotKeyGen<GroupElement>(                     //
            keyPtr + keyOffset, alphaPtr[i], bitWidthIn       //
        );                                                    //
    }
}

template <typename GroupElement>
__global__ static void onehotLutEvalKernel(void*       sharedOutE,
                                           void*       sharedOutT,
                                           const void* maskedX,
                                           const void* key,
                                           const void* lut,
                                           int         partyId,
                                           std::size_t bitWidthIn,
                                           std::size_t elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* maskedXPtr    = (const GroupElement*)maskedX;
    const std::uint8_t* keyPtr        = (const std::uint8_t*)key;
    const GroupElement* lutPtr        = (const GroupElement*)lut;
    GroupElement*       sharedOutEPtr = (GroupElement*)sharedOutE;
    GroupElement*       sharedOutTPtr = (GroupElement*)sharedOutT;

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

int FastFss_cuda_onehotKeyGen(void*       key,
                              size_t      keyDataSize,
                              const void* alpha,
                              size_t      alphaDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cudaStreamPtr) // cudaStream_t*
{
    using namespace impl;
    FSS_ASSERT(bitWidthIn >= 3, ERR_CODE::INVALID_BIT_WIDTH_IN);

    std::size_t needKeyDataSize = onehotGetKeyDataSize( //
        bitWidthIn, elementNum                          //
    );                                                  //
    FSS_ASSERT(needKeyDataSize == keyDataSize, ERR_CODE::INVALID_KEY_DATA_SIZE);

    FSS_ASSERT(alphaDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_ALPHA_DATA_SIZE);

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }

    cudaStream_t stream = 0;
    if (cudaStreamPtr != nullptr)
    {
        stream = *(cudaStream_t*)cudaStreamPtr;
    }
    auto ret = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize,                                //
        { return ERR_CODE::INVALID_ELEMENT_SIZE; }, //
        [&] {
            onehotKeyGenKernel<scalar_t>               //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(  //
                    key, alpha, bitWidthIn, elementNum //
                );                                     //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
    return ret;
}

int FastFss_cuda_onehotLutEval(void*       sharedOutE,
                               void*       sharedOutT,
                               const void* maskedX,
                               size_t      maskedXDataSize,
                               const void* key,
                               size_t      keyDataSize,
                               int         partyId,
                               const void* lookUpTable,
                               size_t      lookUpTableDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum,
                               void*       cudaStreamPtr) // cudaStream_t*
{
    using namespace impl;

    FSS_ASSERT(bitWidthIn >= 3, ERR_CODE::INVALID_BIT_WIDTH_IN);

    std::size_t needKeyDataSize = onehotGetKeyDataSize( //
        bitWidthIn, elementNum                          //
    );                                                  //
    FSS_ASSERT(maskedXDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(lookUpTableDataSize == elementSize * (1ULL << bitWidthIn),
               ERR_CODE::INVALID_LOOKUP_TABLE_DATA_SIZE);

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }

    cudaStream_t stream = 0;
    if (cudaStreamPtr != nullptr)
    {
        stream = *(cudaStream_t*)cudaStreamPtr;
    }
    auto ret = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize,                                //
        { return ERR_CODE::INVALID_ELEMENT_SIZE; }, //
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
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
    return ret;
}

int FastFss_cuda_onehotGetKeyDataSize(size_t* keyDataSize,
                                      size_t  bitWidthIn,
                                      size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn >= 3, ERR_CODE::INVALID_BIT_WIDTH_IN);
    *keyDataSize = impl::onehotGetKeyDataSize(bitWidthIn, elementNum);
    return (int)ERR_CODE::SUCCESS;
}