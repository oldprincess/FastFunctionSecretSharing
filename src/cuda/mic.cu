#include <FastFss/cuda/mic.h>

#include "../impl/mic.h"

using namespace FastFss;

#define FSS_ASSERT(cond, errCode) \
    if (!(cond)) return errCode

enum ERROR_CODE
{
    SUCCESS                            = 0,
    RUNTIME_ERROR                      = -1,
    INVALID_KEY_DATA_SIZE_ERROR        = -2,
    INVALID_ALPHA_DATA_SIZE_ERROR      = -3,
    INVALID_SEED_DATA_SIZE_ERROR       = -4,
    INVALID_BOUNDARY_DATA_SIZE_ERROR   = -5,
    INVALID_Z_DATA_SIZE_ERROR          = -6,
    INVALID_SHARED_OUT_DATA_SIZE_ERROR = -7,
    INVLIAD_MASKED_X_DATA_SIZE_ERROR   = -8,
    INVALID_BITWIDTH_ERROR             = -9,
    INVALID_ELEMENT_SIZE_ERROR         = -10,
    INVALID_PARTY_ID_ERROR             = -11,
    INVALID_MASKED_X_DATA_SIZE_ERROR   = -12,
};

template <typename GroupElement>
__global__ static void dcfMICKeyGenKernel(void*       key,
                                          void*       z,
                                          const void* alpha,
                                          const void* seed0,
                                          const void* seed1,
                                          const void* leftBoundary,
                                          const void* rightBoundary,
                                          size_t      intervalNum,
                                          size_t      bitWidthIn,
                                          size_t      bitWidthOut,
                                          size_t      elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement*       zPtr             = (GroupElement*)z;
    const GroupElement* alphaPtr         = (const GroupElement*)alpha;
    const std::uint8_t* seed0Ptr         = (const std::uint8_t*)seed0;
    const std::uint8_t* seed1Ptr         = (const std::uint8_t*)seed1;
    const GroupElement* leftBoundaryPtr  = (const GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (const GroupElement*)rightBoundary;

    impl::DcfKey<GroupElement> keyObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, i, elementNum);
        impl::dcfMICKeyGen(keyObj,                 //
                           zPtr + intervalNum * i, //
                           alphaPtr[i],            //
                           seed0Ptr + 16 * i,      //
                           seed1Ptr + 16 * i,      //
                           leftBoundaryPtr,        //
                           rightBoundaryPtr,       //
                           intervalNum,            //
                           bitWidthIn,             //
                           bitWidthOut             //
        );
    }
}

int FastFss_cuda_dcfMICKeyGen(void*       key,
                              size_t      keyDataSize,
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
                              size_t      elementNum,
                              void*       cudaStreamPtr)
{
    int         ret;
    std::size_t needKeyDataSize;
    ret = FastFss_cuda_dcfMICGetKeyDataSize(
        &needKeyDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum);
    FSS_ASSERT(ret == 0, ERROR_CODE::RUNTIME_ERROR);

    FSS_ASSERT(keyDataSize == needKeyDataSize,
               ERROR_CODE::INVALID_KEY_DATA_SIZE_ERROR);
    FSS_ASSERT(alphaDataSize == elementNum * elementSize,
               ERROR_CODE::INVALID_ALPHA_DATA_SIZE_ERROR);
    FSS_ASSERT(seedDataSize0 == 16 * elementNum,
               ERROR_CODE::INVALID_SEED_DATA_SIZE_ERROR);
    FSS_ASSERT(seedDataSize1 == 16 * elementNum,
               ERROR_CODE::INVALID_SEED_DATA_SIZE_ERROR);

    std::size_t intervalNum = leftBoundaryDataSize / elementSize;
    FSS_ASSERT(zDataSize == elementNum * intervalNum * elementSize,
               ERROR_CODE::INVALID_Z_DATA_SIZE_ERROR);
    FSS_ASSERT(leftBoundaryDataSize == intervalNum * elementSize,
               ERROR_CODE::INVALID_BOUNDARY_DATA_SIZE_ERROR);
    FSS_ASSERT(rightBoundaryDataSize == intervalNum * elementSize,
               ERROR_CODE::INVALID_BOUNDARY_DATA_SIZE_ERROR);

    FSS_ASSERT(bitWidthIn <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);
    FSS_ASSERT(bitWidthOut <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)(cudaStreamPtr) : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERROR_CODE::INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            auto intervalNum = leftBoundaryDataSize / elementSize;
            dcfMICKeyGenKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                key, z, alpha, seed0, seed1, leftBoundary, rightBoundary,
                intervalNum, bitWidthIn, bitWidthOut, elementNum);
            return ERROR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
__global__ static void dcfMICEvalKernel(void*       sharedOut,
                                        const void* maskedX,
                                        const void* key,
                                        const void* sharedZ,
                                        const void* seed,
                                        int         partyId,
                                        const void* leftBoundary,
                                        const void* rightBoundary,
                                        size_t      intervalNum,
                                        size_t      bitWidthIn,
                                        size_t      bitWidthOut,
                                        size_t      elementSize,
                                        size_t      elementNum,
                                        void*       cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement*       sharedOutPtr     = (GroupElement*)sharedOut;
    const GroupElement* maskedXPtr       = (const GroupElement*)maskedX;
    const GroupElement* sharedZPtr       = (const GroupElement*)sharedZ;
    const std::uint8_t* seedPtr          = (const std::uint8_t*)seed;
    const GroupElement* leftBoundaryPtr  = (const GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (const GroupElement*)rightBoundary;

    impl::DcfKey<GroupElement>    keyObj;
    impl::DcfCache<GroupElement>  cacheObj;
    impl::DcfCache<GroupElement>* cachePtr = nullptr;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, i, elementNum);
        if (cache != nullptr)
        {
            impl::dcfCacheSetPtr(cacheObj, cache, bitWidthIn, bitWidthOut, i,
                                 elementNum);
            cachePtr = &cacheObj;
        }
        impl::dcfMICEval(sharedOutPtr + intervalNum * i, //
                         maskedXPtr[i],                  //
                         keyObj,                         //
                         sharedZPtr + intervalNum * i,   //
                         seedPtr + 16 * i,               //
                         partyId,                        //
                         leftBoundaryPtr,                //
                         rightBoundaryPtr,               //
                         intervalNum,                    //
                         bitWidthIn,                     //
                         bitWidthOut,                    //
                         cachePtr                        //
        );
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
                            size_t      elementNum,
                            void*       cache,
                            size_t      cacheDataSize,
                            void*       cudaStreamPtr)
{
    int         ret;
    std::size_t needKeyDataSize;
    ret = FastFss_cuda_dcfMICGetKeyDataSize(
        &needKeyDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum);
    FSS_ASSERT(ret == 0, ERROR_CODE::RUNTIME_ERROR);

    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERROR_CODE::INVALID_MASKED_X_DATA_SIZE_ERROR);
    FSS_ASSERT(keyDataSize == needKeyDataSize,
               ERROR_CODE::INVALID_KEY_DATA_SIZE_ERROR);
    FSS_ASSERT(seedDataSize == 16 * elementNum,
               ERROR_CODE::INVALID_SEED_DATA_SIZE_ERROR);
    FSS_ASSERT(partyId == 0 || partyId == 1,
               ERROR_CODE::INVALID_PARTY_ID_ERROR);

    std::size_t intervalNum = leftBoundaryDataSize / elementSize;
    FSS_ASSERT(sharedOutDataSize == intervalNum * elementNum * elementSize,
               ERROR_CODE::INVALID_SHARED_OUT_DATA_SIZE_ERROR);
    FSS_ASSERT(sharedZDataSize == elementNum * intervalNum * elementSize,
               ERROR_CODE::INVALID_Z_DATA_SIZE_ERROR);
    FSS_ASSERT(leftBoundaryDataSize == intervalNum * elementSize,
               ERROR_CODE::INVALID_BOUNDARY_DATA_SIZE_ERROR);
    FSS_ASSERT(rightBoundaryDataSize == intervalNum * elementSize,
               ERROR_CODE::INVALID_BOUNDARY_DATA_SIZE_ERROR);

    FSS_ASSERT(bitWidthIn <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);
    FSS_ASSERT(bitWidthOut <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)(cudaStreamPtr) : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERROR_CODE::INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            auto intervalNum = leftBoundaryDataSize / elementSize;
            dcfMICEvalKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                sharedOut, maskedX, key, sharedZ, seed, partyId, leftBoundary,
                rightBoundary, intervalNum, bitWidthIn, bitWidthOut,
                elementSize, elementNum, cache);
            return ERROR_CODE::SUCCESS;
        });
}

int FastFss_cuda_dcfMICKeyZip(void*       zippedKey,
                              size_t      zippedKeyDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      elementSize,
                              size_t      elementNum)
{
    return ERROR_CODE::RUNTIME_ERROR;
}

int FastFss_cuda_dcfMICKeyUnzip(void*       key,
                                size_t      keyDataSize,
                                const void* zippedKey,
                                size_t      zippedKeyDataSize,
                                size_t      bitWidthIn,
                                size_t      bitWidthOut,
                                size_t      elementSize,
                                size_t      elementNum)
{
    return ERROR_CODE::RUNTIME_ERROR;
}

int FastFss_cuda_dcfMICGetCacheDataSize(size_t* cacheDataSize,
                                        size_t  bitWidthIn,
                                        size_t  bitWidthOut,
                                        size_t  elementSize,
                                        size_t  elementNum)
{
    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dcfGetCacheDataSize<scalar_t>(bitWidthIn, elementNum);
        });
    return ERROR_CODE::SUCCESS;
}

int FastFss_cuda_dcfMICGetKeyDataSize(size_t* keyDataSize,
                                      size_t  bitWidthIn,
                                      size_t  bitWidthOut,
                                      size_t  elementSize,
                                      size_t  elementNum)
{
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dcfGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut,
                                                     elementNum);
        });
    return ERROR_CODE::SUCCESS;
}

int FastFss_cuda_dcfMICGetZippedKeyDataSize(size_t* keyDataSize,
                                            size_t  bitWidthIn,
                                            size_t  bitWidthOut,
                                            size_t  elementSize,
                                            size_t  elementNum)
{
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dcfGetZippedKeyDataSize<scalar_t>(
                bitWidthIn, bitWidthOut, elementNum);
        });
    return ERROR_CODE::SUCCESS;
}