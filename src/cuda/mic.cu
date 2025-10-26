#include <FastFss/cuda/config.h>
#include <FastFss/cuda/mic.h>
#include <cuda_runtime.h>

#include "../impl/mic.h"

using namespace FastFss;

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
__global__ static void dcfMICKeyGenKernel(void       *key,
                                          void       *z,
                                          const void *alpha,
                                          const void *seed0,
                                          const void *seed1,
                                          const void *leftBoundary,
                                          const void *rightBoundary,
                                          size_t      intervalNum,
                                          size_t      bitWidthIn,
                                          size_t      bitWidthOut,
                                          size_t      elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement       *zPtr             = (GroupElement *)z;
    const GroupElement *alphaPtr         = (const GroupElement *)alpha;
    const std::uint8_t *seed0Ptr         = (const std::uint8_t *)seed0;
    const std::uint8_t *seed1Ptr         = (const std::uint8_t *)seed1;
    const GroupElement *leftBoundaryPtr  = (const GroupElement *)leftBoundary;
    const GroupElement *rightBoundaryPtr = (const GroupElement *)rightBoundary;

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

int FastFss_cuda_dcfMICKeyGen(void       *key,
                              size_t      keyDataSize,
                              void       *z,
                              size_t      zDataSize,
                              const void *alpha,
                              size_t      alphaDataSize,
                              const void *seed0,
                              size_t      seedDataSize0,
                              const void *seed1,
                              size_t      seedDataSize1,
                              const void *leftBoundary,
                              size_t      leftBoundaryDataSize,
                              const void *rightBoundary,
                              size_t      rightBoundaryDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      elementSize,
                              size_t      elementNum,
                              void       *cudaStreamPtr)
{
    int         ret;
    std::size_t needKeyDataSize;
    ret = FastFss_cuda_dcfMICGetKeyDataSize(                               //
        &needKeyDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum //
    );                                                                     //
    if (ret != 0)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return ERROR_CODE::INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (alphaDataSize != elementNum * elementSize)
    {
        return ERROR_CODE::INVALID_ALPHA_DATA_SIZE_ERROR;
    }
    if (seedDataSize0 != 16 * elementNum)
    {
        return ERROR_CODE::INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (seedDataSize1 != 16 * elementNum)
    {
        return ERROR_CODE::INVALID_SEED_DATA_SIZE_ERROR;
    }

    std::size_t intervalNum = leftBoundaryDataSize / elementSize;
    if (intervalNum * elementSize != leftBoundaryDataSize ||
        intervalNum * elementSize != rightBoundaryDataSize)
    {
        return ERROR_CODE::INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }
    if (zDataSize != elementNum * intervalNum * elementSize)
    {
        return ERROR_CODE::INVALID_Z_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8))
    {
        return ERROR_CODE::INVALID_BITWIDTH_ERROR;
    }
    if (!(bitWidthOut <= elementSize * 8))
    {
        return ERROR_CODE::INVALID_BITWIDTH_ERROR;
    }

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream =
        (cudaStreamPtr) ? *(cudaStream_t *)(cudaStreamPtr) : 0;

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
__global__ static void dcfMICEvalKernel(void       *sharedOut,
                                        const void *maskedX,
                                        const void *key,
                                        const void *sharedZ,
                                        const void *seed,
                                        int         partyId,
                                        const void *leftBoundary,
                                        const void *rightBoundary,
                                        size_t      intervalNum,
                                        size_t      bitWidthIn,
                                        size_t      bitWidthOut,
                                        size_t      elementSize,
                                        size_t      elementNum,
                                        void       *cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement       *sharedOutPtr     = (GroupElement *)sharedOut;
    const GroupElement *maskedXPtr       = (const GroupElement *)maskedX;
    const GroupElement *sharedZPtr       = (const GroupElement *)sharedZ;
    const std::uint8_t *seedPtr          = (const std::uint8_t *)seed;
    const GroupElement *leftBoundaryPtr  = (const GroupElement *)leftBoundary;
    const GroupElement *rightBoundaryPtr = (const GroupElement *)rightBoundary;

    impl::DcfKey<GroupElement>    keyObj;
    impl::DcfCache<GroupElement>  cacheObj;
    impl::DcfCache<GroupElement> *cachePtr = nullptr;
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

template <typename GroupElement>
__global__ static void dcfMICEvalKernelParallel(void       *sharedOut,
                                                const void *maskedX,
                                                const void *key,
                                                const void *sharedZ,
                                                const void *seed,
                                                int         partyId,
                                                const void *leftBoundary,
                                                const void *rightBoundary,
                                                size_t      intervalNum,
                                                size_t      bitWidthIn,
                                                size_t      bitWidthOut,
                                                size_t      elementSize,
                                                size_t      elementNum,
                                                void       *cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement       *sharedOutPtr     = (GroupElement *)sharedOut;
    const GroupElement *maskedXPtr       = (const GroupElement *)maskedX;
    const GroupElement *sharedZPtr       = (const GroupElement *)sharedZ;
    const std::uint8_t *seedPtr          = (const std::uint8_t *)seed;
    const GroupElement *leftBoundaryPtr  = (const GroupElement *)leftBoundary;
    const GroupElement *rightBoundaryPtr = (const GroupElement *)rightBoundary;

    impl::DcfKey<GroupElement>   keyObj;
    impl::DcfCache<GroupElement> cacheObj;

    std::size_t preElementIdx = (std::size_t)(-1);
    std::size_t chunkSize = (intervalNum * elementNum + stride - 1) / stride;
    for (std::size_t i = idx * chunkSize; i < (idx + 1) * chunkSize;)
    {
        std::size_t elementIdx  = i / intervalNum;
        std::size_t intervalIdx = i % intervalNum;
        if (elementIdx >= elementNum)
        {
            break;
        }
        if (i == (idx * chunkSize) || elementIdx != preElementIdx)
        {
            impl::dcfKeySetPtr(                                              //
                keyObj, key, bitWidthIn, bitWidthOut, elementIdx, elementNum //
            );                                                               //
            impl::dcfCacheSetPtr(                                            //
                cacheObj,                                                    //
                cache,                                                       //
                bitWidthIn,                                                  //
                bitWidthOut,                                                 //
                idx,                                                         //
                stride                                                       //
            );                                                               //
        }

        std::size_t size = (idx + 1) * chunkSize - i;
        if (intervalIdx + size > intervalNum)
        {
            size = intervalNum - intervalIdx;
        }
        impl::dcfMICEval(                                          //
            sharedOutPtr + intervalNum * elementIdx + intervalIdx, //
            maskedXPtr[elementIdx],                                //
            keyObj,                                                //
            sharedZPtr + intervalNum * elementIdx + intervalIdx,   //
            seedPtr + 16 * elementIdx,                             //
            partyId,                                               //
            leftBoundaryPtr + intervalIdx,                         //
            rightBoundaryPtr + intervalIdx,                        //
            size,                                                  //
            bitWidthIn,                                            //
            bitWidthOut,                                           //
            &cacheObj                                              //
        );
        i += size;

        preElementIdx = elementIdx;
    }
}

int FastFss_cuda_dcfMICEval(void       *sharedOut,
                            size_t      sharedOutDataSize,
                            const void *maskedX,
                            size_t      maskedXDataSize,
                            const void *key,
                            size_t      keyDataSize,
                            const void *sharedZ,
                            size_t      sharedZDataSize,
                            const void *seed,
                            size_t      seedDataSize,
                            int         partyId,
                            const void *leftBoundary,
                            size_t      leftBoundaryDataSize,
                            const void *rightBoundary,
                            size_t      rightBoundaryDataSize,
                            size_t      bitWidthIn,
                            size_t      bitWidthOut,
                            size_t      elementSize,
                            size_t      elementNum,
                            void       *cache,
                            size_t      cacheDataSize,
                            void       *cudaStreamPtr)
{
    int         ret;
    std::size_t needKeyDataSize;
    std::size_t needCacheDataSize;

    ret = FastFss_cuda_dcfMICGetKeyDataSize(                               //
        &needKeyDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum //
    );                                                                     //
    if (ret != 0)
    {
        return ret;
    }
    ret = FastFss_cuda_dcfMICGetCacheDataSize(                               //
        &needCacheDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum //
    );                                                                       //
    if (ret != 0)
    {
        return ret;
    }

    if (maskedXDataSize != elementNum * elementSize)
    {
        return ERROR_CODE::INVALID_MASKED_X_DATA_SIZE_ERROR;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return ERROR_CODE::INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return ERROR_CODE::INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return ERROR_CODE::INVALID_PARTY_ID_ERROR;
    }

    std::size_t intervalNum = leftBoundaryDataSize / elementSize;
    if (intervalNum * elementSize != leftBoundaryDataSize ||
        intervalNum * elementSize != rightBoundaryDataSize)
    {
        return ERROR_CODE::INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }
    if (sharedOutDataSize != intervalNum * elementNum * elementSize)
    {
        return ERROR_CODE::INVALID_SHARED_OUT_DATA_SIZE_ERROR;
    }
    if (sharedZDataSize != elementNum * intervalNum * elementSize)
    {
        return ERROR_CODE::INVALID_Z_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8))
    {
        return ERROR_CODE::INVALID_BITWIDTH_ERROR;
    }
    if (!(bitWidthOut <= elementSize * 8))
    {
        return ERROR_CODE::INVALID_BITWIDTH_ERROR;
    }

    bool        parallel  = false;
    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    if (GRID_DIM < FastFss_cuda_getGridDim())
    {
        parallel = true;
        GRID_DIM = (elementNum * intervalNum + BLOCK_DIM - 1) / BLOCK_DIM;
        if (GRID_DIM > CUDA_MAX_GRID_DIM)
        {
            GRID_DIM = CUDA_MAX_GRID_DIM;
        }
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t *)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERROR_CODE::INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            if (parallel)
            {
                cudaError_t e          = cudaSuccess;
                void       *dCache     = nullptr;
                std::size_t dCacheSize = (                                    //
                    (needCacheDataSize / elementNum) * (BLOCK_DIM * GRID_DIM) //
                );                                                            //

                e = cudaMalloc(&dCache, dCacheSize);
                if (e != cudaSuccess)
                {
                    return ERROR_CODE::RUNTIME_ERROR;
                }
                dcfMICEvalKernelParallel<scalar_t>
                    <<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                        sharedOut,                        //
                        maskedX,                          //
                        key,                              //
                        sharedZ,                          //
                        seed,                             //
                        partyId,                          //
                        leftBoundary,                     //
                        rightBoundary,                    //
                        intervalNum,                      //
                        bitWidthIn,                       //
                        bitWidthOut,                      //
                        elementSize,                      //
                        elementNum,                       //
                        dCache                            //
                    );                                    //
                e = cudaFree(dCache);
                if (e != cudaSuccess)
                {
                    return ERROR_CODE::RUNTIME_ERROR;
                }
            }
            else
            {
                dcfMICEvalKernel<scalar_t>
                    <<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                        sharedOut,                        //
                        maskedX,                          //
                        key,                              //
                        sharedZ,                          //
                        seed,                             //
                        partyId,                          //
                        leftBoundary,                     //
                        rightBoundary,                    //
                        intervalNum,                      //
                        bitWidthIn,                       //
                        bitWidthOut,                      //
                        elementSize,                      //
                        elementNum,                       //
                        cache                             //
                    );                                    //
            }
            return ERROR_CODE::SUCCESS;
        });
}

int FastFss_cuda_dcfMICKeyZip(void       *zippedKey,
                              size_t      zippedKeyDataSize,
                              const void *key,
                              size_t      keyDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      elementSize,
                              size_t      elementNum)
{
    return ERROR_CODE::RUNTIME_ERROR;
}

int FastFss_cuda_dcfMICKeyUnzip(void       *key,
                                size_t      keyDataSize,
                                const void *zippedKey,
                                size_t      zippedKeyDataSize,
                                size_t      bitWidthIn,
                                size_t      bitWidthOut,
                                size_t      elementSize,
                                size_t      elementNum)
{
    return ERROR_CODE::RUNTIME_ERROR;
}

int FastFss_cuda_dcfMICGetCacheDataSize(size_t *cacheDataSize,
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

int FastFss_cuda_dcfMICGetKeyDataSize(size_t *keyDataSize,
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

int FastFss_cuda_dcfMICGetZippedKeyDataSize(size_t *keyDataSize,
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