#include <FastFss/cuda/config.h>
#include <FastFss/cuda/mic.h>
#include <cuda_runtime.h>

#include "../helper/mic_helper.h"
#include "../impl/mic.h"

using namespace FastFss;

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
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, 1, i,
                           elementNum);
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
    int ret = FastFss_helper_checkDcfMicKeyGenParams(
        keyDataSize, zDataSize, alphaDataSize, seedDataSize0, seedDataSize1,
        leftBoundaryDataSize, rightBoundaryDataSize, bitWidthIn, bitWidthOut,
        elementSize, elementNum, FastFss_cuda_dcfMICGetKeyDataSize);
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
    cudaStream_t stream =
        (cudaStreamPtr) ? *(cudaStream_t *)(cudaStreamPtr) : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            auto intervalNum = leftBoundaryDataSize / elementSize;
            dcfMICKeyGenKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                key, z, alpha, seed0, seed1, leftBoundary, rightBoundary,
                intervalNum, bitWidthIn, bitWidthOut, elementNum);
            return FAST_FSS_SUCCESS;
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
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, 1, i,
                           elementNum);
        if (cache != nullptr)
        {
            impl::dcfCacheSetPtr(cacheObj, cache, bitWidthIn, 1, i, elementNum);
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
            impl::dcfKeySetPtr( //
                keyObj, key, bitWidthIn, bitWidthOut, 1, elementIdx,
                elementNum        //
            );                    //
            impl::dcfCacheSetPtr( //
                cacheObj,         //
                cache,            //
                bitWidthIn,       //
                1,                //
                idx,              //
                stride            //
            );                    //
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
    int ret = FastFss_helper_checkDcfMicEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, sharedZDataSize,
        seedDataSize, leftBoundaryDataSize, rightBoundaryDataSize, partyId,
        bitWidthIn, bitWidthOut, elementSize, elementNum, cacheDataSize,
        FastFss_cuda_dcfMICGetKeyDataSize, FastFss_cuda_dcfMICGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    std::size_t needCacheDataSize = 0;
    std::size_t intervalNum       = leftBoundaryDataSize / elementSize;

    ret = FastFss_cuda_dcfMICGetCacheDataSize(                               //
        &needCacheDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum //
    );                                                                       //

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
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            if (parallel)
            {
                cudaError_t e      = cudaSuccess;
                void       *dCache = nullptr;
                std::size_t dCacheSize =
                    ((needCacheDataSize / elementNum) * (BLOCK_DIM * GRID_DIM));

                e = cudaMalloc(&dCache, dCacheSize);
                if (e != cudaSuccess)
                {
                    return FAST_FSS_RUNTIME_ERROR;
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
                    return FAST_FSS_RUNTIME_ERROR;
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
            return FAST_FSS_SUCCESS;
        });
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
            return impl::dcfGetCacheDataSize<scalar_t>(bitWidthIn, 1,
                                                       elementNum);
        });
    return FAST_FSS_SUCCESS;
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
            return impl::dcfGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut, 1,
                                                     elementNum);
        });
    return FAST_FSS_SUCCESS;
}
