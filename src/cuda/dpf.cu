#include <FastFss/cuda/config.h>
#include <FastFss/cuda/dpf.h>
#include <FastFss/errors.h>
#include <cuda_runtime.h>

#include "../impl/dpf.h"
#include "../kernel/dpf.h"
#include "../kernel/parallel_execute.h"

using namespace FastFss;

int FastFss_cuda_dpfKeyGen(void       *key,
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
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfKeyGenTask<scalar_t> task{
                .key           = key,
                .keyDataSize   = keyDataSize,
                .alpha         = alpha,
                .alphaDataSize = alphaDataSize,
                .beta          = beta,
                .betaDataSize  = betaDataSize,
                .seed0         = seed0,
                .seedDataSize0 = seedDataSize0,
                .seed1         = seed1,
                .seedDataSize1 = seedDataSize1,
                .bitWidthIn    = bitWidthIn,
                .bitWidthOut   = bitWidthOut,
                .groupSize     = groupSize,
                .elementSize   = elementSize,
                .elementNum    = elementNum,
                .cudaStreamPtr = cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
        });
}

int FastFss_cuda_dpfEval(void       *sharedOut,
                         size_t      sharedOutDataSize,
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
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfEvalTask<scalar_t> task{
                .sharedOut         = sharedOut,
                .sharedOutDataSize = sharedOutDataSize,
                .maskedX           = maskedX,
                .maskedXDataSize   = maskedXDataSize,
                .key               = key,
                .keyDataSize       = keyDataSize,
                .seed              = seed,
                .seedDataSize      = seedDataSize,
                .partyId           = partyId,
                .bitWidthIn        = bitWidthIn,
                .bitWidthOut       = bitWidthOut,
                .groupSize         = groupSize,
                .elementSize       = elementSize,
                .elementNum        = elementNum,
                .cache             = cache,
                .cacheDataSize     = cacheDataSize,
                .cudaStreamPtr     = cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
        });
}

int FastFss_cuda_dpfEvalAll(void       *sharedOut,
                            size_t      sharedOutDataSize,
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
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfEvalAllTask<scalar_t> task{
                .sharedOut         = sharedOut,
                .sharedOutDataSize = sharedOutDataSize,
                .maskedX           = maskedX,
                .maskedXDataSize   = maskedXDataSize,
                .key               = key,
                .keyDataSize       = keyDataSize,
                .seed              = seed,
                .seedDataSize      = seedDataSize,
                .partyId           = partyId,
                .bitWidthIn        = bitWidthIn,
                .bitWidthOut       = bitWidthOut,
                .groupSize         = groupSize,
                .elementSize       = elementSize,
                .elementNum        = elementNum,
                .cache             = cache,
                .cacheDataSize     = cacheDataSize,
                .cudaStreamPtr     = cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
        });
}

template <typename GroupElement>
__global__ static void _dpfEvalMultiKernel(void       *sharedOut,
                                           const void *maskedX,
                                           const void *key,
                                           const void *seed,
                                           int         partyId,
                                           const void *point,
                                           size_t      pointNum,
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
    const GroupElement *pointPtr     = (const GroupElement *)point;

    std::size_t chunkSize = (pointNum * elementNum + stride - 1) / stride;

    impl::DpfKey<GroupElement>   keyObj;
    impl::DpfCache<GroupElement> cacheObj;
    std::size_t                  preElementIdx = (std::size_t)(-1);
    for (std::size_t i = idx * chunkSize; i < (idx + 1) * chunkSize; i++)
    {
        std::size_t elementIdx = i / pointNum;
        std::size_t pointIdx   = i % pointNum;
        if (elementIdx >= elementNum)
        {
            break;
        }
        if (i == (idx * chunkSize) || elementIdx != preElementIdx)
        {
            impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, groupSize, elementIdx, elementNum);
            impl::dpfCacheSetPtr(cacheObj, cache, bitWidthIn, idx, stride);
        }
        std::size_t outIdx = pointNum * elementIdx * groupSize + pointIdx * groupSize;
        impl::dpfEval<GroupElement>(                     //
            sharedOutPtr + outIdx,                       //
            keyObj,                                      //
            maskedXPtr[elementIdx] - pointPtr[pointIdx], //
            seedPtr + 16 * elementIdx,                   //
            partyId,                                     //
            bitWidthIn,                                  //
            bitWidthOut,                                 //
            groupSize,                                   //
            &cacheObj);
        preElementIdx = elementIdx;
    }
}

int FastFss_cuda_dpfEvalMulti(void       *sharedOut,
                              size_t      sharedOutDataSize,
                              const void *maskedX,
                              size_t      maskedXDataSize,
                              const void *key,
                              size_t      keyDataSize,
                              const void *seed,
                              size_t      seedDataSize,
                              int         partyId,
                              const void *point,
                              size_t      pointDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      groupSize,
                              size_t      elementSize,
                              size_t      elementNum,
                              void       *cache,
                              size_t      cacheDataSize,
                              void       *cudaStreamPtr)
{
    bool        parallel  = false;
    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    if (GRID_DIM < FastFss_cuda_getGridDim())
    {
        std::size_t pointNum = pointDataSize / elementSize;
        GRID_DIM             = (elementNum * pointNum + BLOCK_DIM - 1) / BLOCK_DIM;
        if (GRID_DIM > CUDA_MAX_GRID_DIM)
        {
            GRID_DIM = CUDA_MAX_GRID_DIM;
        }
        parallel = true;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfEvalMultiTask<scalar_t> task{
                .sharedOut         = sharedOut,
                .sharedOutDataSize = sharedOutDataSize,
                .maskedX           = maskedX,
                .maskedXDataSize   = maskedXDataSize,
                .key               = key,
                .keyDataSize       = keyDataSize,
                .seed              = seed,
                .seedDataSize      = seedDataSize,
                .partyId           = partyId,
                .point             = point,
                .pointDataSize     = pointDataSize,
                .bitWidthIn        = bitWidthIn,
                .bitWidthOut       = bitWidthOut,
                .groupSize         = groupSize,
                .elementSize       = elementSize,
                .elementNum        = elementNum,
                .cache             = cache,
                .cacheDataSize     = cacheDataSize,
                .cudaStreamPtr     = cudaStreamPtr,
            };
            if (!parallel)
            {
                return (int)kernel::parallel_execute(task);
            }
            else
            {
                if (int ret = task.check(); ret != FAST_FSS_SUCCESS)
                {
                    return ret;
                }

                cudaStream_t stream     = (cudaStreamPtr) ? *(cudaStream_t *)cudaStreamPtr : 0;
                std::size_t  dCacheSize = impl::dpfGetCacheDataSize<scalar_t>(bitWidthIn, BLOCK_DIM * GRID_DIM);

                void *dCache = nullptr;
                if (cudaMallocAsync(&dCache, dCacheSize, stream) != cudaSuccess)
                {
                    return (int)FAST_FSS_RUNTIME_ERROR;
                }
                _dpfEvalMultiKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                    sharedOut,                                                     //
                    maskedX,                                                       //
                    key,                                                           //
                    seed,                                                          //
                    partyId,                                                       //
                    point,                                                         //
                    pointDataSize / elementSize,                                   //
                    bitWidthIn,                                                    //
                    bitWidthOut,                                                   //
                    groupSize,                                                     //
                    elementNum,                                                    //
                    dCache                                                         //
                );                                                                 //
                if (cudaPeekAtLastError() != cudaSuccess)
                {
                    cudaFreeAsync(dCache, stream);
                    return (int)FAST_FSS_RUNTIME_ERROR;
                }
                else
                {
                    cudaFreeAsync(dCache, stream);
                    return (int)FAST_FSS_SUCCESS;
                }
            }
        });
}
