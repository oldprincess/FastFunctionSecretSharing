#include <FastFss/cuda/config.h>
#include <FastFss/cuda/dpf.h>
#include <FastFss/errors.h>
#include <cuda_runtime.h>

#include "../impl/dpf.h"
#include "../kernel/dpf.h"

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
                key,       keyDataSize,   alpha,      alphaDataSize,
                beta,      betaDataSize,  seed0,      seedDataSize0,
                seed1,     seedDataSize1, bitWidthIn, bitWidthOut,
                groupSize, elementSize,   elementNum, cudaStreamPtr};
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
                sharedOut,    sharedOutDataSize, maskedX,     maskedXDataSize,
                key,          keyDataSize,       seed,        seedDataSize,
                partyId,      bitWidthIn,        bitWidthOut, groupSize,
                elementSize,  elementNum,        cache,       cacheDataSize,
                cudaStreamPtr};
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
                sharedOut,    sharedOutDataSize, maskedX,     maskedXDataSize,
                key,          keyDataSize,       seed,        seedDataSize,
                partyId,      bitWidthIn,        bitWidthOut, groupSize,
                elementSize,  elementNum,        cache,       cacheDataSize,
                cudaStreamPtr};
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
            impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, groupSize,
                               elementIdx, elementNum);
            impl::dpfCacheSetPtr(cacheObj, cache, bitWidthIn, idx, stride);
        }
        impl::dpfEval<GroupElement>( //
            sharedOutPtr + pointNum * elementIdx * groupSize +
                pointIdx * groupSize,                    //
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
        parallel             = true;
        GRID_DIM = (elementNum * pointNum + BLOCK_DIM - 1) / BLOCK_DIM;
        if (GRID_DIM > CUDA_MAX_GRID_DIM)
        {
            GRID_DIM = CUDA_MAX_GRID_DIM;
        }
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t *)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfEvalMultiTask<scalar_t> task{
                sharedOut,     sharedOutDataSize,
                maskedX,       maskedXDataSize,
                key,           keyDataSize,
                seed,          seedDataSize,
                partyId,       point,
                pointDataSize, bitWidthIn,
                bitWidthOut,   groupSize,
                elementSize,   elementNum,
                cache,         cacheDataSize,
                cudaStreamPtr};
            if (parallel)
            {
                std::size_t needCacheDataSize =
                    impl::dpfGetCacheDataSize<scalar_t>(bitWidthIn, elementNum);
                cudaError_t e          = cudaSuccess;
                void       *dCache     = nullptr;
                std::size_t dCacheSize = (                                    //
                    (needCacheDataSize / elementNum) * (BLOCK_DIM * GRID_DIM) //
                );

                e = cudaMallocAsync(&dCache, dCacheSize, stream);
                if (e != cudaSuccess)
                {
                    return (int)FAST_FSS_RUNTIME_ERROR;
                }
                _dpfEvalMultiKernel<scalar_t>
                    <<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                        sharedOut,                        //
                        maskedX,                          //
                        key,                              //
                        seed,                             //
                        partyId,                          //
                        point,                            //
                        pointDataSize / elementSize,      //
                        bitWidthIn,                       //
                        bitWidthOut,                      //
                        groupSize,                        //
                        elementNum,                       //
                        dCache                            //
                    );                                    //
                if (cudaPeekAtLastError() != cudaSuccess)
                {
                    return (int)FAST_FSS_RUNTIME_ERROR;
                }
                e = cudaFreeAsync(dCache, stream);
                if (e != cudaSuccess)
                {
                    return (int)FAST_FSS_RUNTIME_ERROR;
                }
            }
            else
            {
                return (int)kernel::parallel_execute(task);
            }
            return (int)FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_dpfKeyZip(void       *zippedKey,
                           size_t      zippedKeyDataSize,
                           const void *key,
                           size_t      keyDataSize,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      groupSize,
                           size_t      elementSize,
                           size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cuda_dpfKeyUnzip(void       *key,
                             size_t      keyDataSize,
                             const void *zippedKey,
                             size_t      zippedKeyDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      groupSize,
                             size_t      elementSize,
                             size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cuda_dpfGetKeyDataSize(size_t *keyDataSize,
                                   size_t  bitWidthIn,
                                   size_t  bitWidthOut,
                                   size_t  groupSize,
                                   size_t  elementSize,
                                   size_t  elementNum)
{
    if (bitWidthIn > elementSize * 8 || bitWidthOut > elementSize * 8)
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dpfGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut,
                                                     groupSize, elementNum);
        });
    if (*keyDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_dpfGetZippedKeyDataSize(size_t *keyDataSize,
                                         size_t  bitWidthIn,
                                         size_t  bitWidthOut,
                                         size_t  groupSize,
                                         size_t  elementSize,
                                         size_t  elementNum)
{
    if (bitWidthIn > elementSize * 8 || bitWidthOut > elementSize * 8)
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dpfGetZippedKeyDataSize<scalar_t>(
                bitWidthIn, bitWidthOut, groupSize, elementNum);
        });
    if (*keyDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_dpfGetCacheDataSize(size_t *cacheDataSize,
                                     size_t  bitWidthIn,
                                     size_t  elementSize,
                                     size_t  elementNum)
{
    if (bitWidthIn > elementSize * 8)
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dpfGetCacheDataSize<scalar_t>(bitWidthIn, elementNum);
        });
    if (*cacheDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}
