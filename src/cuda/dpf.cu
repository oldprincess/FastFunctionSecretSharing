#include <FastFss/cuda/config.h>
#include <FastFss/cuda/dpf.h>
#include <cuda_runtime.h>

#include "../helper/dpf_helper.h"
#include "../helper/error_code.h"
#include "../impl/dpf.h"

using namespace FastFss;

template <typename GroupElement>
__global__ static void dpfKeyGenKernel(void       *key,
                                       const void *alpha,
                                       const void *beta,
                                       const void *seed0,
                                       const void *seed1,
                                       std::size_t bitWidthIn,
                                       std::size_t bitWidthOut,
                                       std::size_t elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement *alphaPtr = (const GroupElement *)alpha;
    const GroupElement *betaPtr  = (const GroupElement *)beta;
    const std::uint8_t *seed0Ptr = (const std::uint8_t *)seed0;
    const std::uint8_t *seed1Ptr = (const std::uint8_t *)seed1;

    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::DpfKey<GroupElement> keyObj;
        impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, i, elementNum);
        impl::dpfKeyGen(keyObj,                                   //
                        alphaPtr[i],                              //
                        (betaPtr) ? betaPtr[i] : (GroupElement)1, //
                        seed0Ptr + 16 * i,                        //
                        seed1Ptr + 16 * i,                        //
                        bitWidthIn,                               //
                        bitWidthOut                               //
        );
    }
}

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
                           size_t      elementSize,
                           size_t      elementNum,
                           void       *cudaStreamPtr)
{
    int ret = FastFss_helper_checkDpfKeyGenParams(
        keyDataSize, alphaDataSize, betaDataSize, seedDataSize0, seedDataSize1,
        bitWidthIn, bitWidthOut, elementSize, elementNum,
        FastFss_cuda_dpfGetKeyDataSize);
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
            dpfKeyGenKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                key,                             //
                alpha,                           //
                (betaDataSize) ? beta : nullptr, //
                seed0,                           //
                seed1,                           //
                bitWidthIn,                      //
                bitWidthOut,                     //
                elementNum                       //
            );                                   //

            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
__global__ static void dpfEvalKernel(void       *sharedOut,
                                     const void *maskedX,
                                     const void *key,
                                     const void *seed,
                                     int         partyId,
                                     size_t      bitWidthIn,
                                     size_t      bitWidthOut,
                                     size_t      elementNum,
                                     void       *cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement       *sharedOutPtr = (GroupElement *)sharedOut;
    const GroupElement *maskedXPtr   = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr      = (const std::uint8_t *)seed;

    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement> *cacheObjPtr = nullptr;
        impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, i, elementNum);
        if (cache != nullptr)
        {
            impl::dpfCacheSetPtr(cacheObj, cache, bitWidthIn, bitWidthOut, i,
                                 elementNum);
            cacheObjPtr = &cacheObj;
        }
        sharedOutPtr[i] = impl::dpfEval(keyObj,           //
                                        maskedXPtr[i],    //
                                        seedPtr + 16 * i, //
                                        partyId,          //
                                        bitWidthIn,       //
                                        bitWidthOut, cacheObjPtr);
    }
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
                         size_t      elementSize,
                         size_t      elementNum,
                         void       *cache,
                         size_t      cacheDataSize,
                         void       *cudaStreamPtr)
{
    int ret = FastFss_helper_checkDpfEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize,
        cacheDataSize, partyId, bitWidthIn, bitWidthOut, elementSize,
        elementNum, FastFss_cuda_dpfGetKeyDataSize,
        FastFss_cuda_dpfGetCacheDataSize);
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
            dpfEvalKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                sharedOut,   //
                maskedX,     //
                key,         //
                seed,        //
                partyId,     //
                bitWidthIn,  //
                bitWidthOut, //
                elementNum,  //
                cache        //
            );               //
            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
__global__ static void dpfEvalAllKernel(void       *sharedOut,
                                        const void *maskedX,
                                        const void *key,
                                        const void *seed,
                                        int         partyId,
                                        size_t      bitWidthIn,
                                        size_t      bitWidthOut,
                                        size_t      elementNum,
                                        void       *cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement       *sharedOutPtr = (GroupElement *)sharedOut;
    const GroupElement *maskedXPtr   = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr      = (const std::uint8_t *)seed;

    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement> *cacheObjPtr = nullptr;
        impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, i, elementNum);
        if (cache != nullptr)
        {
            impl::dpfCacheSetPtr(cacheObj,    //
                                 cache,       //
                                 bitWidthIn,  //
                                 bitWidthOut, //
                                 i,           //
                                 elementNum   //
            );                                //
            cacheObjPtr = &cacheObj;
        }
        std::size_t size = (std::size_t)(1ULL << bitWidthIn);
        for (std::size_t j = 0; j < size; j++)
        {
            sharedOutPtr[size * i + (maskedXPtr[i] - j) % size] =
                impl::dpfEval(keyObj,           //
                              (GroupElement)j,  //
                              seedPtr + 16 * i, //
                              partyId,          //
                              bitWidthIn,       //
                              bitWidthOut,      //
                              cacheObjPtr       //
                );
        }
    }
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
                            size_t      elementSize,
                            size_t      elementNum,
                            void       *cache,
                            size_t      cacheDataSize,
                            void       *cudaStreamPtr)
{
    int ret = FastFss_helper_checkDpfEvalAllParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize,
        cacheDataSize, partyId, bitWidthIn, bitWidthOut, elementSize,
        elementNum, FastFss_cuda_dpfGetKeyDataSize,
        FastFss_cuda_dpfGetCacheDataSize);
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
            dpfEvalAllKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                sharedOut,                                                  //
                maskedX,                                                    //
                key,                                                        //
                seed,                                                       //
                partyId,                                                    //
                bitWidthIn,                                                 //
                bitWidthOut,                                                //
                elementNum,                                                 //
                cache                                                       //
            );                                                              //
            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
__global__ static void dpfMultiEvalKernel(void       *sharedOut,
                                          const void *maskedX,
                                          const void *key,
                                          const void *seed,
                                          int         partyId,
                                          const void *point,
                                          size_t      pointNum,
                                          size_t      bitWidthIn,
                                          size_t      bitWidthOut,
                                          size_t      elementNum,
                                          void       *cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement       *sharedOutPtr = (GroupElement *)sharedOut;
    const GroupElement *maskedXPtr   = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr      = (const std::uint8_t *)seed;
    const GroupElement *pointPtr     = (const GroupElement *)point;

    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement> *cacheObjPtr = nullptr;
        impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, i, elementNum);
        if (cache != nullptr)
        {
            impl::dpfCacheSetPtr(cacheObj, cache, bitWidthIn, bitWidthOut, i,
                                 elementNum);
            cacheObjPtr = &cacheObj;
        }
        for (std::size_t j = 0; j < pointNum; j++)
        {
            GroupElement tmp = maskedXPtr[i] - pointPtr[j];
            sharedOutPtr[pointNum * i + j] =
                impl::dpfEval(keyObj,           //
                              tmp,              //
                              seedPtr + 16 * i, //
                              partyId,          //
                              bitWidthIn,       //
                              bitWidthOut, cacheObjPtr);
        }
    }
}

template <typename GroupElement>
__global__ static void dpfMultiEvalKernelParallel(void       *sharedOut,
                                                  const void *maskedX,
                                                  const void *key,
                                                  const void *seed,
                                                  int         partyId,
                                                  const void *point,
                                                  size_t      pointNum,
                                                  size_t      bitWidthIn,
                                                  size_t      bitWidthOut,
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
            impl::dpfKeySetPtr(                                              //
                keyObj, key, bitWidthIn, bitWidthOut, elementIdx, elementNum //
            );                                                               //
            impl::dpfCacheSetPtr(                                            //
                cacheObj,                                                    //
                cache,                                                       //
                bitWidthIn,                                                  //
                bitWidthOut,                                                 //
                idx,                                                         //
                stride                                                       //
            );                                                               //
        }
        sharedOutPtr[pointNum * elementIdx + pointIdx] =
            impl::dpfEval<GroupElement>(                     //
                keyObj,                                      //
                maskedXPtr[elementIdx] - pointPtr[pointIdx], //
                seedPtr + 16 * elementIdx,                   //
                partyId,                                     //
                bitWidthIn,                                  //
                bitWidthOut,                                 //
                &cacheObj);
        preElementIdx = elementIdx;
    }
}

int FastFss_cuda_dpfMultiEval(void       *sharedOut,
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
                              size_t      elementSize,
                              size_t      elementNum,
                              void       *cache,
                              size_t      cacheDataSize,
                              void       *cudaStreamPtr)
{
    int ret = FastFss_helper_checkDpfMultiEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize,
        pointDataSize, cacheDataSize, partyId, bitWidthIn, bitWidthOut,
        elementSize, elementNum, FastFss_cuda_dpfGetKeyDataSize,
        FastFss_cuda_dpfGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }
    std::size_t pointNum          = pointDataSize / elementSize;
    std::size_t needCacheDataSize = 0;

    ret = FastFss_cuda_dpfGetCacheDataSize(                                  //
        &needCacheDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum //
    );                                                                       //
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
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
        GRID_DIM = (elementNum * pointNum + BLOCK_DIM - 1) / BLOCK_DIM;
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
                cudaError_t e          = cudaSuccess;
                void       *dCache     = nullptr;
                std::size_t dCacheSize = (                                    //
                    (needCacheDataSize / elementNum) * (BLOCK_DIM * GRID_DIM) //
                );                                                            //

                e = cudaMalloc(&dCache, dCacheSize);
                if (e != cudaSuccess)
                {
                    return FAST_FSS_RUNTIME_ERROR;
                }
                dpfMultiEvalKernelParallel<scalar_t>
                    <<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                        sharedOut,                        //
                        maskedX,                          //
                        key,                              //
                        seed,                             //
                        partyId,                          //
                        point,                            //
                        pointNum,                         //
                        bitWidthIn,                       //
                        bitWidthOut,                      //
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
                dpfMultiEvalKernel<scalar_t>
                    <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                        sharedOut, maskedX, key, seed, partyId, point, pointNum,
                        bitWidthIn, bitWidthOut, elementNum, cache);
            }
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_dpfKeyZip(void       *zippedKey,
                           size_t      zippedKeyDataSize,
                           const void *key,
                           size_t      keyDataSize,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      elementSize,
                           size_t      elementNum,
                           void       *cudaStreamPtr)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cuda_dpfKeyUnzip(void       *key,
                             size_t      keyDataSize,
                             const void *zippedKey,
                             size_t      zippedKeyDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      elementSize,
                             size_t      elementNum,
                             void       *cudaStreamPtr)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cuda_dpfGetKeyDataSize(size_t *keyDataSize,
                                   size_t  bitWidthIn,
                                   size_t  bitWidthOut,
                                   size_t  elementSize,
                                   size_t  elementNum)
{
    if (!(bitWidthIn <= elementSize * 8 && bitWidthOut <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dpfGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut,
                                                     elementNum);
        });
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_dpfGetZippedKeyDataSize(size_t *keyDataSize,
                                         size_t  bitWidthIn,
                                         size_t  bitWidthOut,
                                         size_t  elementSize,
                                         size_t  elementNum)
{
    if (!(bitWidthIn <= elementSize * 8 && bitWidthOut <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dpfGetZippedKeyDataSize<scalar_t>(
                bitWidthIn, bitWidthOut, elementNum);
        });
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_dpfGetCacheDataSize(size_t *cacheDataSize,
                                     size_t  bitWidthIn,
                                     size_t  bitWidthOut,
                                     size_t  elementSize,
                                     size_t  elementNum)
{
    if (!(bitWidthIn <= elementSize * 8 && bitWidthOut <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dpfGetCacheDataSize<scalar_t>(bitWidthIn, elementNum);
        });
    return FAST_FSS_SUCCESS;
}