#include <FastFss/cuda/config.h>
#include <FastFss/cuda/dpf.h>

#include "../impl/dpf.h"

#define FSS_ASSERT(cond, errCode) \
    if (!(cond)) return errCode

using namespace FastFss;

enum ERROR_CODE
{
    SUCCESS                            = 0,
    RUNTIME_ERROR                      = -1,
    INVALID_KEY_DATA_SIZE_ERROR        = -2,
    INVALID_ALPHA_DATA_SIZE_ERROR      = -3,
    INVALID_BETA_DATA_SIZE_ERROR       = -4,
    INVALID_SEED_DATA_SIZE_ERROR       = -5,
    INVALID_BOUNDARY_DATA_SIZE_ERROR   = -6,
    INVALID_Z_DATA_SIZE_ERROR          = -7,
    INVALID_SHARED_OUT_DATA_SIZE_ERROR = -8,
    INVLIAD_MASKED_X_DATA_SIZE_ERROR   = -9,
    INVALID_BITWIDTH_ERROR             = -10,
    INVALID_ELEMENT_SIZE_ERROR         = -11,
    INVALID_PARTY_ID_ERROR             = -12,
    INVALID_CACHE_DATA_SIZE_ERROR      = -13,
};

template <typename GroupElement>
__global__ static void dpfKeyGenKernel(void*       key,
                                       const void* alpha,
                                       const void* beta,
                                       const void* seed0,
                                       const void* seed1,
                                       std::size_t bitWidthIn,
                                       std::size_t bitWidthOut,
                                       std::size_t elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* alphaPtr = (const GroupElement*)alpha;
    const GroupElement* betaPtr  = (const GroupElement*)beta;
    const std::uint8_t* seed0Ptr = (const std::uint8_t*)seed0;
    const std::uint8_t* seed1Ptr = (const std::uint8_t*)seed1;

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

int FastFss_cuda_dpfKeyGen(void*       key,
                           size_t      keyDataSize,
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
                           size_t      elementNum,
                           void*       cudaStreamPtr)
{
    int         ret;
    std::size_t needKeySize = 0;
    ret = FastFss_cuda_dpfGetKeyDataSize(&needKeySize, bitWidthIn, bitWidthOut,
                                         elementSize, elementNum);
    FSS_ASSERT(ret == 0, ERROR_CODE::RUNTIME_ERROR);

    FSS_ASSERT(keyDataSize == needKeySize,
               ERROR_CODE::INVALID_KEY_DATA_SIZE_ERROR);
    FSS_ASSERT(alphaDataSize == elementNum * elementSize,
               ERROR_CODE::INVALID_ALPHA_DATA_SIZE_ERROR);
    if (betaDataSize != 0)
    {
        FSS_ASSERT(betaDataSize == elementNum * elementSize,
                   ERROR_CODE::INVALID_BETA_DATA_SIZE_ERROR);
    }
    FSS_ASSERT(seedDataSize0 == elementNum * 16,
               ERROR_CODE::INVALID_SEED_DATA_SIZE_ERROR);
    FSS_ASSERT(seedDataSize1 == elementNum * 16,
               ERROR_CODE::INVALID_SEED_DATA_SIZE_ERROR);
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
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERROR_CODE::INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dpfKeyGenKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                key, alpha, (betaDataSize) ? beta : nullptr, seed0, seed1,
                bitWidthIn, bitWidthOut, elementNum);

            return ERROR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
__global__ static void dpfEvalKernel(void*       sharedOut,
                                     const void* maskedX,
                                     const void* key,
                                     const void* seed,
                                     int         partyId,
                                     size_t      bitWidthIn,
                                     size_t      bitWidthOut,
                                     size_t      elementNum,
                                     void*       cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement*       sharedOutPtr = (GroupElement*)sharedOut;
    const GroupElement* maskedXPtr   = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr      = (const std::uint8_t*)seed;

    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement>* cacheObjPtr = nullptr;
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

template <typename GroupElement>
__global__ static void dpfMultiEvalKernel(void*       sharedOut,
                                          const void* maskedX,
                                          const void* key,
                                          const void* seed,
                                          int         partyId,
                                          const void* point,
                                          size_t      pointNum,
                                          size_t      bitWidthIn,
                                          size_t      bitWidthOut,
                                          size_t      elementNum,
                                          void*       cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement*       sharedOutPtr = (GroupElement*)sharedOut;
    const GroupElement* maskedXPtr   = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr      = (const std::uint8_t*)seed;
    const GroupElement* pointPtr     = (const GroupElement*)point;

    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement>* cacheObjPtr = nullptr;
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
__global__ static void dpfMultiEvalKernelParallelAll(void*       sharedOut,
                                                     const void* maskedX,
                                                     const void* key,
                                                     const void* seed,
                                                     int         partyId,
                                                     const void* point,
                                                     size_t      pointNum,
                                                     size_t      bitWidthIn,
                                                     size_t      bitWidthOut,
                                                     size_t      elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement*       sharedOutPtr = (GroupElement*)sharedOut;
    const GroupElement* maskedXPtr   = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr      = (const std::uint8_t*)seed;
    const GroupElement* pointPtr     = (const GroupElement*)point;

    std::size_t totalNum = pointNum * elementNum;
    for (std::size_t i = idx; i < totalNum; i += stride)
    {
        impl::DpfKey<GroupElement> keyObj;

        std::size_t elementIdx = i / pointNum;
        std::size_t pointIdx   = i % pointNum;

        impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, elementIdx,
                           elementNum);
        GroupElement tmp = maskedXPtr[elementIdx] - pointPtr[pointIdx];
        sharedOutPtr[pointNum * elementIdx + pointIdx] =
            impl::dpfEval<GroupElement>(keyObj,                    //
                                        tmp,                       //
                                        seedPtr + 16 * elementIdx, //
                                        partyId,                   //
                                        bitWidthIn,                //
                                        bitWidthOut,               //
                                        nullptr);
    }
}

int FastFss_cuda_dpfEval(void*       sharedOut,
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
                         size_t      elementNum,
                         void*       cache,
                         size_t      cacheDataSize,
                         void*       cudaStreamPtr)
{
    int         ret;
    std::size_t needKeySize = 0;
    ret = FastFss_cuda_dpfGetKeyDataSize(&needKeySize, bitWidthIn, bitWidthOut,
                                         elementSize, elementNum);
    FSS_ASSERT(ret == 0, ERROR_CODE::RUNTIME_ERROR);

    FSS_ASSERT(keyDataSize == needKeySize,
               ERROR_CODE::INVALID_KEY_DATA_SIZE_ERROR);
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERROR_CODE::INVLIAD_MASKED_X_DATA_SIZE_ERROR);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERROR_CODE::INVALID_SEED_DATA_SIZE_ERROR);
    FSS_ASSERT(partyId == 0 || partyId == 1,
               ERROR_CODE::INVALID_PARTY_ID_ERROR);
    FSS_ASSERT(bitWidthIn <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);
    FSS_ASSERT(bitWidthOut <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);
    if (cacheDataSize != 0)
    {
        std::size_t needCacheSize = 0;

        ret = FastFss_cuda_dpfGetCacheDataSize(
            &needCacheSize, bitWidthIn, bitWidthOut, elementSize, elementNum);

        FSS_ASSERT(ret == 0, ERROR_CODE::RUNTIME_ERROR);
        FSS_ASSERT(cacheDataSize == needCacheSize,
                   ERROR_CODE::INVALID_CACHE_DATA_SIZE_ERROR);
    }

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERROR_CODE::INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dpfEvalKernel<scalar_t><<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                sharedOut, maskedX, key, seed, partyId, bitWidthIn, bitWidthOut,
                elementNum, cache);
            return ERROR_CODE::SUCCESS;
        });
}

int FastFss_cuda_dpfMultiEval(void*       sharedOut,
                              size_t      sharedOutDataSize,
                              const void* maskedX,
                              size_t      maskedXDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              const void* seed,
                              size_t      seedDataSize,
                              int         partyId,
                              const void* point,
                              size_t      pointDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cache,
                              size_t      cacheDataSize,
                              void*       cudaStreamPtr)
{
    int         ret;
    std::size_t needKeySize = 0;
    ret = FastFss_cuda_dpfGetKeyDataSize(&needKeySize, bitWidthIn, bitWidthOut,
                                         elementSize, elementNum);
    FSS_ASSERT(ret == 0, ERROR_CODE::RUNTIME_ERROR);

    std::size_t pointNum = pointDataSize / elementSize;
    FSS_ASSERT(sharedOutDataSize == pointNum * elementNum * elementSize,
               ERROR_CODE::INVALID_SHARED_OUT_DATA_SIZE_ERROR);
    FSS_ASSERT(keyDataSize == needKeySize,
               ERROR_CODE::INVALID_KEY_DATA_SIZE_ERROR);
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERROR_CODE::INVLIAD_MASKED_X_DATA_SIZE_ERROR);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERROR_CODE::INVALID_SEED_DATA_SIZE_ERROR);
    FSS_ASSERT(partyId == 0 || partyId == 1,
               ERROR_CODE::INVALID_PARTY_ID_ERROR);
    FSS_ASSERT(bitWidthIn <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);
    FSS_ASSERT(bitWidthOut <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);

    if (cacheDataSize != 0)
    {
        std::size_t needCacheSize = 0;

        ret = FastFss_cuda_dpfGetCacheDataSize(
            &needCacheSize, bitWidthIn, bitWidthOut, elementSize, elementNum);

        FSS_ASSERT(ret == 0, ERROR_CODE::RUNTIME_ERROR);
        FSS_ASSERT(cacheDataSize == needCacheSize,
                   ERROR_CODE::INVALID_CACHE_DATA_SIZE_ERROR);
    }

    bool        parallelAll = false;
    std::size_t BLOCK_DIM   = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM    = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    if (GRID_DIM < FastFss_cuda_getGridDim())
    {
        parallelAll = true;
        GRID_DIM    = (elementNum * pointNum + BLOCK_DIM - 1) / BLOCK_DIM;
        if (GRID_DIM > CUDA_MAX_GRID_DIM)
        {
            GRID_DIM = CUDA_MAX_GRID_DIM;
        }
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERROR_CODE::INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            if (parallelAll)
            {
                dpfMultiEvalKernelParallelAll<scalar_t>
                    <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                        sharedOut, maskedX, key, seed, partyId, point, pointNum,
                        bitWidthIn, bitWidthOut, elementNum);
            }
            else
            {
                dpfMultiEvalKernel<scalar_t>
                    <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(
                        sharedOut, maskedX, key, seed, partyId, point, pointNum,
                        bitWidthIn, bitWidthOut, elementNum, cache);
            }
            return ERROR_CODE::SUCCESS;
        });
}

int FastFss_cuda_dpfKeyZip(void*       zippedKey,
                           size_t      zippedKeyDataSize,
                           const void* key,
                           size_t      keyDataSize,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      elementSize,
                           size_t      elementNum,
                           void*       cudaStreamPtr)
{
    return ERROR_CODE::RUNTIME_ERROR;
}

int FastFss_cuda_dpfKeyUnzip(void*       key,
                             size_t      keyDataSize,
                             const void* zippedKey,
                             size_t      zippedKeyDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      elementSize,
                             size_t      elementNum,
                             void*       cudaStreamPtr)
{
    return ERROR_CODE::RUNTIME_ERROR;
}

int FastFss_cuda_dpfGetKeyDataSize(size_t* keyDataSize,
                                   size_t  bitWidthIn,
                                   size_t  bitWidthOut,
                                   size_t  elementSize,
                                   size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);
    FSS_ASSERT(bitWidthOut <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dpfGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut,
                                                     elementNum);
        });
    return ERROR_CODE::SUCCESS;
}

int FastFss_cuda_dpfGetZippedKeyDataSize(size_t* keyDataSize,
                                         size_t  bitWidthIn,
                                         size_t  bitWidthOut,
                                         size_t  elementSize,
                                         size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);
    FSS_ASSERT(bitWidthOut <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dpfGetZippedKeyDataSize<scalar_t>(
                bitWidthIn, bitWidthOut, elementNum);
        });
    return ERROR_CODE::SUCCESS;
}

int FastFss_cuda_dpfGetCacheDataSize(size_t* cacheDataSize,
                                     size_t  bitWidthIn,
                                     size_t  bitWidthOut,
                                     size_t  elementSize,
                                     size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);
    FSS_ASSERT(bitWidthOut <= elementSize * 8,
               ERROR_CODE::INVALID_BITWIDTH_ERROR);

    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dpfGetCacheDataSize<scalar_t>(bitWidthIn, elementNum);
        });
    return ERROR_CODE::SUCCESS;
}