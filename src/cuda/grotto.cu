#include <FastFss/cuda/grotto.h>

#include "../impl/grotto.h"

#define FSS_ASSERT(cond, errCode) \
    if (!(cond)) return errCode

using namespace FastFss;

enum ERR_CODE
{
    SUCCESS                      = 0,
    RUNTIME_ERROR                = -1,
    INVALID_ELEMENT_SIZE         = -2,
    INVALID_ALPHA_DATA_SIZE      = -3,
    INVALID_SEED_DATA_SIZE       = -4,
    INVALID_BITWIDTH             = -5,
    INVALID_PARTY_ID             = -6,
    INVALID_KEY_DATA_SIZE        = -7,
    INVALID_MASKED_X_DATA_SIZE   = -8,
    INVALID_CACHE_DATA_SIZE      = -9,
    INVALID_BOUNDARY_DATA_SIZE   = -10,
    INVALID_LUT_DATA_SIZE        = -11,
    INVALID_MIC_OUT_DATA_SIZE    = -12,
    INVALID_SHARED_OUT_DATA_SIZE = -13,
};

static std::size_t grottoGetKeyDataSize(size_t bitWidthIn,
                                        size_t elementSize,
                                        size_t elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetKeyDataSize<scalar_t>(bitWidthIn, elementNum);
        });
}

static std::size_t grottoGetCacheDataSize(size_t bitWidthIn,
                                          size_t elementSize,
                                          size_t elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetCacheDataSize<scalar_t>(bitWidthIn,
                                                          elementNum);
        });
}

template <typename GroupElement>
__global__ static void grottoKeyGenKernel(void*       key,
                                          const void* alpha,
                                          const void* seed0,
                                          const void* seed1,
                                          size_t      bitWidthIn,
                                          size_t      elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* alphaPtr = (const GroupElement*)alpha;
    const std::uint8_t* seed0Ptr = (const std::uint8_t*)seed0;
    const std::uint8_t* seed1Ptr = (const std::uint8_t*)seed1;

    impl::GrottoKey<GroupElement> keyObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        std::size_t alphaOffset = i;
        std::size_t seed0Offset = 16 * i;
        std::size_t seed1Offset = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        impl::grottoKeyGen(keyObj,                 //
                           alphaPtr[alphaOffset],  //
                           seed0Ptr + seed0Offset, //
                           seed1Ptr + seed1Offset, //
                           bitWidthIn);            //
    }
}

int FastFss_cuda_grottoKeyGen(void*       key,
                              size_t      keyDataSize,
                              const void* alpha,
                              size_t      alphaDataSize,
                              const void* seed0,
                              size_t      seedDataSize0,
                              const void* seed1,
                              size_t      seedDataSize1,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cudaStreamPtr)
{
    FSS_ASSERT(alphaDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_ALPHA_DATA_SIZE);
    FSS_ASSERT(seedDataSize0 == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(seedDataSize1 == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);

    std::size_t BLOCK_DIM = 512;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *((cudaStream_t*)cudaStreamPtr) : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoKeyGenKernel<scalar_t>                             //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(                //
                    key, alpha, seed0, seed1, bitWidthIn, elementNum //
                );                                                   //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
__global__ static void grottoEvalKernel(void*       out,
                                        const void* maskedX,
                                        const void* key,
                                        const void* seed,
                                        bool        equalBound,
                                        int         partyId,
                                        size_t      bitWidthIn,
                                        size_t      elementNum,
                                        void*       cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* maskedXPtr = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr    = (const std::uint8_t*)seed;
    GroupElement*       outPtr     = (GroupElement*)out;

    impl::GrottoKey<GroupElement>   keyObj;
    impl::GrottoCache<GroupElement> cacheObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        outPtr[i] = impl::grottoEval(keyObj,                    //
                                     maskedXPtr[maskedXOffset], //
                                     seedPtr + seedOffset,      //
                                     partyId,                   //
                                     bitWidthIn,                //
                                     equalBound,                //
                                     cacheObjPtr                //
        );
    }
}

int FastFss_cuda_grottoEval(void*       sharedBooleanOut,
                            const void* maskedX,
                            size_t      maskedXDataSize,
                            const void* key,
                            size_t      keyDataSize,
                            const void* seed,
                            size_t      seedDataSize,
                            bool        equalBound,
                            int         partyId,
                            size_t      bitWidthIn,
                            size_t      elementSize,
                            size_t      elementNum,
                            void*       cache,
                            size_t      cacheDataSize,
                            void*       cudaStreamPtr)
{
    FSS_ASSERT(maskedXDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);

    std::size_t BLOCK_DIM = 512;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *((cudaStream_t*)cudaStreamPtr) : 0;
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoEvalKernel<scalar_t>                //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                    sharedBooleanOut,                 //
                    maskedX,                          //
                    key,                              //
                    seed,                             //
                    equalBound,                       //
                    partyId,                          //
                    bitWidthIn,                       //
                    elementNum,                       //
                    cache                             //
                );                                    //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
__global__ static void grottoEvalEqKernel(void*       out,
                                          const void* maskedX,
                                          const void* key,
                                          const void* seed,
                                          int         partyId,
                                          size_t      bitWidthIn,
                                          size_t      elementNum,
                                          void*       cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* maskedXPtr = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr    = (const std::uint8_t*)seed;
    GroupElement*       outPtr     = (GroupElement*)out;

    impl::GrottoKey<GroupElement>   keyObj;
    impl::GrottoCache<GroupElement> cacheObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        outPtr[i] = impl::grottoEvalEq(keyObj,                    //
                                       maskedXPtr[maskedXOffset], //
                                       seedPtr + seedOffset,      //
                                       partyId,                   //
                                       bitWidthIn,                //
                                       cacheObjPtr                //
        );
    }
}

int FastFss_cuda_grottoEvalEq(void*       sharedBooleanOut,
                              const void* maskedX,
                              size_t      maskedXDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              const void* seed,
                              size_t      seedDataSize,
                              int         partyId,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cache,
                              size_t      cacheDataSize,
                              void*       cudaStreamPtr)
{
    FSS_ASSERT(maskedXDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);

    std::size_t BLOCK_DIM = 512;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *((cudaStream_t*)cudaStreamPtr) : 0;
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoEvalEqKernel<scalar_t>              //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                    sharedBooleanOut,                 //
                    maskedX,                          //
                    key,                              //
                    seed,                             //
                    partyId,                          //
                    bitWidthIn,                       //
                    elementNum,                       //
                    cache                             //
                );                                    //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
__global__ static void grottoMICEvalKernel(void*       out,
                                           const void* maskedX,
                                           const void* key,
                                           const void* seed,
                                           int         partyId,
                                           const void* leftBoundary,
                                           const void* rightBoundary,
                                           size_t      intervalNum,
                                           size_t      bitWidthIn,
                                           size_t      elementNum,
                                           void*       cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* maskedXPtr       = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr          = (const std::uint8_t*)seed;
    GroupElement*       outPtr           = (GroupElement*)out;
    const GroupElement* leftBoundaryPtr  = (GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (GroupElement*)rightBoundary;

    impl::GrottoKey<GroupElement>   keyObj;
    impl::GrottoCache<GroupElement> cacheObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        std::size_t                      outOffset     = intervalNum * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        impl::grottoMICEval(           //
            outPtr + outOffset,        //
            keyObj,                    //
            maskedXPtr[maskedXOffset], //
            seedPtr + seedOffset,      //
            partyId,                   //
            leftBoundaryPtr,           //
            rightBoundaryPtr,          //
            intervalNum,               //
            bitWidthIn,                //
            cacheObjPtr                //
        );
    }
}

int FastFss_cuda_grottoMICEval(void*       sharedBooleanOut,
                               size_t      sharedBooleanOutDataSize,
                               const void* maskedX,
                               size_t      maskedXDataSize,
                               const void* key,
                               size_t      keyDataSize,
                               const void* seed,
                               size_t      seedDataSize,
                               int         partyId,
                               const void* leftBoundary,
                               size_t      leftBoundaryDataSize,
                               const void* rightBoundary,
                               size_t      rightBoundaryDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum,
                               void*       cache,
                               size_t      cacheDataSize,
                               void*       cudaStreamPtr)
{
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);
    FSS_ASSERT(leftBoundaryDataSize == rightBoundaryDataSize &&
                   leftBoundaryDataSize % elementSize == 0,
               ERR_CODE::INVALID_BOUNDARY_DATA_SIZE);
    std::size_t intervalNum = leftBoundaryDataSize / elementSize;

    FSS_ASSERT(
        sharedBooleanOutDataSize == intervalNum * elementNum * elementSize,
        ERR_CODE::INVALID_MIC_OUT_DATA_SIZE);

    std::size_t BLOCK_DIM = 512;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoMICEvalKernel<scalar_t>             //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                    sharedBooleanOut,                 //
                    maskedX,                          //
                    key,                              //
                    seed,                             //
                    partyId,                          //
                    leftBoundary,                     //
                    rightBoundary,                    //
                    intervalNum,                      //
                    bitWidthIn,                       //
                    elementNum,                       //
                    cache                             //
                );                                    //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
__global__ static void grottoIntervalLutEvalKernel(void*       outE,
                                                   void*       outT,
                                                   const void* maskedX,
                                                   const void* key,
                                                   const void* seed,
                                                   int         partyId,
                                                   const void* leftBoundary,
                                                   const void* rightBoundary,
                                                   const void* lookUpTable,
                                                   size_t      lutNum,
                                                   size_t      intervalNum,
                                                   size_t      bitWidthIn,
                                                   size_t      elementNum,
                                                   void*       cache)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* maskedXPtr       = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr          = (const std::uint8_t*)seed;
    GroupElement*       outEPtr          = (GroupElement*)outE;
    GroupElement*       outTPtr          = (GroupElement*)outT;
    const GroupElement* leftBoundaryPtr  = (GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (GroupElement*)rightBoundary;
    const GroupElement* lookUpTablePtr   = (GroupElement*)lookUpTable;

    impl::GrottoKey<GroupElement>   keyObj;
    impl::GrottoCache<GroupElement> cacheObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        impl::grottoIntervalLutEval(   //
            outEPtr + i,               //
            outTPtr + i * lutNum,      //
            keyObj,                    //
            maskedXPtr[maskedXOffset], //
            seedPtr + seedOffset,      //
            partyId,                   //
            leftBoundaryPtr,           //
            rightBoundaryPtr,          //
            lookUpTablePtr,            //
            lutNum,                    //
            intervalNum,               //
            bitWidthIn,                //
            cacheObjPtr                //
        );
    }
}

int FastFss_cuda_grottoIntervalLutEval(void*       sharedOutE,
                                       void*       sharedOutT,
                                       const void* maskedX,
                                       size_t      maskedXDataSize,
                                       const void* key,
                                       size_t      keyDataSize,
                                       const void* seed,
                                       size_t      seedDataSize,
                                       int         partyId,
                                       const void* leftBoundary,
                                       size_t      leftBoundaryDataSize,
                                       const void* rightBoundary,
                                       size_t      rightBoundaryDataSize,
                                       const void* lookUpTable,
                                       size_t      lookUpTableDataSize,
                                       size_t      bitWidthIn,
                                       size_t      bitWidthOut,
                                       size_t      elementSize,
                                       size_t      elementNum,
                                       void*       cache,
                                       size_t      cacheDataSize,
                                       void*       cudaStreamPtr)
{
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);
    FSS_ASSERT(leftBoundaryDataSize == rightBoundaryDataSize &&
                   leftBoundaryDataSize % elementSize == 0,
               ERR_CODE::INVALID_BOUNDARY_DATA_SIZE);
    std::size_t intervalNum = leftBoundaryDataSize / elementSize;

    FSS_ASSERT(lookUpTableDataSize % (elementSize * intervalNum) == 0,
               INVALID_LUT_DATA_SIZE);
    std::size_t lutNum = lookUpTableDataSize / leftBoundaryDataSize;

    std::size_t BLOCK_DIM = 512;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoIntervalLutEvalKernel<scalar_t>     //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                    sharedOutE,                       //
                    sharedOutT,                       //
                    maskedX,                          //
                    key,                              //
                    seed,                             //
                    partyId,                          //
                    leftBoundary,                     //
                    rightBoundary,                    //
                    lookUpTable,                      //
                    lutNum,                           //
                    intervalNum,                      //
                    bitWidthIn,                       //
                    elementNum,                       //
                    cache                             //
                );                                    //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
}

int FastFss_cuda_grottoKeyZip(void*       zippedKey,
                              size_t      zippedKeyDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cudaStreamPtr)
{
    return -1;
}

int FastFss_cuda_grottoKeyUnzip(void*       key,
                                size_t      keyDataSize,
                                const void* zippedKey,
                                size_t      zippedKeyDataSize,
                                size_t      bitWidthIn,
                                size_t      elementSize,
                                size_t      elementNum,
                                void*       cudaStreamPtr)
{
    return -1;
}

int FastFss_cuda_grottoGetKeyDataSize(size_t* keyDataSize,
                                      size_t  bitWidthIn,
                                      size_t  elementSize,
                                      size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetKeyDataSize<scalar_t>(bitWidthIn, elementNum);
        });
    return ERR_CODE::SUCCESS;
}

int FastFss_cuda_grottoGetZippedKeyDataSize(size_t* keyDataSize,
                                            size_t  bitWidthIn,
                                            size_t  elementSize,
                                            size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetZippedKeyDataSize<scalar_t>(bitWidthIn,
                                                              elementNum);
        });
    return ERR_CODE::SUCCESS;
}

int FastFss_cuda_grottoGetCacheDataSize(size_t* cacheDataSize,
                                        size_t  bitWidthIn,
                                        size_t  elementSize,
                                        size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetCacheDataSize<scalar_t>(bitWidthIn,
                                                          elementNum);
        });
    return ERR_CODE::SUCCESS;
}