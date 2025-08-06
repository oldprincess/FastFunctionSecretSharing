#include <FastFss/cuda/config.h>
#include <FastFss/cuda/grotto.h>

// #if !defined(AES_IMPL)
// #include "aes.cuh"
// #define AES_IMPL
// #endif

#include "grotto/def.cuh"
#include "grotto/eq.cuh"
#include "grotto/eqMulti.cuh"
#include "grotto/eval.cuh"
#include "grotto/itervalLut.cuh"
#include "grotto/keyGen.cuh"
#include "grotto/lut.cuh"
#include "grotto/mic.cuh"

#define FSS_ASSERT(cond, errCode) \
    if (!(cond)) return errCode

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
    INVALID_POINT_DATA_SIZE      = -14,
};

using namespace FastFss;

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
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    FSS_ASSERT(alphaDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_ALPHA_DATA_SIZE);
    FSS_ASSERT(seedDataSize0 == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(seedDataSize1 == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
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
                printf("Error: %s\n",
                       cudaGetErrorString(cudaPeekAtLastError()));
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
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
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
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

int FastFss_cuda_grottoEqEval(void*       sharedBooleanOut,
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
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *((cudaStream_t*)cudaStreamPtr) : 0;
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoEqEvalKernel<scalar_t>              //
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

int FastFss_cuda_grottoEqMultiEval(void*       sharedBooleanOut,
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
                                   size_t      elementSize,
                                   size_t      elementNum,
                                   void*       cache,
                                   size_t      cacheDataSize,
                                   void*       cudaStreamPtr)
{
    std::size_t pointsNum = pointDataSize / elementSize;
    FSS_ASSERT(sharedOutDataSize == pointsNum * elementSize * elementNum,
               ERR_CODE::INVALID_SHARED_OUT_DATA_SIZE);
    FSS_ASSERT(maskedXDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);
    FSS_ASSERT(pointDataSize % elementSize == 0,
               ERR_CODE::INVALID_POINT_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }

    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)cudaStreamPtr : 0;
    std::size_t  BLOCK_DIM     = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t  GRID_DIM      = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    bool         isParallelAll = false;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    else if (GRID_DIM < FastFss_cuda_getGridDim())
    {
        isParallelAll = true;
        GRID_DIM      = (elementNum * pointsNum + BLOCK_DIM - 1) / BLOCK_DIM;
        if (GRID_DIM > CUDA_MAX_GRID_DIM)
        {
            GRID_DIM = CUDA_MAX_GRID_DIM;
        }
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            if (isParallelAll)
            {
                grottoEqMultiEvalParallelAllKernel<scalar_t> //
                    <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(    //
                        sharedBooleanOut,                    //
                        maskedX,                             //
                        key,                                 //
                        seed,                                //
                        partyId,                             //
                        point,                               //
                        pointsNum,                           //
                        bitWidthIn,                          //
                        elementNum                           //
                    );                                       //
            }
            else
            {
                grottoEqMultiEvalKernel<scalar_t>         //
                    <<<GRID_DIM, BLOCK_DIM, 0, stream>>>( //
                        sharedBooleanOut,                 //
                        maskedX,                          //
                        key,                              //
                        seed,                             //
                        partyId,                          //
                        point,                            //
                        pointsNum,                        //
                        bitWidthIn,                       //
                        elementNum,                       //
                        cache                             //
                    );                                    //
            }
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
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
    std::size_t intervalNum = leftBoundaryDataSize / elementSize;
    FSS_ASSERT(
        sharedBooleanOutDataSize == intervalNum * elementNum * elementSize,
        ERR_CODE::INVALID_MIC_OUT_DATA_SIZE);
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);
    FSS_ASSERT(leftBoundaryDataSize == rightBoundaryDataSize &&
                   leftBoundaryDataSize % elementSize == 0,
               ERR_CODE::INVALID_BOUNDARY_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }

    std::size_t BLOCK_DIM     = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM      = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    bool        isParallelAll = false;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    else if (GRID_DIM < FastFss_cuda_getGridDim())
    {
        isParallelAll = true;
        GRID_DIM      = (elementNum * intervalNum + BLOCK_DIM - 1) / BLOCK_DIM;
        if (GRID_DIM > CUDA_MAX_GRID_DIM)
        {
            GRID_DIM = CUDA_MAX_GRID_DIM;
        }
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            if (isParallelAll)
            {
                grottoMICEvalParallelAllKernel<scalar_t>  //
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
                        elementNum                        //
                    );                                    //
            }
            else
            {
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
            }
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
}

int FastFss_cuda_grottoLutEval(void*       sharedOutE,
                               void*       sharedOutT,
                               const void* maskedX,
                               size_t      maskedXDataSize,
                               const void* key,
                               size_t      keyDataSize,
                               const void* seed,
                               size_t      seedDataSize,
                               int         partyId,
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

    FSS_ASSERT(lookUpTableDataSize % (elementSize * (1ULL << bitWidthIn)) == 0,
               INVALID_LUT_DATA_SIZE);
    std::size_t lutNum =
        lookUpTableDataSize / (elementSize * (1ULL << bitWidthIn));

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoLutEvalKernel<scalar_t>            //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>> //
                (                                    //
                    sharedOutE,                      //
                    sharedOutT,                      //
                    maskedX,                         //
                    key,                             //
                    seed,                            //
                    partyId,                         //
                    lookUpTable,                     //
                    lutNum,                          //
                    bitWidthIn,                      //
                    elementNum,                      //
                    cache                            //
                );                                   //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
}

int FastFss_cuda_grottoLutEval_ex(void*       sharedOutE,
                                  void*       sharedOutT,
                                  const void* maskedX,
                                  size_t      maskedXDataSize,
                                  const void* key,
                                  size_t      keyDataSize,
                                  const void* seed,
                                  size_t      seedDataSize,
                                  int         partyId,
                                  const void* lookUpTable,
                                  size_t      lookUpTableDataSize,
                                  size_t      lutBitWidth,
                                  size_t      bitWidthIn,
                                  size_t      bitWidthOut,
                                  size_t      elementSize,
                                  size_t      elementNum,
                                  void*       cache0,
                                  void*       cache1,
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
    if (cache0 != nullptr || cache1 != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);

    FSS_ASSERT(lutBitWidth <= bitWidthIn, INVALID_LUT_DATA_SIZE);
    FSS_ASSERT(lookUpTableDataSize % (elementSize * (1ULL << lutBitWidth)) == 0,
               INVALID_LUT_DATA_SIZE);
    std::size_t lutNum =
        lookUpTableDataSize / (elementSize * (1ULL << lutBitWidth));

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoLutEvalKernel_ex<scalar_t>         //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>> //
                (                                    //
                    sharedOutE,                      //
                    sharedOutT,                      //
                    maskedX,                         //
                    key,                             //
                    seed,                            //
                    partyId,                         //
                    lookUpTable,                     //
                    lutNum,                          //
                    lutBitWidth,                     //
                    bitWidthIn,                      //
                    elementNum,                      //
                    cache0,                          //
                    cache1                           //
                );                                   //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
}

int FastFss_cuda_grottoLutEval_ex2(void*       sharedOutE,
                                   void*       sharedOutT,
                                   const void* maskedX,
                                   size_t      maskedXDataSize,
                                   const void* key,
                                   size_t      keyDataSize,
                                   const void* seed,
                                   size_t      seedDataSize,
                                   int         partyId,
                                   const void* points,
                                   size_t      pointsDataSize,
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

    FSS_ASSERT(lookUpTableDataSize % pointsDataSize == 0,
               INVALID_LUT_DATA_SIZE);
    std::size_t lutNum = lookUpTableDataSize / pointsDataSize;
    FSS_ASSERT(pointsDataSize % elementSize == 0, INVALID_POINT_DATA_SIZE);
    std::size_t pointsNum = pointsDataSize / elementSize;

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t GRID_DIM  = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t*)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoLutEvalKernel_ex2<scalar_t>        //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>> //
                (                                    //
                    sharedOutE,                      //
                    sharedOutT,                      //
                    maskedX,                         //
                    key,                             //
                    seed,                            //
                    partyId,                         //
                    points,                          //
                    pointsNum,                       //
                    lookUpTable,                     //
                    lutNum,                          //
                    bitWidthIn,                      //
                    elementNum,                      //
                    cache                            //
                );                                   //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return ERR_CODE::RUNTIME_ERROR;
            }
            return ERR_CODE::SUCCESS;
        });
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
    std::size_t intervalNum = leftBoundaryDataSize / elementSize;
    std::size_t lutNum      = lookUpTableDataSize / leftBoundaryDataSize;
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);
    FSS_ASSERT(leftBoundaryDataSize == rightBoundaryDataSize &&
                   leftBoundaryDataSize % elementSize == 0,
               ERR_CODE::INVALID_BOUNDARY_DATA_SIZE);
    FSS_ASSERT(lookUpTableDataSize % (elementSize * intervalNum) == 0,
               INVALID_LUT_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }

    std::size_t BLOCK_DIM = CUDA_DEFAULT_BLOCK_DIM;
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