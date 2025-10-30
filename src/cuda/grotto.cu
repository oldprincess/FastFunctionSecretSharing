#include <FastFss/cuda/config.h>
#include <FastFss/cuda/grotto.h>

// #if !defined(AES_IMPL)
// #include "aes.cuh"
// #define AES_IMPL
// #endif

#include "../helper/error_code.h"
#include "../helper/grotto_helper.h"
#include "grotto/def.cuh"
#include "grotto/eq.cuh"
#include "grotto/eqMulti.cuh"
#include "grotto/eval.cuh"
#include "grotto/itervalLut.cuh"
#include "grotto/keyGen.cuh"
#include "grotto/lut.cuh"
#include "grotto/mic.cuh"

using namespace FastFss;

int FastFss_cuda_grottoKeyGen(void       *key,
                              size_t      keyDataSize,
                              const void *alpha,
                              size_t      alphaDataSize,
                              const void *seed0,
                              size_t      seedDataSize0,
                              const void *seed1,
                              size_t      seedDataSize1,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void       *cudaStreamPtr)
{
    int ret = FastFss_helper_checkGrottoKeyGenParams(
        keyDataSize, alphaDataSize, seedDataSize0, seedDataSize1, bitWidthIn,
        elementSize, elementNum, FastFss_cuda_grottoGetKeyDataSize);
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
        (cudaStreamPtr) ? *((cudaStream_t *)cudaStreamPtr) : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            grottoKeyGenKernel<scalar_t>                             //
                <<<GRID_DIM, BLOCK_DIM, 0, stream>>>(                //
                    key, alpha, seed0, seed1, bitWidthIn, elementNum //
                );                                                   //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                printf("Error: %s\n",
                       cudaGetErrorString(cudaPeekAtLastError()));
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_grottoEval(void       *sharedBooleanOut,
                            size_t      sharedOutDataSize,
                            const void *maskedX,
                            size_t      maskedXDataSize,
                            const void *key,
                            size_t      keyDataSize,
                            const void *seed,
                            size_t      seedDataSize,
                            bool        equalBound,
                            int         partyId,
                            size_t      bitWidthIn,
                            size_t      elementSize,
                            size_t      elementNum,
                            void       *cache,
                            size_t      cacheDataSize,
                            void       *cudaStreamPtr)
{
    int ret = FastFss_helper_checkGrottoEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize, partyId,
        bitWidthIn, elementSize, elementNum, cacheDataSize,
        FastFss_cuda_grottoGetKeyDataSize, FastFss_cuda_grottoGetCacheDataSize);
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
        (cudaStreamPtr) ? *((cudaStream_t *)cudaStreamPtr) : 0;
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
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
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_grottoEqEval(void       *sharedBooleanOut,
                              size_t      sharedOutDataSize,
                              const void *maskedX,
                              size_t      maskedXDataSize,
                              const void *key,
                              size_t      keyDataSize,
                              const void *seed,
                              size_t      seedDataSize,
                              int         partyId,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void       *cache,
                              size_t      cacheDataSize,
                              void       *cudaStreamPtr)
{
    int ret = FastFss_helper_checkGrottoEqEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize, partyId,
        bitWidthIn, elementSize, elementNum, cacheDataSize,
        FastFss_cuda_grottoGetKeyDataSize, FastFss_cuda_grottoGetCacheDataSize);
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
        (cudaStreamPtr) ? *((cudaStream_t *)cudaStreamPtr) : 0;
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
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
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_grottoEqMultiEval(void       *sharedBooleanOut,
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
                                   size_t      elementSize,
                                   size_t      elementNum,
                                   void       *cache,
                                   size_t      cacheDataSize,
                                   void       *cudaStreamPtr)
{
    int ret = FastFss_helper_checkGrottoEqMultiEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize, partyId,
        pointDataSize, bitWidthIn, elementSize, elementNum, cacheDataSize,
        FastFss_cuda_grottoGetKeyDataSize, FastFss_cuda_grottoGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t *)cudaStreamPtr : 0;
    std::size_t  BLOCK_DIM     = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t  GRID_DIM      = (elementNum + BLOCK_DIM - 1) / BLOCK_DIM;
    bool         isParallelAll = false;
    if (GRID_DIM > CUDA_MAX_GRID_DIM)
    {
        GRID_DIM = CUDA_MAX_GRID_DIM;
    }
    else if (GRID_DIM < FastFss_cuda_getGridDim())
    {
        auto pointNum = pointDataSize / elementSize;

        isParallelAll = true;
        GRID_DIM      = (elementNum * pointNum + BLOCK_DIM - 1) / BLOCK_DIM;
        if (GRID_DIM > CUDA_MAX_GRID_DIM)
        {
            GRID_DIM = CUDA_MAX_GRID_DIM;
        }
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            auto pointNum = pointDataSize / elementSize;
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
                        pointNum,                            //
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
                        pointNum,                         //
                        bitWidthIn,                       //
                        elementNum,                       //
                        cache                             //
                    );                                    //
            }
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_grottoMICEval(void       *sharedBooleanOut,
                               size_t      sharedBooleanOutDataSize,
                               const void *maskedX,
                               size_t      maskedXDataSize,
                               const void *key,
                               size_t      keyDataSize,
                               const void *seed,
                               size_t      seedDataSize,
                               int         partyId,
                               const void *leftBoundary,
                               size_t      leftBoundaryDataSize,
                               const void *rightBoundary,
                               size_t      rightBoundaryDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum,
                               void       *cache,
                               size_t      cacheDataSize,
                               void       *cudaStreamPtr)
{
    int         ret               = 0;
    std::size_t needKeyDataSize   = 0;
    std::size_t needCacheDataSize = 0;

    ret = FastFss_cuda_grottoGetKeyDataSize(                  //
        &needKeyDataSize, bitWidthIn, elementSize, elementNum //
    );                                                        //
    if (ret != 0)
    {
        return ret;
    }

    ret = FastFss_cuda_grottoGetCacheDataSize(                  //
        &needCacheDataSize, bitWidthIn, elementSize, elementNum //
    );                                                          //
    if (ret != 0)
    {
        return ret;
    }

    std::size_t intervalNum = leftBoundaryDataSize / elementSize;
    if (intervalNum * elementSize != leftBoundaryDataSize ||
        intervalNum * elementSize != rightBoundaryDataSize)
    {
        return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }
    if (sharedBooleanOutDataSize != elementNum * elementSize * intervalNum)
    {
        return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
    }
    if (maskedXDataSize != elementNum * elementSize)
    {
        return FAST_FSS_INVALID_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != elementNum * 16)
    {
        return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return FAST_FSS_INVALID_PARTY_ID_ERROR;
    }
    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    if (cache != nullptr)
    {
        if (cacheDataSize != needCacheDataSize)
        {
            return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
        }
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
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t *)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
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
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_grottoLutEval(void       *sharedOutE,
                               size_t      sharedOutEDataSize,
                               void       *sharedOutT,
                               size_t      sharedOutTDataSize,
                               const void *maskedX,
                               size_t      maskedXDataSize,
                               const void *key,
                               size_t      keyDataSize,
                               const void *seed,
                               size_t      seedDataSize,
                               int         partyId,
                               const void *lookUpTable,
                               size_t      lookUpTableDataSize,
                               size_t      bitWidthIn,
                               size_t      bitWidthOut,
                               size_t      elementSize,
                               size_t      elementNum,
                               void       *cache,
                               size_t      cacheDataSize,
                               void       *cudaStreamPtr)
{
    return FastFss_cuda_grottoLutEval_ex(                               //
        sharedOutE, sharedOutEDataSize, sharedOutT, sharedOutTDataSize, //
        maskedX, maskedXDataSize,                                       //
        key, keyDataSize,                                               //
        seed, seedDataSize,                                             //
        partyId,                                                        //
        lookUpTable, lookUpTableDataSize,                               //
        bitWidthIn,                                                     //
        bitWidthIn,                                                     //
        bitWidthOut,                                                    //
        elementSize,                                                    //
        elementNum,                                                     //
        cache, cacheDataSize,                                           //
        cudaStreamPtr                                                   //
    );
}

int FastFss_cuda_grottoLutEval_ex(void       *sharedOutE,
                                  size_t      sharedOutEDataSize,
                                  void       *sharedOutT,
                                  size_t      sharedOutTDataSize,
                                  const void *maskedX,
                                  size_t      maskedXDataSize,
                                  const void *key,
                                  size_t      keyDataSize,
                                  const void *seed,
                                  size_t      seedDataSize,
                                  int         partyId,
                                  const void *lookUpTable,
                                  size_t      lookUpTableDataSize,
                                  size_t      lutBitWidth,
                                  size_t      bitWidthIn,
                                  size_t      bitWidthOut,
                                  size_t      elementSize,
                                  size_t      elementNum,
                                  void       *cache,
                                  size_t      cacheDataSize,
                                  void       *cudaStreamPtr)
{
    int ret = 0;
    ret     = FastFss_helper_checkGrottoLutEval_exParams( //
        sharedOutEDataSize, sharedOutTDataSize, maskedXDataSize, keyDataSize,
        seedDataSize, partyId, lookUpTableDataSize, lutBitWidth, bitWidthIn,
        bitWidthOut, elementSize, elementNum, cacheDataSize,
        FastFss_cuda_grottoGetKeyDataSize, FastFss_cuda_grottoGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }
    std::size_t lutNum =
        lookUpTableDataSize / (elementSize * (1ULL << lutBitWidth));

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
                    cache                            //
                );                                   //
            if (cudaPeekAtLastError() != cudaSuccess)
            {
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_grottoLutEval_ex2(void       *sharedOutE,
                                   size_t      sharedOutEDataSize,
                                   void       *sharedOutT,
                                   size_t      sharedOutTDataSize,
                                   const void *maskedX,
                                   size_t      maskedXDataSize,
                                   const void *key,
                                   size_t      keyDataSize,
                                   const void *seed,
                                   size_t      seedDataSize,
                                   int         partyId,
                                   const void *points,
                                   size_t      pointsDataSize,
                                   const void *lookUpTable,
                                   size_t      lookUpTableDataSize,
                                   size_t      bitWidthIn,
                                   size_t      bitWidthOut,
                                   size_t      elementSize,
                                   size_t      elementNum,
                                   void       *cache,
                                   size_t      cacheDataSize,
                                   void       *cudaStreamPtr)
{
    int ret = 0;
    ret     = FastFss_helper_checkGrottoLutEval_ex2Params( //
        sharedOutEDataSize, sharedOutTDataSize, maskedXDataSize, keyDataSize,
        seedDataSize, partyId, pointsDataSize, lookUpTableDataSize, bitWidthIn,
        bitWidthOut, elementSize, elementNum, cacheDataSize,
        FastFss_cuda_grottoGetKeyDataSize, FastFss_cuda_grottoGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }
    std::size_t pointsNum = pointsDataSize / elementSize;
    std::size_t lutNum    = lookUpTableDataSize / pointsDataSize;

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
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_grottoIntervalLutEval(void       *sharedOutE,
                                       size_t      sharedOutEDataSize,
                                       void       *sharedOutT,
                                       size_t      sharedOutTDataSize,
                                       const void *maskedX,
                                       size_t      maskedXDataSize,
                                       const void *key,
                                       size_t      keyDataSize,
                                       const void *seed,
                                       size_t      seedDataSize,
                                       int         partyId,
                                       const void *leftBoundary,
                                       size_t      leftBoundaryDataSize,
                                       const void *rightBoundary,
                                       size_t      rightBoundaryDataSize,
                                       const void *lookUpTable,
                                       size_t      lookUpTableDataSize,
                                       size_t      bitWidthIn,
                                       size_t      bitWidthOut,
                                       size_t      elementSize,
                                       size_t      elementNum,
                                       void       *cache,
                                       size_t      cacheDataSize,
                                       void       *cudaStreamPtr)
{
    std::size_t intervalNum = leftBoundaryDataSize / elementSize;
    std::size_t lutNum      = lookUpTableDataSize / leftBoundaryDataSize;
    int         ret         = 0;
    ret = FastFss_helper_checkGrottoIntervalLutEvalParams( //
        sharedOutEDataSize, sharedOutTDataSize, maskedXDataSize, keyDataSize,
        seedDataSize, partyId, leftBoundaryDataSize, rightBoundaryDataSize,
        lookUpTableDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum,
        cacheDataSize, FastFss_cuda_grottoGetKeyDataSize,
        FastFss_cuda_grottoGetCacheDataSize);
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
                return FAST_FSS_RUNTIME_ERROR;
            }
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cuda_grottoKeyZip(void       *zippedKey,
                              size_t      zippedKeyDataSize,
                              const void *key,
                              size_t      keyDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void       *cudaStreamPtr)
{
    return -1;
}

int FastFss_cuda_grottoKeyUnzip(void       *key,
                                size_t      keyDataSize,
                                const void *zippedKey,
                                size_t      zippedKeyDataSize,
                                size_t      bitWidthIn,
                                size_t      elementSize,
                                size_t      elementNum,
                                void       *cudaStreamPtr)
{
    return -1;
}

int FastFss_cuda_grottoGetKeyDataSize(size_t *keyDataSize,
                                      size_t  bitWidthIn,
                                      size_t  elementSize,
                                      size_t  elementNum)
{
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetKeyDataSize<scalar_t>(bitWidthIn, elementNum);
        });
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_grottoGetZippedKeyDataSize(size_t *keyDataSize,
                                            size_t  bitWidthIn,
                                            size_t  elementSize,
                                            size_t  elementNum)
{
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetZippedKeyDataSize<scalar_t>(bitWidthIn,
                                                              elementNum);
        });
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_grottoGetCacheDataSize(size_t *cacheDataSize,
                                        size_t  bitWidthIn,
                                        size_t  elementSize,
                                        size_t  elementNum)
{
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetCacheDataSize<scalar_t>(bitWidthIn,
                                                          elementNum);
        });
    return FAST_FSS_SUCCESS;
}