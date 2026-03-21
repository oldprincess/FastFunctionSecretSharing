#include <FastFss/cuda/config.h>
#include <FastFss/cuda/grotto.h>
#include <FastFss/errors.h>

#include "../impl/grotto.h"
#include "../kernel/grotto.h"
#include "grotto/eqMulti.cuh"
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
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::GrottoKeyGenTask<scalar_t> task{
                key,        keyDataSize,   alpha,      alphaDataSize,
                seed0,      seedDataSize0, seed1,      seedDataSize1,
                bitWidthIn, elementSize,   elementNum, cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
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
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::GrottoEvalTask<scalar_t> task{
                sharedBooleanOut,
                sharedOutDataSize,
                maskedX,
                maskedXDataSize,
                key,
                keyDataSize,
                seed,
                seedDataSize,
                equalBound,
                partyId,
                bitWidthIn,
                elementSize,
                elementNum,
                cache,
                cacheDataSize,
                cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
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
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::GrottoEqEvalTask<scalar_t> task{
                sharedBooleanOut,
                sharedOutDataSize,
                maskedX,
                maskedXDataSize,
                key,
                keyDataSize,
                seed,
                seedDataSize,
                partyId,
                bitWidthIn,
                elementSize,
                elementNum,
                cache,
                cacheDataSize,
                cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
        });
}

int FastFss_cuda_grottoEqEvalMulti(void       *sharedBooleanOut,
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
    int ret = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::GrottoEqEvalMultiTask<scalar_t> task{
                sharedBooleanOut,
                sharedOutDataSize,
                maskedX,
                maskedXDataSize,
                key,
                keyDataSize,
                seed,
                seedDataSize,
                partyId,
                point,
                pointDataSize,
                bitWidthIn,
                elementSize,
                elementNum,
                cache,
                cacheDataSize,
                cudaStreamPtr,
            };
            return task.check();
        });
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t *)cudaStreamPtr : 0;
    std::size_t  block  = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t  grid   = (elementNum + block - 1) / block;
    bool         parallelAll = false;
    if (grid > CUDA_MAX_GRID_DIM)
    {
        grid = CUDA_MAX_GRID_DIM;
    }
    else if (grid < FastFss_cuda_getGridDim())
    {
        parallelAll   = true;
        std::size_t n = elementNum * (pointDataSize / elementSize);
        grid          = (n + block - 1) / block;
        if (grid > CUDA_MAX_GRID_DIM)
        {
            grid = CUDA_MAX_GRID_DIM;
        }
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            if (parallelAll)
            {
                grottoEqMultiEvalParallelAllKernel<scalar_t>
                    <<<grid, block, 0, stream>>>(
                        sharedBooleanOut, maskedX, key, seed, partyId, point,
                        pointDataSize / elementSize, bitWidthIn, elementNum);
                if (cudaPeekAtLastError() != cudaSuccess)
                {
                    return (int)FAST_FSS_RUNTIME_ERROR;
                }
                return (int)FAST_FSS_SUCCESS;
            }

            kernel::GrottoEqEvalMultiTask<scalar_t> task{
                sharedBooleanOut,
                sharedOutDataSize,
                maskedX,
                maskedXDataSize,
                key,
                keyDataSize,
                seed,
                seedDataSize,
                partyId,
                point,
                pointDataSize,
                bitWidthIn,
                elementSize,
                elementNum,
                cache,
                cacheDataSize,
                cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
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
                               const void *leftEndpoints,
                               size_t      leftEndpointsDataSize,
                               const void *rightEndpoints,
                               size_t      rightEndpointsDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum,
                               void       *cache,
                               size_t      cacheDataSize,
                               void       *cudaStreamPtr)
{
    int ret = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::GrottoMICEvalTask<scalar_t> task{
                sharedBooleanOut,
                sharedBooleanOutDataSize,
                maskedX,
                maskedXDataSize,
                key,
                keyDataSize,
                seed,
                seedDataSize,
                partyId,
                leftEndpoints,
                leftEndpointsDataSize,
                rightEndpoints,
                rightEndpointsDataSize,
                bitWidthIn,
                elementSize,
                elementNum,
                cache,
                cacheDataSize,
                cudaStreamPtr,
            };
            return task.check();
        });
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    std::size_t intervalNum = leftEndpointsDataSize / elementSize;
    std::size_t block       = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t grid        = (elementNum + block - 1) / block;
    bool        parallelAll = false;
    if (grid > CUDA_MAX_GRID_DIM)
    {
        grid = CUDA_MAX_GRID_DIM;
    }
    else if (grid < FastFss_cuda_getGridDim())
    {
        parallelAll = true;
        grid        = (elementNum * intervalNum + block - 1) / block;
        if (grid > CUDA_MAX_GRID_DIM)
        {
            grid = CUDA_MAX_GRID_DIM;
        }
    }
    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t *)cudaStreamPtr : 0;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            if (parallelAll)
            {
                grottoMICEvalParallelAllKernel<scalar_t>
                    <<<grid, block, 0, stream>>>(sharedBooleanOut, maskedX, key,
                                                 seed, partyId, leftEndpoints,
                                                 rightEndpoints, intervalNum,
                                                 bitWidthIn, elementNum);
                if (cudaPeekAtLastError() != cudaSuccess)
                {
                    return (int)FAST_FSS_RUNTIME_ERROR;
                }
                return (int)FAST_FSS_SUCCESS;
            }

            kernel::GrottoMICEvalTask<scalar_t> task{
                sharedBooleanOut,
                sharedBooleanOutDataSize,
                maskedX,
                maskedXDataSize,
                key,
                keyDataSize,
                seed,
                seedDataSize,
                partyId,
                leftEndpoints,
                leftEndpointsDataSize,
                rightEndpoints,
                rightEndpointsDataSize,
                bitWidthIn,
                elementSize,
                elementNum,
                cache,
                cacheDataSize,
                cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
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
    return FastFss_cuda_grottoLutEval_ex(
        sharedOutE, sharedOutEDataSize, sharedOutT, sharedOutTDataSize, maskedX,
        maskedXDataSize, key, keyDataSize, seed, seedDataSize, partyId,
        lookUpTable, lookUpTableDataSize, bitWidthIn, bitWidthIn, bitWidthOut,
        elementSize, elementNum, cache, cacheDataSize, cudaStreamPtr);
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
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::GrottoLutEvalExTask<scalar_t> task{
                sharedOutE,
                sharedOutEDataSize,
                sharedOutT,
                sharedOutTDataSize,
                maskedX,
                maskedXDataSize,
                key,
                keyDataSize,
                seed,
                seedDataSize,
                partyId,
                lookUpTable,
                lookUpTableDataSize,
                lutBitWidth,
                bitWidthIn,
                bitWidthOut,
                elementSize,
                elementNum,
                cache,
                cacheDataSize,
                cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
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
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::GrottoLutEvalEx2Task<scalar_t> task{
                sharedOutE,
                sharedOutEDataSize,
                sharedOutT,
                sharedOutTDataSize,
                maskedX,
                maskedXDataSize,
                key,
                keyDataSize,
                seed,
                seedDataSize,
                partyId,
                points,
                pointsDataSize,
                lookUpTable,
                lookUpTableDataSize,
                bitWidthIn,
                bitWidthOut,
                elementSize,
                elementNum,
                cache,
                cacheDataSize,
                cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
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
                                       const void *leftEndpoints,
                                       size_t      leftEndpointsDataSize,
                                       const void *rightEndpoints,
                                       size_t      rightEndpointsDataSize,
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
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::GrottoIntervalLutEvalTask<scalar_t> task{
                sharedOutE,
                sharedOutEDataSize,
                sharedOutT,
                sharedOutTDataSize,
                maskedX,
                maskedXDataSize,
                key,
                keyDataSize,
                seed,
                seedDataSize,
                partyId,
                leftEndpoints,
                leftEndpointsDataSize,
                rightEndpoints,
                rightEndpointsDataSize,
                lookUpTable,
                lookUpTableDataSize,
                bitWidthIn,
                bitWidthOut,
                elementSize,
                elementNum,
                cache,
                cacheDataSize,
                cudaStreamPtr,
            };
            return (int)kernel::parallel_execute(task);
        });
}

int FastFss_cuda_grottoKeyZip(void       *zippedKey,
                              size_t      zippedKeyDataSize,
                              const void *key,
                              size_t      keyDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cuda_grottoKeyUnzip(void       *key,
                                size_t      keyDataSize,
                                const void *zippedKey,
                                size_t      zippedKeyDataSize,
                                size_t      bitWidthIn,
                                size_t      elementSize,
                                size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cuda_grottoGetKeyDataSize(size_t *keyDataSize,
                                      size_t  bitWidthIn,
                                      size_t  elementSize,
                                      size_t  elementNum)
{
    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
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
    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
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
    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
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
