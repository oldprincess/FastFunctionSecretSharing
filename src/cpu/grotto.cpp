#include <FastFss/cpu/config.h>
#include <FastFss/cpu/grotto.h>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include <FastFss/errors.h>

#include "../impl/grotto.h"
#include "../kernel/grotto.h"

using namespace FastFss;

int FastFss_cpu_grottoKeyGen(void       *key,
                             size_t      keyDataSize,
                             const void *alpha,
                             size_t      alphaDataSize,
                             const void *seed0,
                             size_t      seedDataSize0,
                             const void *seed1,
                             size_t      seedDataSize1,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::GrottoKeyGenTask<scalar_t> task{
                key,        keyDataSize,   alpha,      alphaDataSize,
                seed0,      seedDataSize0, seed1,      seedDataSize1,
                bitWidthIn, elementSize,   elementNum,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_grottoEval(void       *sharedBooleanOut,
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
                           size_t      cacheDataSize)
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
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_grottoEqEval(void       *sharedBooleanOut,
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
                             size_t      cacheDataSize)
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
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_grottoEqEvalMulti(void       *sharedBooleanOut,
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
                                  size_t      cacheDataSize)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
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
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_grottoMICEval(void       *sharedBooleanOut,
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
                              size_t      cacheDataSize)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
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
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_grottoLutEval(void       *sharedOutE,
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
                              size_t      cacheDataSize)
{
    return FastFss_cpu_grottoLutEval_ex(
        sharedOutE, sharedOutEDataSize, sharedOutT, sharedOutTDataSize, maskedX,
        maskedXDataSize, key, keyDataSize, seed, seedDataSize, partyId,
        lookUpTable, lookUpTableDataSize, bitWidthIn, bitWidthIn, bitWidthOut,
        elementSize, elementNum, cache, cacheDataSize);
}

int FastFss_cpu_grottoLutEval_ex(void       *sharedOutE,
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
                                 size_t      cacheDataSize)
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
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_grottoLutEval_ex2(void       *sharedOutE,
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
                                  size_t      cacheDataSize)
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
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_grottoIntervalLutEval(void       *sharedOutE,
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
                                      size_t      cacheDataSize)
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
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_grottoKeyZip(void       *zippedKey,
                             size_t      zippedKeyDataSize,
                             const void *key,
                             size_t      keyDataSize,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cpu_grottoKeyUnzip(void       *key,
                               size_t      keyDataSize,
                               const void *zippedKey,
                               size_t      zippedKeyDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cpu_grottoGetKeyDataSize(size_t *keyDataSize,
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

int FastFss_cpu_grottoGetZippedKeyDataSize(size_t *keyDataSize,
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

int FastFss_cpu_grottoGetCacheDataSize(size_t *cacheDataSize,
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
