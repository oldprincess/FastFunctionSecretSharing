#include <FastFss/cpu/grotto.h>
#include <FastFss/errors.h>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../impl/grotto.h"
#include "../kernel/grotto.h"
#include "../kernel/parallel_execute.h"

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
                .key           = key,
                .keyDataSize   = keyDataSize,
                .alpha         = alpha,
                .alphaDataSize = alphaDataSize,
                .seed0         = seed0,
                .seedDataSize0 = seedDataSize0,
                .seed1         = seed1,
                .seedDataSize1 = seedDataSize1,
                .bitWidthIn    = bitWidthIn,
                .elementSize   = elementSize,
                .elementNum    = elementNum,
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
                .sharedBooleanOut  = sharedBooleanOut,
                .sharedOutDataSize = sharedOutDataSize,
                .maskedX           = maskedX,
                .maskedXDataSize   = maskedXDataSize,
                .key               = key,
                .keyDataSize       = keyDataSize,
                .seed              = seed,
                .seedDataSize      = seedDataSize,
                .partyId           = partyId,
                .bitWidthIn        = bitWidthIn,
                .elementSize       = elementSize,
                .elementNum        = elementNum,
                .cache             = cache,
                .cacheDataSize     = cacheDataSize,
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
                .sharedBooleanOut  = sharedBooleanOut,
                .sharedOutDataSize = sharedOutDataSize,
                .maskedX           = maskedX,
                .maskedXDataSize   = maskedXDataSize,
                .key               = key,
                .keyDataSize       = keyDataSize,
                .seed              = seed,
                .seedDataSize      = seedDataSize,
                .equalBound        = equalBound,
                .partyId           = partyId,
                .bitWidthIn        = bitWidthIn,
                .elementSize       = elementSize,
                .elementNum        = elementNum,
                .cache             = cache,
                .cacheDataSize     = cacheDataSize,
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
                .sharedBooleanOut         = sharedBooleanOut,
                .sharedBooleanOutDataSize = sharedBooleanOutDataSize,
                .maskedX                  = maskedX,
                .maskedXDataSize          = maskedXDataSize,
                .key                      = key,
                .keyDataSize              = keyDataSize,
                .seed                     = seed,
                .seedDataSize             = seedDataSize,
                .partyId                  = partyId,
                .leftEndpoints            = leftEndpoints,
                .leftEndpointsDataSize    = leftEndpointsDataSize,
                .rightEndpoints           = rightEndpoints,
                .rightEndpointsDataSize   = rightEndpointsDataSize,
                .bitWidthIn               = bitWidthIn,
                .elementSize              = elementSize,
                .elementNum               = elementNum,
                .cache                    = cache,
                .cacheDataSize            = cacheDataSize,
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
                .sharedOutE             = sharedOutE,
                .sharedOutEDataSize     = sharedOutEDataSize,
                .sharedOutT             = sharedOutT,
                .sharedOutTDataSize     = sharedOutTDataSize,
                .maskedX                = maskedX,
                .maskedXDataSize        = maskedXDataSize,
                .key                    = key,
                .keyDataSize            = keyDataSize,
                .seed                   = seed,
                .seedDataSize           = seedDataSize,
                .partyId                = partyId,
                .leftEndpoints          = leftEndpoints,
                .leftEndpointsDataSize  = leftEndpointsDataSize,
                .rightEndpoints         = rightEndpoints,
                .rightEndpointsDataSize = rightEndpointsDataSize,
                .lookUpTable            = lookUpTable,
                .lookUpTableDataSize    = lookUpTableDataSize,
                .bitWidthIn             = bitWidthIn,
                .bitWidthOut            = bitWidthOut,
                .elementSize            = elementSize,
                .elementNum             = elementNum,
                .cache                  = cache,
                .cacheDataSize          = cacheDataSize,
            };
            return kernel::parallel_execute(task);
        });
}
