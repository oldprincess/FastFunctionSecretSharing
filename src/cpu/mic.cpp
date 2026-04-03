#include <FastFss/cpu/mic.h>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../impl/mic.h"
#include "../kernel/mic.h"
#include "../kernel/parallel_execute.h"

using namespace FastFss;

int FastFss_cpu_dcfMICKeyGen(void       *key,
                             size_t      keyDataSize,
                             void       *z,
                             size_t      zDataSize,
                             const void *alpha,
                             size_t      alphaDataSize,
                             const void *seed0,
                             size_t      seedDataSize0,
                             const void *seed1,
                             size_t      seedDataSize1,
                             const void *leftEndpoints,
                             size_t      leftEndpointsDataSize,
                             const void *rightEndpoints,
                             size_t      rightEndpointsDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      elementSize,
                             size_t      elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DcfMICKeyGenTask<scalar_t> task{
                .key                    = key,
                .keyDataSize            = keyDataSize,
                .z                      = z,
                .zDataSize              = zDataSize,
                .alpha                  = alpha,
                .alphaDataSize          = alphaDataSize,
                .seed0                  = seed0,
                .seedDataSize0          = seedDataSize0,
                .seed1                  = seed1,
                .seedDataSize1          = seedDataSize1,
                .leftEndpoints          = leftEndpoints,
                .leftEndpointsDataSize  = leftEndpointsDataSize,
                .rightEndpoints         = rightEndpoints,
                .rightEndpointsDataSize = rightEndpointsDataSize,
                .bitWidthIn             = bitWidthIn,
                .bitWidthOut            = bitWidthOut,
                .elementSize            = elementSize,
                .elementNum             = elementNum,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_dcfMICEval(void       *sharedOut,
                           size_t      sharedOutDataSize,
                           const void *maskedX,
                           size_t      maskedXDataSize,
                           const void *key,
                           size_t      keyDataSize,
                           const void *sharedZ,
                           size_t      sharedZDataSize,
                           const void *seed,
                           size_t      seedDataSize,
                           int         partyId,
                           const void *leftEndpoints,
                           size_t      leftEndpointsDataSize,
                           const void *rightEndpoints,
                           size_t      rightEndpointsDataSize,
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
            kernel::DcfMICEvalTask<scalar_t> task{
                .sharedOut              = sharedOut,
                .sharedOutDataSize      = sharedOutDataSize,
                .maskedX                = maskedX,
                .maskedXDataSize        = maskedXDataSize,
                .key                    = key,
                .keyDataSize            = keyDataSize,
                .sharedZ                = sharedZ,
                .sharedZDataSize        = sharedZDataSize,
                .seed                   = seed,
                .seedDataSize           = seedDataSize,
                .partyId                = partyId,
                .leftEndpoints          = leftEndpoints,
                .leftEndpointsDataSize  = leftEndpointsDataSize,
                .rightEndpoints         = rightEndpoints,
                .rightEndpointsDataSize = rightEndpointsDataSize,
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
