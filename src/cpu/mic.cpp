#include <FastFss/cpu/config.h>
#include <FastFss/cpu/mic.h>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../impl/mic.h"
#include "../kernel/mic.h"

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
                key,
                keyDataSize,
                z,
                zDataSize,
                alpha,
                alphaDataSize,
                seed0,
                seedDataSize0,
                seed1,
                seedDataSize1,
                leftEndpoints,
                leftEndpointsDataSize,
                rightEndpoints,
                rightEndpointsDataSize,
                bitWidthIn,
                bitWidthOut,
                elementSize,
                elementNum,
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
                sharedOut,
                sharedOutDataSize,
                maskedX,
                maskedXDataSize,
                key,
                keyDataSize,
                sharedZ,
                sharedZDataSize,
                seed,
                seedDataSize,
                partyId,
                leftEndpoints,
                leftEndpointsDataSize,
                rightEndpoints,
                rightEndpointsDataSize,
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

int FastFss_cpu_dcfMICKeyZip(void       *zippedKey,
                             size_t      zippedKeyDataSize,
                             const void *key,
                             size_t      keyDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      elementSize,
                             size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cpu_dcfMICKeyUnzip(void       *key,
                               size_t      keyDataSize,
                               const void *zippedKey,
                               size_t      zippedKeyDataSize,
                               size_t      bitWidthIn,
                               size_t      bitWidthOut,
                               size_t      elementSize,
                               size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cpu_dcfMICGetCacheDataSize(size_t *cacheDataSize,
                                       size_t  bitWidthIn,
                                       size_t  bitWidthOut,
                                       size_t  elementSize,
                                       size_t  elementNum)
{
    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dcfGetCacheDataSize<scalar_t>(bitWidthIn, 1,
                                                       elementNum);
        });
    return FAST_FSS_SUCCESS;
}

int FastFss_cpu_dcfMICGetKeyDataSize(size_t *keyDataSize,
                                     size_t  bitWidthIn,
                                     size_t  bitWidthOut,
                                     size_t  elementSize,
                                     size_t  elementNum)
{
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dcfGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut, 1,
                                                     elementNum);
        });
    return FAST_FSS_SUCCESS;
}

int FastFss_cpu_dcfMICGetZippedKeyDataSize(size_t *keyDataSize,
                                           size_t  bitWidthIn,
                                           size_t  bitWidthOut,
                                           size_t  elementSize,
                                           size_t  elementNum)
{
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dcfGetZippedKeyDataSize<scalar_t>(
                bitWidthIn, bitWidthOut, 1, elementNum);
        });
    return FAST_FSS_SUCCESS;
}
