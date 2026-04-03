#include "impl/grotto.h"

#include <FastFss/errors.h>
#include <FastFss/grotto.h>

#include "impl/def.h"

using namespace FastFss;

int FastFss_grottoKeyZip(void       *zippedKey,
                         size_t      zippedKeyDataSize,
                         const void *key,
                         size_t      keyDataSize,
                         size_t      bitWidthIn,
                         size_t      elementSize,
                         size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_grottoKeyUnzip(void       *key,
                           size_t      keyDataSize,
                           const void *zippedKey,
                           size_t      zippedKeyDataSize,
                           size_t      bitWidthIn,
                           size_t      elementSize,
                           size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_grottoGetKeyDataSize(size_t *keyDataSize,
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

int FastFss_grottoGetZippedKeyDataSize(size_t *keyDataSize,
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

int FastFss_grottoGetCacheDataSize(size_t *cacheDataSize,
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
