#include "impl/mic.h"

#include <FastFss/errors.h>
#include <FastFss/mic.h>

#include "impl/def.h"

using namespace FastFss;

int FastFss_dcfMICKeyZip(void       *zippedKey,
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

int FastFss_dcfMICKeyUnzip(void       *key,
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

int FastFss_dcfMICGetCacheDataSize(size_t *cacheDataSize,
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

int FastFss_dcfMICGetKeyDataSize(size_t *keyDataSize,
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

int FastFss_dcfMICGetZippedKeyDataSize(size_t *keyDataSize,
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
