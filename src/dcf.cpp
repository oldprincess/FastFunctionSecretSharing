#include "impl/dcf.h"

#include <FastFss/dcf.h>
#include <FastFss/errors.h>

#include "impl/def.h"

using namespace FastFss;

int FastFss_dcfKeyZip(void       *zippedKey,
                      size_t      zippedKeyDataSize,
                      const void *key,
                      size_t      keyDataSize,
                      size_t      bitWidthIn,
                      size_t      bitWidthOut,
                      size_t      groupSize,
                      size_t      elementSize,
                      size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_dcfKeyUnzip(void       *key,
                        size_t      keyDataSize,
                        const void *zippedKey,
                        size_t      zippedKeyDataSize,
                        size_t      bitWidthIn,
                        size_t      bitWidthOut,
                        size_t      groupSize,
                        size_t      elementSize,
                        size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_dcfGetKeyDataSize(size_t *keyDataSize,
                              size_t  bitWidthIn,
                              size_t  bitWidthOut,
                              size_t  groupSize,
                              size_t  elementSize,
                              size_t  elementNum)
{
    if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dcfGetKeyDataSize<scalar_t>(          //
                bitWidthIn, bitWidthOut, groupSize, elementNum //
            );                                                 //
        });
    if (*keyDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_dcfGetZippedKeyDataSize(size_t *keyDataSize,
                                    size_t  bitWidthIn,
                                    size_t  bitWidthOut,
                                    size_t  groupSize,
                                    size_t  elementSize,
                                    size_t  elementNum)
{
    if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dcfGetZippedKeyDataSize<scalar_t>(    //
                bitWidthIn, bitWidthOut, groupSize, elementNum //
            );                                                 //
        });
    if (*keyDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_dcfGetCacheDataSize(size_t *cacheDataSize,
                                size_t  bitWidthIn,
                                size_t  bitWidthOut,
                                size_t  groupSize,
                                size_t  elementSize,
                                size_t  elementNum)
{
    if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dcfGetCacheDataSize<scalar_t>( //
                bitWidthIn, groupSize, elementNum       //
            );                                          //
        });
    if (*cacheDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}
