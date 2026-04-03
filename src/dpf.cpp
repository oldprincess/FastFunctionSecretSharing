#include "impl/dpf.h"

#include <FastFss/dpf.h>
#include <FastFss/errors.h>

#include "impl/def.h"

using namespace FastFss;

int FastFss_dpfKeyZip(void       *zippedKey,
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

int FastFss_dpfKeyUnzip(void       *key,
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

int FastFss_dpfGetKeyDataSize(size_t *keyDataSize,
                              size_t  bitWidthIn,
                              size_t  bitWidthOut,
                              size_t  groupSize,
                              size_t  elementSize,
                              size_t  elementNum)
{
    if (bitWidthIn > elementSize * 8 || bitWidthOut > elementSize * 8)
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dpfGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut,
                                                     groupSize, elementNum);
        });
    if (*keyDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_dpfGetZippedKeyDataSize(size_t *keyDataSize,
                                    size_t  bitWidthIn,
                                    size_t  bitWidthOut,
                                    size_t  groupSize,
                                    size_t  elementSize,
                                    size_t  elementNum)
{
    if (bitWidthIn > elementSize * 8 || bitWidthOut > elementSize * 8)
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dpfGetZippedKeyDataSize<scalar_t>(
                bitWidthIn, bitWidthOut, groupSize, elementNum);
        });
    if (*keyDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_dpfGetCacheDataSize(size_t *cacheDataSize,
                                size_t  bitWidthIn,
                                size_t  elementSize,
                                size_t  elementNum)
{
    if (bitWidthIn > elementSize * 8)
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)(-1); },
        [&] {
            return impl::dpfGetCacheDataSize<scalar_t>(bitWidthIn, elementNum);
        });
    if (*cacheDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}
