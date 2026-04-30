#include "impl/spline.h"

#include <FastFss/errors.h>
#include <FastFss/spline.h>

#include "impl/def.h"

using namespace FastFss;

int FastFss_dcfSplineKeyZip(void       *zippedKey,
                            size_t      zippedKeyDataSize,
                            const void *key,
                            size_t      keyDataSize,
                            size_t      degree,
                            size_t      intervalNum,
                            size_t      bitWidthIn,
                            size_t      bitWidthOut,
                            size_t      elementSize,
                            size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_dcfSplineKeyUnzip(void       *key,
                              size_t      keyDataSize,
                              const void *zippedKey,
                              size_t      zippedKeyDataSize,
                              size_t      degree,
                              size_t      intervalNum,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      elementSize,
                              size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_dcfSplineGetCacheDataSize(size_t *cacheDataSize,
                                      size_t  degree,
                                      size_t  intervalNum,
                                      size_t  bitWidthIn,
                                      size_t  bitWidthOut,
                                      size_t  elementSize,
                                      size_t  elementNum)
{
    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] { return impl::dcfSplineGetCacheDataSize<scalar_t>(bitWidthIn, bitWidthOut, intervalNum, degree, elementNum); });
    return FAST_FSS_SUCCESS;
}

int FastFss_dcfSplineGetKeyDataSize(size_t *keyDataSize,
                                    size_t  degree,
                                    size_t  intervalNum,
                                    size_t  bitWidthIn,
                                    size_t  bitWidthOut,
                                    size_t  elementSize,
                                    size_t  elementNum)
{
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] { return impl::dcfSplineGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut, intervalNum, degree, elementNum); });
    return FAST_FSS_SUCCESS;
}

int FastFss_dcfSplineGetZippedKeyDataSize(size_t *keyDataSize,
                                          size_t  degree,
                                          size_t  intervalNum,
                                          size_t  bitWidthIn,
                                          size_t  bitWidthOut,
                                          size_t  elementSize,
                                          size_t  elementNum)
{
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dcfGetZippedKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut, intervalNum * (degree + 1),
                                                           elementNum);
        });
    return FAST_FSS_SUCCESS;
}
