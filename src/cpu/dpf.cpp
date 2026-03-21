#include <FastFss/cpu/config.h>
#include <FastFss/cpu/dpf.h>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include <FastFss/errors.h>

#include "../impl/dpf.h"
#include "../kernel/dpf.h"

using namespace FastFss;

int FastFss_cpu_dpfKeyGen(void       *key,
                          size_t      keyDataSize,
                          const void *alpha,
                          size_t      alphaDataSize,
                          const void *beta,
                          size_t      betaDataSize,
                          const void *seed0,
                          size_t      seedDataSize0,
                          const void *seed1,
                          size_t      seedDataSize1,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      groupSize,
                          size_t      elementSize,
                          size_t      elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfKeyGenTask<scalar_t> task{
                key,       keyDataSize,   alpha,      alphaDataSize,
                beta,      betaDataSize,  seed0,      seedDataSize0,
                seed1,     seedDataSize1, bitWidthIn, bitWidthOut,
                groupSize, elementSize,   elementNum,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_dpfEval(void       *sharedOut,
                        size_t      sharedOutDataSize,
                        const void *maskedX,
                        size_t      maskedXDataSize,
                        const void *key,
                        size_t      keyDataSize,
                        const void *seed,
                        size_t      seedDataSize,
                        int         partyId,
                        size_t      bitWidthIn,
                        size_t      bitWidthOut,
                        size_t      groupSize,
                        size_t      elementSize,
                        size_t      elementNum,
                        void       *cache,
                        size_t      cacheDataSize)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfEvalTask<scalar_t> task{
                sharedOut,   sharedOutDataSize, maskedX,     maskedXDataSize,
                key,         keyDataSize,       seed,        seedDataSize,
                partyId,     bitWidthIn,        bitWidthOut, groupSize,
                elementSize, elementNum,        cache,       cacheDataSize,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_dpfEvalAll(void       *sharedOut,
                           size_t      sharedOutDataSize,
                           const void *maskedX,
                           size_t      maskedXDataSize,
                           const void *key,
                           size_t      keyDataSize,
                           const void *seed,
                           size_t      seedDataSize,
                           int         partyId,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      groupSize,
                           size_t      elementSize,
                           size_t      elementNum,
                           void       *cache,
                           size_t      cacheDataSize)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfEvalAllTask<scalar_t> task{
                sharedOut,   sharedOutDataSize, maskedX,     maskedXDataSize,
                key,         keyDataSize,       seed,        seedDataSize,
                partyId,     bitWidthIn,        bitWidthOut, groupSize,
                elementSize, elementNum,        cache,       cacheDataSize,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_dpfEvalMulti(void       *sharedOut,
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
                             size_t      bitWidthOut,
                             size_t      groupSize,
                             size_t      elementSize,
                             size_t      elementNum,
                             void       *cache,
                             size_t      cacheDataSize)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfEvalMultiTask<scalar_t> task{
                sharedOut,     sharedOutDataSize,
                maskedX,       maskedXDataSize,
                key,           keyDataSize,
                seed,          seedDataSize,
                partyId,       point,
                pointDataSize, bitWidthIn,
                bitWidthOut,   groupSize,
                elementSize,   elementNum,
                cache,         cacheDataSize,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_dpfKeyZip(void       *zippedKey,
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

int FastFss_cpu_dpfKeyUnzip(void       *key,
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

int FastFss_cpu_dpfGetKeyDataSize(size_t *keyDataSize,
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

int FastFss_cpu_dpfGetZippedKeyDataSize(size_t *keyDataSize,
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

int FastFss_cpu_dpfGetCacheDataSize(size_t *cacheDataSize,
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
