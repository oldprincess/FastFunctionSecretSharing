#include <FastFss/cuda/dcf.h>

#include "../impl/dcf.h"
#include "../kernel/dcf.h"

using namespace FastFss;
using namespace FastFss::kernel;

int FastFss_cuda_dcfKeyGen(void       *key,
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
                           size_t      elementNum,
                           void       *cudaStreamPtr)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DcfKeyGenTask<scalar_t> task{
                key,       keyDataSize,   alpha,      alphaDataSize,
                beta,      betaDataSize,  seed0,      seedDataSize0,
                seed1,     seedDataSize1, bitWidthIn, bitWidthOut,
                groupSize, elementSize,   elementNum, cudaStreamPtr,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cuda_dcfEval(void       *sharedOut,
                         size_t      sharedOutSize,
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
                         size_t      cacheDataSize,
                         void       *cudaStreamPtr)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DcfEvalTask<scalar_t> task{
                sharedOut,     sharedOutSize, maskedX,     maskedXDataSize,
                key,           keyDataSize,   seed,        seedDataSize,
                partyId,       bitWidthIn,    bitWidthOut, groupSize,
                elementSize,   elementNum,    cache,       cacheDataSize,
                cudaStreamPtr,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cuda_dcfKeyZip(void       *zippedKey,
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

int FastFss_cuda_dcfKeyUnzip(void       *key,
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

int FastFss_cuda_dcfGetKeyDataSize(size_t *keyDataSize,
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
            return impl::dcfGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut,
                                                     groupSize, elementNum);
        });
    if (*keyDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_dcfGetZippedKeyDataSize(size_t *keyDataSize,
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
            return impl::dcfGetZippedKeyDataSize<scalar_t>(
                bitWidthIn, bitWidthOut, groupSize, elementNum);
        });
    if (*keyDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_dcfGetCacheDataSize(size_t *cacheDataSize,
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
            return impl::dcfGetCacheDataSize<scalar_t>(bitWidthIn, groupSize,
                                                       elementNum);
        });
    if (*cacheDataSize == (std::size_t)(-1))
    {
        return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR;
    }
    return FAST_FSS_SUCCESS;
}
