#include <FastFss/cpu/config.h>
#include <FastFss/cpu/dcf.h>
#include <omp.h>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../helper/dcf_helper.h"
#include "../impl/dcf.h"

using namespace FastFss;

template <typename GroupElement>
static void dcfKeyGenKernel(void       *key,
                            const void *alpha,
                            const void *beta,
                            const void *seed0,
                            const void *seed1,
                            std::size_t bitWidthIn,
                            std::size_t bitWidthOut,
                            std::size_t groupSize,
                            std::size_t elementNum)
{
    const GroupElement ONE = 1;

    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *alphaPtr = (const GroupElement *)alpha;
    const GroupElement *betaPtr  = (const GroupElement *)beta;
    const std::uint8_t *seed0Ptr = (const std::uint8_t *)seed0;
    const std::uint8_t *seed1Ptr = (const std::uint8_t *)seed1;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::DcfKey<GroupElement> keyObj;
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, groupSize, i,
                           elementNum);
        const GroupElement *ptr = &ONE;
        if (betaPtr != nullptr)
        {
            ptr = betaPtr + groupSize * i;
        }
        impl::dcfKeyGen(keyObj,            //
                        alphaPtr[i],       //
                        ptr,               //
                        seed0Ptr + 16 * i, //
                        seed1Ptr + 16 * i, //
                        bitWidthIn,        //
                        bitWidthOut,       //
                        groupSize          //
        );
    }
}

int FastFss_cpu_dcfKeyGen(void       *key,
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
    int ret = FastFss_helper_checkDcfKeyGenParams( //
        keyDataSize,                               //
        alphaDataSize,                             //
        betaDataSize,                              //
        seedDataSize0,                             //
        seedDataSize1,                             //
        bitWidthIn,                                //
        bitWidthOut,                               //
        groupSize,                                 //
        elementSize,                               //
        elementNum,                                //
        FastFss_cpu_dcfGetKeyDataSize              //
    );

    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dcfKeyGenKernel<scalar_t>(           //
                key,                             //
                alpha,                           //
                (betaDataSize) ? beta : nullptr, //
                seed0,                           //
                seed1,                           //
                bitWidthIn,                      //
                bitWidthOut,                     //
                groupSize,                       //
                elementNum                       //
            );

            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void dcfEvalKernel(void       *sharedOut,
                          const void *maskedX,
                          const void *key,
                          const void *seed,
                          int         partyId,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      groupSize,
                          size_t      elementNum,
                          void       *cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    GroupElement       *sharedOutPtr = (GroupElement *)sharedOut;
    const GroupElement *maskedXPtr   = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr      = (const std::uint8_t *)seed;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::DcfKey<GroupElement> keyObj;
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, groupSize, i,
                           elementNum);
        impl::dcfEval(sharedOutPtr + i * groupSize, //
                      keyObj,                       //
                      maskedXPtr[i],                //
                      seedPtr + 16 * i,             //
                      partyId,                      //
                      bitWidthIn,                   //
                      bitWidthOut,                  //
                      groupSize                     //
        );
    }
}

int FastFss_cpu_dcfEval(void       *sharedOut,
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
                        size_t      cacheDataSize)
{
    int ret = FastFss_helper_checkDcfEvalParams( //
        sharedOutSize,                           //
        maskedXDataSize,                         //
        keyDataSize,                             //
        seedDataSize,                            //
        partyId,                                 //
        bitWidthIn,                              //
        bitWidthOut,                             //
        groupSize,                               //
        elementSize,                             //
        elementNum,                              //
        cacheDataSize,                           //
        FastFss_cpu_dcfGetKeyDataSize,           //
        FastFss_cpu_dcfGetCacheDataSize          //
    );

    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dcfEvalKernel<scalar_t>(sharedOut, maskedX, key, seed, partyId,
                                    bitWidthIn, bitWidthOut, groupSize,
                                    elementNum, cache);
            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cpu_dcfKeyZip(void       *zippedKey,
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

int FastFss_cpu_dcfKeyUnzip(void       *key,
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

int FastFss_cpu_dcfGetKeyDataSize(size_t *keyDataSize,
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

int FastFss_cpu_dcfGetZippedKeyDataSize(size_t *keyDataSize,
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

int FastFss_cpu_dcfGetCacheDataSize(size_t *cacheDataSize,
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