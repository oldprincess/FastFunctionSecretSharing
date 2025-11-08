#include <FastFss/cpu/config.h>
#include <FastFss/cpu/dpf.h>
#include <omp.h>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../helper/dpf_helper.h"
#include "../helper/error_code.h"
#include "../impl/dpf.h"

using namespace FastFss;

template <typename GroupElement>
static void dpfKeyGenKernel(void       *key,
                            const void *alpha,
                            const void *beta,
                            const void *seed0,
                            const void *seed1,
                            std::size_t bitWidthIn,
                            std::size_t bitWidthOut,
                            std::size_t groupSize,
                            std::size_t elementNum)
{
    static const GroupElement ONE = 1;

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
        impl::DpfKey<GroupElement> keyObj;
        impl::dpfKeySetPtr(                                                //
            keyObj, key, bitWidthIn, bitWidthOut, groupSize, i, elementNum //
        );
        const GroupElement *ptr = &ONE;
        if (betaPtr != nullptr)
        {
            ptr = betaPtr + groupSize * i;
        }
        impl::dpfKeyGen(keyObj,            //
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
    int ret = FastFss_helper_checkDpfKeyGenParams(
        keyDataSize, alphaDataSize, betaDataSize, seedDataSize0, seedDataSize1,
        bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum,
        FastFss_cpu_dpfGetKeyDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dpfKeyGenKernel<scalar_t>(
                key, alpha, (betaDataSize) ? beta : nullptr, seed0, seed1,
                bitWidthIn, bitWidthOut, groupSize, elementNum);

            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void dpfEvalKernel(void       *sharedOut,
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
        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement> *cacheObjPtr = nullptr;
        impl::dpfKeySetPtr(                                                //
            keyObj, key, bitWidthIn, bitWidthOut, groupSize, i, elementNum //
        );                                                                 //
        if (cache != nullptr)
        {
            impl::dpfCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        impl::dpfEval(sharedOutPtr + i * groupSize, //
                      keyObj,                       //
                      maskedXPtr[i],                //
                      seedPtr + 16 * i,             //
                      partyId,                      //
                      bitWidthIn,                   //
                      bitWidthOut,                  //
                      groupSize,                    //
                      cacheObjPtr                   //
        );
    }
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
    int ret = FastFss_helper_checkDpfEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize,
        cacheDataSize, partyId, bitWidthIn, bitWidthOut, groupSize, elementSize,
        elementNum, FastFss_cpu_dpfGetKeyDataSize,
        FastFss_cpu_dpfGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dpfEvalKernel<scalar_t>(sharedOut, maskedX, key, seed, partyId,
                                    bitWidthIn, bitWidthOut, groupSize,
                                    elementNum, cache);
            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void dpfEvalAllKernel(void       *sharedOut,
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
        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement> *cacheObjPtr = nullptr;
        impl::dpfKeySetPtr(                                                //
            keyObj, key, bitWidthIn, bitWidthOut, groupSize, i, elementNum //
        );                                                                 //
        if (cache != nullptr)
        {
            impl::dpfCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        std::size_t size = (std::size_t)(1ULL << bitWidthIn);
        for (std::size_t j = 0; j < size; j++)
        {
            std::size_t k =
                groupSize * (std::size_t)((maskedXPtr[i] - j) % size);
            impl::dpfEval(sharedOutPtr + size * i * groupSize + k,
                          keyObj,           //
                          (GroupElement)j,  //
                          seedPtr + 16 * i, //
                          partyId,          //
                          bitWidthIn,       //
                          bitWidthOut,      //
                          groupSize,        //
                          cacheObjPtr       //
            );
        }
    }
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
    int ret = FastFss_helper_checkDpfEvalAllParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize,
        cacheDataSize, partyId, bitWidthIn, bitWidthOut, groupSize, elementSize,
        elementNum, FastFss_cpu_dpfGetKeyDataSize,
        FastFss_cpu_dpfGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dpfEvalAllKernel<scalar_t>( //
                sharedOut,              //
                maskedX,                //
                key,                    //
                seed,                   //
                partyId,                //
                bitWidthIn,             //
                bitWidthOut,            //
                groupSize,              //
                elementNum,             //
                cache                   //
            );                          //
            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void dpfMultiEvalKernel(void       *sharedOut,
                               const void *maskedX,
                               const void *key,
                               const void *seed,
                               int         partyId,
                               const void *point,
                               size_t      pointNum,
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
    const GroupElement *pointPtr     = (const GroupElement *)point;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement> *cacheObjPtr = nullptr;
        impl::dpfKeySetPtr(                                                //
            keyObj, key, bitWidthIn, bitWidthOut, groupSize, i, elementNum //
        );                                                                 //
        if (cache != nullptr)
        {
            impl::dpfCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        for (std::size_t j = 0; j < pointNum; j++)
        {
            GroupElement tmp = maskedXPtr[i] - pointPtr[j];
            impl::dpfEval(                                               //
                sharedOutPtr + pointNum * i * groupSize + j * groupSize, //
                keyObj,                                                  //
                tmp,                                                     //
                seedPtr + 16 * i,                                        //
                partyId,                                                 //
                bitWidthIn,                                              //
                bitWidthOut,                                             //
                groupSize,                                               //
                cacheObjPtr                                              //
            );                                                           //
        }
    }
}

int FastFss_cpu_dpfMultiEval(void       *sharedOut,
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
    int ret = FastFss_helper_checkDpfMultiEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize,
        pointDataSize, cacheDataSize, partyId, bitWidthIn, bitWidthOut,
        groupSize, elementSize, elementNum, FastFss_cpu_dpfGetKeyDataSize,
        FastFss_cpu_dpfGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dpfMultiEvalKernel<scalar_t>(    //
                sharedOut,                   //
                maskedX,                     //
                key,                         //
                seed,                        //
                partyId,                     //
                point,                       //
                pointDataSize / elementSize, //
                bitWidthIn,                  //
                bitWidthOut,                 //
                groupSize,                   //
                elementNum,                  //
                cache                        //
            );                               //
            return FAST_FSS_SUCCESS;
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
            return impl::dpfGetKeyDataSize<scalar_t>(          //
                bitWidthIn, bitWidthOut, groupSize, elementNum //
            );                                                 //
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
            return impl::dpfGetZippedKeyDataSize<scalar_t>(    //
                bitWidthIn, bitWidthOut, groupSize, elementNum //
            );                                                 //
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