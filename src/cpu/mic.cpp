#include <FastFss/cpu/config.h>
#include <FastFss/cpu/mic.h>
#include <omp.h>
#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../helper/mic_helper.h"
#include "../impl/mic.h"

using namespace FastFss;

template <typename GroupElement>
static void dcfMICKeyGenKernel(void       *key,
                               void       *z,
                               const void *alpha,
                               const void *seed0,
                               const void *seed1,
                               const void *leftBoundary,
                               const void *rightBoundary,
                               size_t      intervalNum,
                               size_t      bitWidthIn,
                               size_t      bitWidthOut,
                               size_t      elementNum)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    GroupElement       *zPtr             = (GroupElement *)z;
    const GroupElement *alphaPtr         = (const GroupElement *)alpha;
    const std::uint8_t *seed0Ptr         = (const std::uint8_t *)seed0;
    const std::uint8_t *seed1Ptr         = (const std::uint8_t *)seed1;
    const GroupElement *leftBoundaryPtr  = (const GroupElement *)leftBoundary;
    const GroupElement *rightBoundaryPtr = (const GroupElement *)rightBoundary;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::DcfKey<GroupElement> keyObj;
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, 1, i,
                           elementNum);
        impl::dcfMICKeyGen(keyObj,                 //
                           zPtr + intervalNum * i, //
                           alphaPtr[i],            //
                           seed0Ptr + 16 * i,      //
                           seed1Ptr + 16 * i,      //
                           leftBoundaryPtr,        //
                           rightBoundaryPtr,       //
                           intervalNum,            //
                           bitWidthIn,             //
                           bitWidthOut             //
        );
    }
}

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
                             const void *leftBoundary,
                             size_t      leftBoundaryDataSize,
                             const void *rightBoundary,
                             size_t      rightBoundaryDataSize,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      elementSize,
                             size_t      elementNum)
{
    int ret = FastFss_helper_checkDcfMicKeyGenParams(
        keyDataSize, zDataSize, alphaDataSize, seedDataSize0, seedDataSize1,
        leftBoundaryDataSize, rightBoundaryDataSize, bitWidthIn, bitWidthOut,
        elementSize, elementNum, FastFss_cpu_dcfMICGetKeyDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dcfMICKeyGenKernel<scalar_t>(           //
                key,                                //
                z,                                  //
                alpha,                              //
                seed0,                              //
                seed1,                              //
                leftBoundary,                       //
                rightBoundary,                      //
                leftBoundaryDataSize / elementSize, //
                bitWidthIn,                         //
                bitWidthOut,                        //
                elementNum                          //
            );                                      //
            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void dcfMICEvalKernel(void       *sharedOut,
                             const void *maskedX,
                             const void *key,
                             const void *sharedZ,
                             const void *seed,
                             int         partyId,
                             const void *leftBoundary,
                             const void *rightBoundary,
                             size_t      intervalNum,
                             size_t      bitWidthIn,
                             size_t      bitWidthOut,
                             size_t      elementSize,
                             size_t      elementNum,
                             void       *cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    GroupElement       *sharedOutPtr     = (GroupElement *)sharedOut;
    const GroupElement *maskedXPtr       = (const GroupElement *)maskedX;
    const GroupElement *sharedZPtr       = (const GroupElement *)sharedZ;
    const std::uint8_t *seedPtr          = (const std::uint8_t *)seed;
    const GroupElement *leftBoundaryPtr  = (const GroupElement *)leftBoundary;
    const GroupElement *rightBoundaryPtr = (const GroupElement *)rightBoundary;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::DcfKey<GroupElement>    keyObj;
        impl::DcfCache<GroupElement>  cacheObj;
        impl::DcfCache<GroupElement> *cachePtr = nullptr;
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, 1, i,
                           elementNum);
        if (cache != nullptr)
        {
            impl::dcfCacheSetPtr(cacheObj, cache, bitWidthIn, 1, i, elementNum);
            cachePtr = &cacheObj;
        }
        impl::dcfMICEval(sharedOutPtr + intervalNum * i, //
                         maskedXPtr[i],                  //
                         keyObj,                         //
                         sharedZPtr + intervalNum * i,   //
                         seedPtr + 16 * i,               //
                         partyId,                        //
                         leftBoundaryPtr,                //
                         rightBoundaryPtr,               //
                         intervalNum,                    //
                         bitWidthIn,                     //
                         bitWidthOut,                    //
                         cachePtr                        //
        );
    }
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
                           const void *leftBoundary,
                           size_t      leftBoundaryDataSize,
                           const void *rightBoundary,
                           size_t      rightBoundaryDataSize,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      elementSize,
                           size_t      elementNum,
                           void       *cache,
                           size_t      cacheDataSize)
{
    int ret = FastFss_helper_checkDcfMicEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, sharedZDataSize,
        seedDataSize, leftBoundaryDataSize, rightBoundaryDataSize, partyId,
        bitWidthIn, bitWidthOut, elementSize, elementNum, cacheDataSize,
        FastFss_cpu_dcfMICGetKeyDataSize, FastFss_cpu_dcfMICGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            auto intervalNum = leftBoundaryDataSize / elementSize;
            dcfMICEvalKernel<scalar_t>( //
                sharedOut,              //
                maskedX,                //
                key,                    //
                sharedZ,                //
                seed,                   //
                partyId,                //
                leftBoundary,           //
                rightBoundary,          //
                intervalNum,            //
                bitWidthIn,             //
                bitWidthOut,            //
                elementSize,            //
                elementNum,             //
                cache                   //
            );
            return FAST_FSS_SUCCESS;
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
