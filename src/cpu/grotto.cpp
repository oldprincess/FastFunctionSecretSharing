#include <FastFss/cpu/config.h>
#include <FastFss/cpu/grotto.h>
#include <omp.h>

#include "../helper/error_code.h"
#include "../helper/grotto_helper.h"

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../impl/grotto.h"

using namespace FastFss;

template <typename GroupElement>
static void grottoKeyGenKernel(void       *key,
                               const void *alpha,
                               const void *seed0,
                               const void *seed1,
                               size_t      bitWidthIn,
                               size_t      elementNum)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *alphaPtr = (const GroupElement *)alpha;
    const std::uint8_t *seed0Ptr = (const std::uint8_t *)seed0;
    const std::uint8_t *seed1Ptr = (const std::uint8_t *)seed1;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement> keyObj;
        std::size_t                   alphaOffset = i;
        std::size_t                   seed0Offset = 16 * i;
        std::size_t                   seed1Offset = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        impl::grottoKeyGen(keyObj,                 //
                           alphaPtr[alphaOffset],  //
                           seed0Ptr + seed0Offset, //
                           seed1Ptr + seed1Offset, //
                           bitWidthIn);            //
    }
}

int FastFss_cpu_grottoKeyGen(void       *key,
                             size_t      keyDataSize,
                             const void *alpha,
                             size_t      alphaDataSize,
                             const void *seed0,
                             size_t      seedDataSize0,
                             const void *seed1,
                             size_t      seedDataSize1,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    int ret = FastFss_helper_checkGrottoKeyGenParams(
        keyDataSize, alphaDataSize, seedDataSize0, seedDataSize1, bitWidthIn,
        elementSize, elementNum, FastFss_cpu_grottoGetKeyDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            grottoKeyGenKernel<scalar_t>                             //
                (                                                    //
                    key, alpha, seed0, seed1, bitWidthIn, elementNum //
                );                                                   //
            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void grottoEvalKernel(void       *out,
                             const void *maskedX,
                             const void *key,
                             const void *seed,
                             bool        equalBound,
                             int         partyId,
                             size_t      bitWidthIn,
                             size_t      elementNum,
                             void       *cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *maskedXPtr = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr    = (const std::uint8_t *)seed;
    GroupElement       *outPtr     = (GroupElement *)out;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>   keyObj;
        impl::GrottoCache<GroupElement> cacheObj;

        impl::GrottoCache<GroupElement> *cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        outPtr[i] = impl::grottoEval(keyObj,                    //
                                     maskedXPtr[maskedXOffset], //
                                     seedPtr + seedOffset,      //
                                     partyId,                   //
                                     bitWidthIn,                //
                                     equalBound,                //
                                     cacheObjPtr                //
        );
    }
}

int FastFss_cpu_grottoEval(void       *sharedBooleanOut,
                           size_t      sharedOutDataSize,
                           const void *maskedX,
                           size_t      maskedXDataSize,
                           const void *key,
                           size_t      keyDataSize,
                           const void *seed,
                           size_t      seedDataSize,
                           bool        equalBound,
                           int         partyId,
                           size_t      bitWidthIn,
                           size_t      elementSize,
                           size_t      elementNum,
                           void       *cache,
                           size_t      cacheDataSize)
{
    int ret = FastFss_helper_checkGrottoEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize, partyId,
        bitWidthIn, elementSize, elementNum, cacheDataSize,
        FastFss_cpu_grottoGetKeyDataSize, FastFss_cpu_grottoGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            grottoEvalKernel<scalar_t> //
                (                      //
                    sharedBooleanOut,  //
                    maskedX,           //
                    key,               //
                    seed,              //
                    equalBound,        //
                    partyId,           //
                    bitWidthIn,        //
                    elementNum,        //
                    cache              //
                );                     //

            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void grottoEqEvalKernel(void       *out,
                               const void *maskedX,
                               const void *key,
                               const void *seed,
                               int         partyId,
                               size_t      bitWidthIn,
                               size_t      elementNum,
                               void       *cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *maskedXPtr = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr    = (const std::uint8_t *)seed;
    GroupElement       *outPtr     = (GroupElement *)out;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement> *cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        outPtr[i] = impl::grottoEqEval(keyObj,                    //
                                       maskedXPtr[maskedXOffset], //
                                       seedPtr + seedOffset,      //
                                       partyId,                   //
                                       bitWidthIn,                //
                                       cacheObjPtr                //
        );
    }
}

int FastFss_cpu_grottoEqEval(void       *sharedBooleanOut,
                             size_t      sharedOutDataSize,
                             const void *maskedX,
                             size_t      maskedXDataSize,
                             const void *key,
                             size_t      keyDataSize,
                             const void *seed,
                             size_t      seedDataSize,
                             int         partyId,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum,
                             void       *cache,
                             size_t      cacheDataSize)
{
    int ret = FastFss_helper_checkGrottoEqEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize, partyId,
        bitWidthIn, elementSize, elementNum, cacheDataSize,
        FastFss_cpu_grottoGetKeyDataSize, FastFss_cpu_grottoGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            grottoEqEvalKernel<scalar_t> //
                (                        //
                    sharedBooleanOut,    //
                    maskedX,             //
                    key,                 //
                    seed,                //
                    partyId,             //
                    bitWidthIn,          //
                    elementNum,          //
                    cache                //
                );                       //

            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void grottoEqMultiEvalKernel(void       *out,
                                    const void *maskedX,
                                    const void *key,
                                    const void *seed,
                                    int         partyId,
                                    const void *point,
                                    size_t      pointNum,
                                    size_t      bitWidthIn,
                                    size_t      elementNum,
                                    void       *cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *maskedXPtr = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr    = (const std::uint8_t *)seed;
    GroupElement       *outPtr     = (GroupElement *)out;
    const GroupElement *pointPtr   = (const GroupElement *)point;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement> *cacheObjPtr = nullptr;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        for (std::size_t j = 0; j < pointNum; j++)
        {
            GroupElement tmp         = maskedXPtr[i] - pointPtr[j];
            outPtr[pointNum * i + j] = impl::grottoEqEval( //
                keyObj,                                    //
                tmp,                                       //
                seedPtr + 16 * i,                          //
                partyId,                                   //
                bitWidthIn,                                //
                cacheObjPtr);
        }
    }
}

int FastFss_cpu_grottoEqMultiEval(void       *sharedBooleanOut,
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
                                  size_t      elementSize,
                                  size_t      elementNum,
                                  void       *cache,
                                  size_t      cacheDataSize)
{
    int ret = FastFss_helper_checkGrottoEqMultiEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize, partyId,
        pointDataSize, bitWidthIn, elementSize, elementNum, cacheDataSize,
        FastFss_cpu_grottoGetKeyDataSize, FastFss_cpu_grottoGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            size_t pointNum = pointDataSize / elementSize;
            grottoEqMultiEvalKernel<scalar_t> //
                (                             //
                    sharedBooleanOut,         //
                    maskedX,                  //
                    key,                      //
                    seed,                     //
                    partyId,                  //
                    point,                    //
                    pointNum,                 //
                    bitWidthIn,               //
                    elementNum,               //
                    cache                     //
                );                            //
            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void grottoMICEvalKernel(void       *out,
                                const void *maskedX,
                                const void *key,
                                const void *seed,
                                int         partyId,
                                const void *leftBoundary,
                                const void *rightBoundary,
                                size_t      intervalNum,
                                size_t      bitWidthIn,
                                size_t      elementNum,
                                void       *cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *maskedXPtr       = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr          = (const std::uint8_t *)seed;
    GroupElement       *outPtr           = (GroupElement *)out;
    const GroupElement *leftBoundaryPtr  = (GroupElement *)leftBoundary;
    const GroupElement *rightBoundaryPtr = (GroupElement *)rightBoundary;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement> *cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        std::size_t                      outOffset     = intervalNum * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        impl::grottoMICEval(           //
            outPtr + outOffset,        //
            keyObj,                    //
            maskedXPtr[maskedXOffset], //
            seedPtr + seedOffset,      //
            partyId,                   //
            leftBoundaryPtr,           //
            rightBoundaryPtr,          //
            intervalNum,               //
            bitWidthIn,                //
            cacheObjPtr                //
        );
    }
}

int FastFss_cpu_grottoMICEval(void       *sharedBooleanOut,
                              size_t      sharedBooleanOutDataSize,
                              const void *maskedX,
                              size_t      maskedXDataSize,
                              const void *key,
                              size_t      keyDataSize,
                              const void *seed,
                              size_t      seedDataSize,
                              int         partyId,
                              const void *leftBoundary,
                              size_t      leftBoundaryDataSize,
                              const void *rightBoundary,
                              size_t      rightBoundaryDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void       *cache,
                              size_t      cacheDataSize)
{
    int ret = FastFss_helper_checkGrottoMICEvalParams(
        sharedBooleanOutDataSize, maskedXDataSize, keyDataSize, seedDataSize,
        partyId, leftBoundaryDataSize, rightBoundaryDataSize, bitWidthIn,
        elementSize, elementNum, cacheDataSize,
        FastFss_cpu_grottoGetKeyDataSize, FastFss_cpu_grottoGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            std::size_t intervalNum = leftBoundaryDataSize / elementSize;
            grottoMICEvalKernel<scalar_t> //
                (                         //
                    sharedBooleanOut,     //
                    maskedX,              //
                    key,                  //
                    seed,                 //
                    partyId,              //
                    leftBoundary,         //
                    rightBoundary,        //
                    intervalNum,          //
                    bitWidthIn,           //
                    elementNum,           //
                    cache                 //
                );                        //

            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void grottoLutEvalKernel_ex(void       *outE,
                                   void       *outT,
                                   const void *maskedX,
                                   const void *key,
                                   const void *seed,
                                   int         partyId,
                                   const void *lookUpTable,
                                   size_t      lutNum,
                                   size_t      lutBitWidth,
                                   size_t      bitWidthIn,
                                   size_t      elementNum,
                                   void       *cache)
{
    using namespace FastFss;

    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *maskedXPtr     = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr        = (const std::uint8_t *)seed;
    GroupElement       *outEPtr        = (GroupElement *)outE;
    GroupElement       *outTPtr        = (GroupElement *)outT;
    const GroupElement *lookUpTablePtr = (GroupElement *)lookUpTable;

    impl::AES128GlobalContext aesCtx;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement> *cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }

        impl::grottoLutEval_ex<GroupElement>( //
            outEPtr + i,                      //
            outTPtr + i * lutNum,             //
            keyObj,                           //
            maskedXPtr[maskedXOffset],        //
            seedPtr + seedOffset,             //
            partyId,                          //
            lookUpTablePtr,                   //
            lutNum,                           //
            lutBitWidth,                      //
            bitWidthIn,                       //
            cacheObjPtr,                      //
            &aesCtx                           //
        );
    }
}

int FastFss_cpu_grottoLutEval(void       *sharedOutE,
                              size_t      sharedOutEDataSize,
                              void       *sharedOutT,
                              size_t      sharedOutTDataSize,
                              const void *maskedX,
                              size_t      maskedXDataSize,
                              const void *key,
                              size_t      keyDataSize,
                              const void *seed,
                              size_t      seedDataSize,
                              int         partyId,
                              const void *lookUpTable,
                              size_t      lookUpTableDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      elementSize,
                              size_t      elementNum,
                              void       *cache,
                              size_t      cacheDataSize)
{
    return FastFss_cpu_grottoLutEval_ex(  //
        sharedOutE, sharedOutEDataSize,   //
        sharedOutT, sharedOutTDataSize,   //
        maskedX, maskedXDataSize,         //
        key, keyDataSize,                 //
        seed, seedDataSize,               //
        partyId,                          //
        lookUpTable, lookUpTableDataSize, //
        bitWidthIn,                       //
        bitWidthIn,                       //
        bitWidthOut,                      //
        elementSize,                      //
        elementNum,                       //
        cache, cacheDataSize              //
    );
}

int FastFss_cpu_grottoLutEval_ex(void       *sharedOutE,
                                 size_t      sharedOutEDataSize,
                                 void       *sharedOutT,
                                 size_t      sharedOutTDataSize,
                                 const void *maskedX,
                                 size_t      maskedXDataSize,
                                 const void *key,
                                 size_t      keyDataSize,
                                 const void *seed,
                                 size_t      seedDataSize,
                                 int         partyId,
                                 const void *lookUpTable,
                                 size_t      lookUpTableDataSize,
                                 size_t      lutBitWidth,
                                 size_t      bitWidthIn,
                                 size_t      bitWidthOut,
                                 size_t      elementSize,
                                 size_t      elementNum,
                                 void       *cache,
                                 size_t      cacheDataSize)
{
    int ret = FastFss_helper_checkGrottoLutEval_exParams(
        sharedOutEDataSize, sharedOutTDataSize, maskedXDataSize, keyDataSize,
        seedDataSize, partyId, lookUpTableDataSize, lutBitWidth, bitWidthIn,
        bitWidthOut, elementSize, elementNum, cacheDataSize,
        FastFss_cpu_grottoGetKeyDataSize, FastFss_cpu_grottoGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            std::size_t pointNum = (std::size_t)1 << lutBitWidth;
            std::size_t lutNum = lookUpTableDataSize / (pointNum * elementSize);
            grottoLutEvalKernel_ex<scalar_t> //
                (                            //
                    sharedOutE,              //
                    sharedOutT,              //
                    maskedX,                 //
                    key,                     //
                    seed,                    //
                    partyId,                 //
                    lookUpTable,             //
                    lutNum,                  //
                    lutBitWidth,             //
                    bitWidthIn,              //
                    elementNum,              //
                    cache                    //
                );                           //

            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void grottoLutEvalKernel_ex2(void       *outE,
                                    void       *outT,
                                    const void *maskedX,
                                    const void *key,
                                    const void *seed,
                                    int         partyId,
                                    const void *points,
                                    size_t      pointsNum,
                                    const void *lookUpTable,
                                    size_t      lutNum,
                                    size_t      bitWidthIn,
                                    size_t      elementNum,
                                    void       *cache)
{
    using namespace FastFss;

    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *maskedXPtr     = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr        = (const std::uint8_t *)seed;
    GroupElement       *outEPtr        = (GroupElement *)outE;
    GroupElement       *outTPtr        = (GroupElement *)outT;
    const GroupElement *lookUpTablePtr = (GroupElement *)lookUpTable;
    const GroupElement *pointsPtr      = (const GroupElement *)points;

    impl::AES128GlobalContext aesCtx;
    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement> *cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        impl::grottoLutEval_ex2(       //
            outEPtr + i,               //
            outTPtr + i * lutNum,      //
            keyObj,                    //
            maskedXPtr[maskedXOffset], //
            seedPtr + seedOffset,      //
            partyId,                   //
            pointsPtr,                 //
            pointsNum,                 //
            lookUpTablePtr,            //
            lutNum,                    //
            bitWidthIn,                //
            cacheObjPtr,               //
            &aesCtx                    //
        );
    }
}

int FastFss_cpu_grottoLutEval_ex2(void       *sharedOutE,
                                  size_t      sharedOutEDataSize,
                                  void       *sharedOutT,
                                  size_t      sharedOutTDataSize,
                                  const void *maskedX,
                                  size_t      maskedXDataSize,
                                  const void *key,
                                  size_t      keyDataSize,
                                  const void *seed,
                                  size_t      seedDataSize,
                                  int         partyId,
                                  const void *points,
                                  size_t      pointsDataSize,
                                  const void *lookUpTable,
                                  size_t      lookUpTableDataSize,
                                  size_t      bitWidthIn,
                                  size_t      bitWidthOut,
                                  size_t      elementSize,
                                  size_t      elementNum,
                                  void       *cache,
                                  size_t      cacheDataSize)
{
    int ret = FastFss_helper_checkGrottoLutEval_ex2Params(
        sharedOutEDataSize, sharedOutTDataSize, maskedXDataSize, keyDataSize,
        seedDataSize, partyId, pointsDataSize, lookUpTableDataSize, bitWidthIn,
        bitWidthOut, elementSize, elementNum, cacheDataSize,
        FastFss_cpu_grottoGetKeyDataSize, FastFss_cpu_grottoGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            std::size_t pointNum = pointsDataSize / elementSize;
            std::size_t lutNum = lookUpTableDataSize / (pointNum * elementSize);
            grottoLutEvalKernel_ex2<scalar_t> //
                (                             //
                    sharedOutE,               //
                    sharedOutT,               //
                    maskedX,                  //
                    key,                      //
                    seed,                     //
                    partyId,                  //
                    points,                   //
                    pointNum,                 //
                    lookUpTable,              //
                    lutNum,                   //
                    bitWidthIn,               //
                    elementNum,               //
                    cache                     //
                );                            //

            return FAST_FSS_SUCCESS;
        });
}

template <typename GroupElement>
static void grottoIntervalLutEvalKernel(void       *outE,
                                        void       *outT,
                                        const void *maskedX,
                                        const void *key,
                                        const void *seed,
                                        int         partyId,
                                        const void *leftBoundary,
                                        const void *rightBoundary,
                                        const void *lookUpTable,
                                        size_t      lutNum,
                                        size_t      intervalNum,
                                        size_t      bitWidthIn,
                                        size_t      elementNum,
                                        void       *cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *maskedXPtr       = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr          = (const std::uint8_t *)seed;
    GroupElement       *outEPtr          = (GroupElement *)outE;
    GroupElement       *outTPtr          = (GroupElement *)outT;
    const GroupElement *leftBoundaryPtr  = (GroupElement *)leftBoundary;
    const GroupElement *rightBoundaryPtr = (GroupElement *)rightBoundary;
    const GroupElement *lookUpTablePtr   = (GroupElement *)lookUpTable;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>   keyObj;
        impl::GrottoCache<GroupElement> cacheObj;

        impl::GrottoCache<GroupElement> *cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        impl::grottoIntervalLutEval(   //
            outEPtr + i,               //
            outTPtr + i * lutNum,      //
            keyObj,                    //
            maskedXPtr[maskedXOffset], //
            seedPtr + seedOffset,      //
            partyId,                   //
            leftBoundaryPtr,           //
            rightBoundaryPtr,          //
            lookUpTablePtr,            //
            lutNum,                    //
            intervalNum,               //
            bitWidthIn,                //
            cacheObjPtr                //
        );
    }
}

int FastFss_cpu_grottoIntervalLutEval(void       *sharedOutE,
                                      size_t      sharedOutEDataSize,
                                      void       *sharedOutT,
                                      size_t      sharedOutTDataSize,
                                      const void *maskedX,
                                      size_t      maskedXDataSize,
                                      const void *key,
                                      size_t      keyDataSize,
                                      const void *seed,
                                      size_t      seedDataSize,
                                      int         partyId,
                                      const void *leftBoundary,
                                      size_t      leftBoundaryDataSize,
                                      const void *rightBoundary,
                                      size_t      rightBoundaryDataSize,
                                      const void *lookUpTable,
                                      size_t      lookUpTableDataSize,
                                      size_t      bitWidthIn,
                                      size_t      bitWidthOut,
                                      size_t      elementSize,
                                      size_t      elementNum,
                                      void       *cache,
                                      size_t      cacheDataSize)
{
    int ret = FastFss_helper_checkGrottoIntervalLutEvalParams(
        sharedOutEDataSize, sharedOutTDataSize, maskedXDataSize, keyDataSize,
        seedDataSize, partyId, leftBoundaryDataSize, rightBoundaryDataSize,
        lookUpTableDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum,
        cacheDataSize, FastFss_cpu_grottoGetKeyDataSize,
        FastFss_cpu_grottoGetCacheDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            std::size_t intervalNum = leftBoundaryDataSize / elementSize;
            std::size_t lutNum = lookUpTableDataSize / leftBoundaryDataSize;
            grottoIntervalLutEvalKernel<scalar_t> //
                (                                 //
                    sharedOutE,                   //
                    sharedOutT,                   //
                    maskedX,                      //
                    key,                          //
                    seed,                         //
                    partyId,                      //
                    leftBoundary,                 //
                    rightBoundary,                //
                    lookUpTable,                  //
                    lutNum,                       //
                    intervalNum,                  //
                    bitWidthIn,                   //
                    elementNum,                   //
                    cache                         //
                );                                //

            return FAST_FSS_SUCCESS;
        });
}

int FastFss_cpu_grottoKeyZip(void       *zippedKey,
                             size_t      zippedKeyDataSize,
                             const void *key,
                             size_t      keyDataSize,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cpu_grottoKeyUnzip(void       *key,
                               size_t      keyDataSize,
                               const void *zippedKey,
                               size_t      zippedKeyDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum)
{
    return FAST_FSS_RUNTIME_ERROR;
}

int FastFss_cpu_grottoGetKeyDataSize(size_t *keyDataSize,
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

int FastFss_cpu_grottoGetZippedKeyDataSize(size_t *keyDataSize,
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

int FastFss_cpu_grottoGetCacheDataSize(size_t *cacheDataSize,
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