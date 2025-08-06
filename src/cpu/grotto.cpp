#include <FastFss/cpu/config.h>
#include <FastFss/cpu/grotto.h>
#include <omp.h>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../impl/grotto.h"

#define FSS_ASSERT(cond, errCode) \
    if (!(cond)) return errCode

using namespace FastFss;

enum ERR_CODE
{
    SUCCESS                      = 0,
    RUNTIME_ERROR                = -1,
    INVALID_ELEMENT_SIZE         = -2,
    INVALID_ALPHA_DATA_SIZE      = -3,
    INVALID_SEED_DATA_SIZE       = -4,
    INVALID_BITWIDTH             = -5,
    INVALID_PARTY_ID             = -6,
    INVALID_KEY_DATA_SIZE        = -7,
    INVALID_MASKED_X_DATA_SIZE   = -8,
    INVALID_CACHE_DATA_SIZE      = -9,
    INVALID_BOUNDARY_DATA_SIZE   = -10,
    INVALID_LUT_DATA_SIZE        = -11,
    INVALID_MIC_OUT_DATA_SIZE    = -12,
    INVALID_SHARED_OUT_DATA_SIZE = -13,
    INVALID_POINT_DATA_SIZE      = -14,
};

static std::size_t grottoGetKeyDataSize(size_t bitWidthIn,
                                        size_t elementSize,
                                        size_t elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetKeyDataSize<scalar_t>(bitWidthIn, elementNum);
        });
}

static std::size_t grottoGetCacheDataSize(size_t bitWidthIn,
                                          size_t elementSize,
                                          size_t elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetCacheDataSize<scalar_t>(bitWidthIn,
                                                          elementNum);
        });
}

template <typename GroupElement>
static void grottoKeyGenKernel(void*       key,
                               const void* alpha,
                               const void* seed0,
                               const void* seed1,
                               size_t      bitWidthIn,
                               size_t      elementNum)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* alphaPtr = (const GroupElement*)alpha;
    const std::uint8_t* seed0Ptr = (const std::uint8_t*)seed0;
    const std::uint8_t* seed1Ptr = (const std::uint8_t*)seed1;

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

int FastFss_cpu_grottoKeyGen(void*       key,
                             size_t      keyDataSize,
                             const void* alpha,
                             size_t      alphaDataSize,
                             const void* seed0,
                             size_t      seedDataSize0,
                             const void* seed1,
                             size_t      seedDataSize1,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    FSS_ASSERT(alphaDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_ALPHA_DATA_SIZE);
    FSS_ASSERT(seedDataSize0 == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(seedDataSize1 == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoKeyGenKernel<scalar_t>                             //
                (                                                    //
                    key, alpha, seed0, seed1, bitWidthIn, elementNum //
                );                                                   //
            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
static void grottoEvalKernel(void*       out,
                             const void* maskedX,
                             const void* key,
                             const void* seed,
                             bool        equalBound,
                             int         partyId,
                             size_t      bitWidthIn,
                             size_t      elementNum,
                             void*       cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* maskedXPtr = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr    = (const std::uint8_t*)seed;
    GroupElement*       outPtr     = (GroupElement*)out;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>   keyObj;
        impl::GrottoCache<GroupElement> cacheObj;

        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
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

int FastFss_cpu_grottoEval(void*       sharedBooleanOut,
                           const void* maskedX,
                           size_t      maskedXDataSize,
                           const void* key,
                           size_t      keyDataSize,
                           const void* seed,
                           size_t      seedDataSize,
                           bool        equalBound,
                           int         partyId,
                           size_t      bitWidthIn,
                           size_t      elementSize,
                           size_t      elementNum,
                           void*       cache,
                           size_t      cacheDataSize)
{
    FSS_ASSERT(maskedXDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
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

            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
static void grottoEqEvalKernel(void*       out,
                               const void* maskedX,
                               const void* key,
                               const void* seed,
                               int         partyId,
                               size_t      bitWidthIn,
                               size_t      elementNum,
                               void*       cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* maskedXPtr = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr    = (const std::uint8_t*)seed;
    GroupElement*       outPtr     = (GroupElement*)out;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
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

int FastFss_cpu_grottoEqEval(void*       sharedBooleanOut,
                             const void* maskedX,
                             size_t      maskedXDataSize,
                             const void* key,
                             size_t      keyDataSize,
                             const void* seed,
                             size_t      seedDataSize,
                             int         partyId,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum,
                             void*       cache,
                             size_t      cacheDataSize)
{
    FSS_ASSERT(maskedXDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
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

            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
static void grottoEqMultiEvalKernel(void*       out,
                                    const void* maskedX,
                                    const void* key,
                                    const void* seed,
                                    int         partyId,
                                    const void* point,
                                    size_t      pointNum,
                                    size_t      bitWidthIn,
                                    size_t      elementNum,
                                    void*       cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* maskedXPtr = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr    = (const std::uint8_t*)seed;
    GroupElement*       outPtr     = (GroupElement*)out;
    const GroupElement* pointPtr   = (const GroupElement*)point;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement>* cacheObjPtr = nullptr;
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

int FastFss_cpu_grottoEqMultiEval(void*       sharedBooleanOut,
                                  size_t      sharedOutDataSize,
                                  const void* maskedX,
                                  size_t      maskedXDataSize,
                                  const void* key,
                                  size_t      keyDataSize,
                                  const void* seed,
                                  size_t      seedDataSize,
                                  int         partyId,
                                  const void* point,
                                  size_t      pointDataSize,
                                  size_t      bitWidthIn,
                                  size_t      elementSize,
                                  size_t      elementNum,
                                  void*       cache,
                                  size_t      cacheDataSize)
{
    FSS_ASSERT(maskedXDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    FSS_ASSERT(pointDataSize % elementSize == 0,
               ERR_CODE::INVALID_POINT_DATA_SIZE);
    FSS_ASSERT(sharedOutDataSize == elementNum * pointDataSize,
               ERR_CODE::INVALID_SHARED_OUT_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
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
            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
static void grottoMICEvalKernel(void*       out,
                                const void* maskedX,
                                const void* key,
                                const void* seed,
                                int         partyId,
                                const void* leftBoundary,
                                const void* rightBoundary,
                                size_t      intervalNum,
                                size_t      bitWidthIn,
                                size_t      elementNum,
                                void*       cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* maskedXPtr       = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr          = (const std::uint8_t*)seed;
    GroupElement*       outPtr           = (GroupElement*)out;
    const GroupElement* leftBoundaryPtr  = (GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (GroupElement*)rightBoundary;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
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

int FastFss_cpu_grottoMICEval(void*       sharedBooleanOut,
                              size_t      sharedBooleanOutDataSize,
                              const void* maskedX,
                              size_t      maskedXDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              const void* seed,
                              size_t      seedDataSize,
                              int         partyId,
                              const void* leftBoundary,
                              size_t      leftBoundaryDataSize,
                              const void* rightBoundary,
                              size_t      rightBoundaryDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cache,
                              size_t      cacheDataSize)
{
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);
    FSS_ASSERT(leftBoundaryDataSize == rightBoundaryDataSize &&
                   leftBoundaryDataSize % elementSize == 0,
               ERR_CODE::INVALID_BOUNDARY_DATA_SIZE);
    std::size_t intervalNum = leftBoundaryDataSize / elementSize;

    FSS_ASSERT(
        sharedBooleanOutDataSize == intervalNum * elementNum * elementSize,
        ERR_CODE::INVALID_MIC_OUT_DATA_SIZE);

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
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

            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
static void grottoLutEvalKernel(void*       outE,
                                void*       outT,
                                const void* maskedX,
                                const void* key,
                                const void* seed,
                                int         partyId,
                                const void* lookUpTable,
                                size_t      lutNum,
                                size_t      bitWidthIn,
                                size_t      elementNum,
                                void*       cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* maskedXPtr     = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr        = (const std::uint8_t*)seed;
    GroupElement*       outEPtr        = (GroupElement*)outE;
    GroupElement*       outTPtr        = (GroupElement*)outT;
    const GroupElement* lookUpTablePtr = (GroupElement*)lookUpTable;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>   keyObj;
        impl::GrottoCache<GroupElement> cacheObj;

        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        impl::grottoLutEval(           //
            outEPtr + i,               //
            outTPtr + i * lutNum,      //
            keyObj,                    //
            maskedXPtr[maskedXOffset], //
            seedPtr + seedOffset,      //
            partyId,                   //
            lookUpTablePtr,            //
            lutNum,                    //
            bitWidthIn,                //
            cacheObjPtr                //
        );
    }
}

int FastFss_cpu_grottoLutEval(void*       sharedOutE,
                              void*       sharedOutT,
                              const void* maskedX,
                              size_t      maskedXDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              const void* seed,
                              size_t      seedDataSize,
                              int         partyId,
                              const void* lookUpTable,
                              size_t      lookUpTableDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      elementSize,
                              size_t      elementNum,
                              void*       cache,
                              size_t      cacheDataSize)
{
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);

    FSS_ASSERT(lookUpTableDataSize % (elementSize * (1ULL << bitWidthIn)) == 0,
               INVALID_LUT_DATA_SIZE);
    std::size_t lutNum =
        lookUpTableDataSize / (elementSize * (1ULL << bitWidthIn));

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoLutEvalKernel<scalar_t> //
                (                         //
                    sharedOutE,           //
                    sharedOutT,           //
                    maskedX,              //
                    key,                  //
                    seed,                 //
                    partyId,              //
                    lookUpTable,          //
                    lutNum,               //
                    bitWidthIn,           //
                    elementNum,           //
                    cache                 //
                );                        //

            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
static void grottoLutEvalKernel_ex(void*       outE,
                                   void*       outT,
                                   const void* maskedX,
                                   const void* key,
                                   const void* seed,
                                   int         partyId,
                                   const void* lookUpTable,
                                   size_t      lutNum,
                                   size_t      lutBitWidth,
                                   size_t      bitWidthIn,
                                   size_t      elementNum,
                                   void*       cache0,
                                   void*       cache1)
{
    using namespace FastFss;

    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* maskedXPtr     = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr        = (const std::uint8_t*)seed;
    GroupElement*       outEPtr        = (GroupElement*)outE;
    GroupElement*       outTPtr        = (GroupElement*)outT;
    const GroupElement* lookUpTablePtr = (GroupElement*)lookUpTable;

    impl::AES128GlobalContext aesCtx;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache0 != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache0, bitWidthIn, i,
                                    elementNum);
            cacheObjPtr = &cacheObj;
        }
        if (cache0 == nullptr && cache1 != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache1, bitWidthIn, i,
                                    elementNum);
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
            nullptr,                          //
            &aesCtx                           //
        );
    }
}

int FastFss_cpu_grottoLutEval_ex(void*       sharedOutE,
                                 void*       sharedOutT,
                                 const void* maskedX,
                                 size_t      maskedXDataSize,
                                 const void* key,
                                 size_t      keyDataSize,
                                 const void* seed,
                                 size_t      seedDataSize,
                                 int         partyId,
                                 const void* lookUpTable,
                                 size_t      lookUpTableDataSize,
                                 size_t      lutBitWidth,
                                 size_t      bitWidthIn,
                                 size_t      bitWidthOut,
                                 size_t      elementSize,
                                 size_t      elementNum,
                                 void*       cache0,
                                 void*       cache1,
                                 size_t      cacheDataSize)
{
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache0 != nullptr || cache1 != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);

    FSS_ASSERT(lutBitWidth <= bitWidthIn, INVALID_LUT_DATA_SIZE);
    FSS_ASSERT(lookUpTableDataSize % (elementSize * (1ULL << lutBitWidth)) == 0,
               INVALID_LUT_DATA_SIZE);
    std::size_t lutNum =
        lookUpTableDataSize / (elementSize * (1ULL << lutBitWidth));

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
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
                    cache0,                  //
                    cache1                   //
                );                           //

            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
static void grottoLutEvalKernel_ex2(void*       outE,
                                    void*       outT,
                                    const void* maskedX,
                                    const void* key,
                                    const void* seed,
                                    int         partyId,
                                    const void* points,
                                    size_t      pointsNum,
                                    const void* lookUpTable,
                                    size_t      lutNum,
                                    size_t      bitWidthIn,
                                    size_t      elementNum,
                                    void*       cache)
{
    using namespace FastFss;

    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* maskedXPtr     = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr        = (const std::uint8_t*)seed;
    GroupElement*       outEPtr        = (GroupElement*)outE;
    GroupElement*       outTPtr        = (GroupElement*)outT;
    const GroupElement* lookUpTablePtr = (GroupElement*)lookUpTable;
    const GroupElement* pointsPtr      = (const GroupElement*)points;

    impl::AES128GlobalContext aesCtx;
    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
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

int FastFss_cpu_grottoLutEval_ex2(void*       sharedOutE,
                                  void*       sharedOutT,
                                  const void* maskedX,
                                  size_t      maskedXDataSize,
                                  const void* key,
                                  size_t      keyDataSize,
                                  const void* seed,
                                  size_t      seedDataSize,
                                  int         partyId,
                                  const void* points,
                                  size_t      pointsDataSize,
                                  const void* lookUpTable,
                                  size_t      lookUpTableDataSize,
                                  size_t      bitWidthIn,
                                  size_t      bitWidthOut,
                                  size_t      elementSize,
                                  size_t      elementNum,
                                  void*       cache,
                                  size_t      cacheDataSize)
{
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);

    FSS_ASSERT(lookUpTableDataSize % pointsDataSize == 0,
               INVALID_LUT_DATA_SIZE);
    std::size_t lutNum = lookUpTableDataSize / pointsDataSize;
    FSS_ASSERT(pointsDataSize % elementSize == 0, INVALID_POINT_DATA_SIZE);
    std::size_t pointsNum = pointsDataSize / elementSize;
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoLutEvalKernel_ex2<scalar_t> //
                (                             //
                    sharedOutE,               //
                    sharedOutT,               //
                    maskedX,                  //
                    key,                      //
                    seed,                     //
                    partyId,                  //
                    points,                   //
                    pointsNum,                //
                    lookUpTable,              //
                    lutNum,                   //
                    bitWidthIn,               //
                    elementNum,               //
                    cache                     //
                );                            //

            return ERR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
static void grottoIntervalLutEvalKernel(void*       outE,
                                        void*       outT,
                                        const void* maskedX,
                                        const void* key,
                                        const void* seed,
                                        int         partyId,
                                        const void* leftBoundary,
                                        const void* rightBoundary,
                                        const void* lookUpTable,
                                        size_t      lutNum,
                                        size_t      intervalNum,
                                        size_t      bitWidthIn,
                                        size_t      elementNum,
                                        void*       cache)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* maskedXPtr       = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr          = (const std::uint8_t*)seed;
    GroupElement*       outEPtr          = (GroupElement*)outE;
    GroupElement*       outTPtr          = (GroupElement*)outT;
    const GroupElement* leftBoundaryPtr  = (GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (GroupElement*)rightBoundary;
    const GroupElement* lookUpTablePtr   = (GroupElement*)lookUpTable;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        impl::GrottoKey<GroupElement>   keyObj;
        impl::GrottoCache<GroupElement> cacheObj;

        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
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

int FastFss_cpu_grottoIntervalLutEval(void*       sharedOutE,
                                      void*       sharedOutT,
                                      const void* maskedX,
                                      size_t      maskedXDataSize,
                                      const void* key,
                                      size_t      keyDataSize,
                                      const void* seed,
                                      size_t      seedDataSize,
                                      int         partyId,
                                      const void* leftBoundary,
                                      size_t      leftBoundaryDataSize,
                                      const void* rightBoundary,
                                      size_t      rightBoundaryDataSize,
                                      const void* lookUpTable,
                                      size_t      lookUpTableDataSize,
                                      size_t      bitWidthIn,
                                      size_t      bitWidthOut,
                                      size_t      elementSize,
                                      size_t      elementNum,
                                      void*       cache,
                                      size_t      cacheDataSize)
{
    FSS_ASSERT(maskedXDataSize == elementNum * elementSize,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(seedDataSize == elementNum * 16,
               ERR_CODE::INVALID_SEED_DATA_SIZE);
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    FSS_ASSERT(keyDataSize ==
                   grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum),
               ERR_CODE::INVALID_KEY_DATA_SIZE);
    if (cache != nullptr)
    {
        std::size_t needCacheDataSize =
            grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum);
        FSS_ASSERT(cacheDataSize == needCacheDataSize,
                   ERR_CODE::INVALID_CACHE_DATA_SIZE);
    }
    FSS_ASSERT(partyId == 0 || partyId == 1, ERR_CODE::INVALID_PARTY_ID);
    FSS_ASSERT(leftBoundaryDataSize == rightBoundaryDataSize &&
                   leftBoundaryDataSize % elementSize == 0,
               ERR_CODE::INVALID_BOUNDARY_DATA_SIZE);
    std::size_t intervalNum = leftBoundaryDataSize / elementSize;

    FSS_ASSERT(lookUpTableDataSize % (elementSize * intervalNum) == 0,
               INVALID_LUT_DATA_SIZE);
    std::size_t lutNum = lookUpTableDataSize / leftBoundaryDataSize;

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
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

            return ERR_CODE::SUCCESS;
        });
}

int FastFss_cpu_grottoKeyZip(void*       zippedKey,
                             size_t      zippedKeyDataSize,
                             const void* key,
                             size_t      keyDataSize,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    return -1;
}

int FastFss_cpu_grottoKeyUnzip(void*       key,
                               size_t      keyDataSize,
                               const void* zippedKey,
                               size_t      zippedKeyDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum)
{
    return -1;
}

int FastFss_cpu_grottoGetKeyDataSize(size_t* keyDataSize,
                                     size_t  bitWidthIn,
                                     size_t  elementSize,
                                     size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetKeyDataSize<scalar_t>(bitWidthIn, elementNum);
        });
    return ERR_CODE::SUCCESS;
}

int FastFss_cpu_grottoGetZippedKeyDataSize(size_t* keyDataSize,
                                           size_t  bitWidthIn,
                                           size_t  elementSize,
                                           size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetZippedKeyDataSize<scalar_t>(bitWidthIn,
                                                              elementNum);
        });
    return ERR_CODE::SUCCESS;
}

int FastFss_cpu_grottoGetCacheDataSize(size_t* cacheDataSize,
                                       size_t  bitWidthIn,
                                       size_t  elementSize,
                                       size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn <= elementSize * 8 && bitWidthIn >= 6,
               ERR_CODE::INVALID_BITWIDTH);
    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetCacheDataSize<scalar_t>(bitWidthIn,
                                                          elementNum);
        });
    return ERR_CODE::SUCCESS;
}