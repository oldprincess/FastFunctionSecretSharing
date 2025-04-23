#include <FastFss/cpu/grotto.h>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../impl/grotto.h"

using namespace FastFss;

enum ERR_CODE
{
    SUCCESS                    = 0,
    RUNTIME_ERROR              = -1,
    INVALID_ELEMENT_SIZE       = -2,
    INVALID_ALPHA_DATA_SIZE    = -3,
    INVALID_SEED_DATA_SIZE     = -4,
    INVALID_BITWIDTH           = -5,
    INVALID_PARTY_ID           = -6,
    INVALID_KEY_DATA_SIZE      = -7,
    INVLIAD_MASKED_X_DATA_SIZE = -8,
    INVALID_CACHE_DATA_SIZE    = -9,
    INVALID_BOUNDARY_DATA_SIZE = -10,
    INVALID_LUT_DATA_SIZE      = -11,
    INVALID_MIC_OUT_DATA_SIZE  = -12,
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

#define CHECK_ALPHA_DATA_SIZE(alphaDataSize, elementSize, elementNum) \
    if (alphaDataSize != elementSize * elementNum)                    \
    {                                                                 \
        return ERR_CODE::INVALID_ALPHA_DATA_SIZE;                     \
    }

#define CHECK_MASKED_X_DATA_SIZE(maskedXDataSize, elementSize, elementNum) \
    if (maskedXDataSize != elementSize * elementNum)                       \
    {                                                                      \
        return ERR_CODE::INVLIAD_MASKED_X_DATA_SIZE;                       \
    }

#define CHECK_SEED_DATA_SIZE(seedDataSize, elementNum) \
    if (seedDataSize != 16 * elementNum)               \
    {                                                  \
        return ERR_CODE::INVALID_SEED_DATA_SIZE;       \
    }

#define CHECK_BIT_WIDTH_IN(bitWidthIn, elementSize)         \
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6)) \
    {                                                       \
        return ERR_CODE::INVALID_BITWIDTH;                  \
    }

#define CHECK_PARTY_ID(partyId)            \
    if (!(partyId == 0 || partyId == 1))   \
    {                                      \
        return ERR_CODE::INVALID_PARTY_ID; \
    }

#define CHECK_KEY_DATA_SIZE(keyDataSize, bitWidthIn, elementSize, elementNum) \
    if (!(keyDataSize ==                                                      \
          grottoGetKeyDataSize(bitWidthIn, elementSize, elementNum)))         \
    {                                                                         \
        return ERR_CODE::INVALID_KEY_DATA_SIZE;                               \
    }

#define CHECK_CACHE_DATA_SIZE(cachePtr, cacheDataSize, bitWidthIn,            \
                              elementSize, elementNum)                        \
    if (!(cachePtr == nullptr ||                                              \
          (cachePtr != nullptr &&                                             \
           cacheDataSize ==                                                   \
               grottoGetCacheDataSize(bitWidthIn, elementSize, elementNum)))) \
    {                                                                         \
        return ERR_CODE::INVALID_CACHE_DATA_SIZE;                             \
    }

#define CHECK_BOUNDARY_DATA_SIZE(leftBoundaryDataSize, rightBoundaryDataSize, \
                                 elementSize)                                 \
    if (!(leftBoundaryDataSize == rightBoundaryDataSize &&                    \
          leftBoundaryDataSize % elementSize == 0))                           \
    {                                                                         \
        return ERR_CODE::INVALID_BOUNDARY_DATA_SIZE;                          \
    }

#define CHECK_MIC_OUT_DATA_SIZE(outDataSize, intervalNum, elementSize, \
                                elementNum)                            \
    if (!(outDataSize == intervalNum * elementSize * elementNum))      \
    {                                                                  \
        return ERR_CODE::INVALID_MIC_OUT_DATA_SIZE;                    \
    }

#define CHECK_LUT_DATA_SIZE(lutDataSize, intervalNum, elementSize) \
    if (!(lutDataSize == intervalNum * elementSize))               \
    {                                                              \
        return ERR_CODE::INVALID_LUT_DATA_SIZE;                    \
    }

#define CAL_INTERVAL_NUM(boundaryDataSize, elementSize) \
    ((boundaryDataSize) / (elementSize))

#define CAST_CUDA_STREAM_PTR(cpuStreamPtr) \
    ((cpuStreamPtr) == nullptr) ? 0 : *(cpuStream_t*)(cpuStreamPtr)

template <typename GroupElement>
static void grottoKeyGenKernel(void*       key,
                               const void* alpha,
                               const void* seed0,
                               const void* seed1,
                               size_t      bitWidthIn,
                               size_t      elementNum)
{
    std::size_t idx    = 0;
    std::size_t stride = 1;

    const GroupElement* alphaPtr = (const GroupElement*)alpha;
    const std::uint8_t* seed0Ptr = (const std::uint8_t*)seed0;
    const std::uint8_t* seed1Ptr = (const std::uint8_t*)seed1;

    impl::GrottoKey<GroupElement> keyObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        std::size_t alphaOffset = i;
        std::size_t seed0Offset = 16 * i;
        std::size_t seed1Offset = 16 * i;
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
    CHECK_ALPHA_DATA_SIZE(alphaDataSize, elementSize, elementNum);
    CHECK_SEED_DATA_SIZE(seedDataSize0, elementNum);
    CHECK_SEED_DATA_SIZE(seedDataSize1, elementNum);
    CHECK_BIT_WIDTH_IN(bitWidthIn, elementSize);
    CHECK_KEY_DATA_SIZE(keyDataSize, bitWidthIn, elementSize, elementNum);

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
    std::size_t idx    = 0;
    std::size_t stride = 1;

    const GroupElement* maskedXPtr = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr    = (const std::uint8_t*)seed;
    GroupElement*       outPtr     = (GroupElement*)out;

    impl::GrottoKey<GroupElement>   keyObj;
    impl::GrottoCache<GroupElement> cacheObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
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
    CHECK_MASKED_X_DATA_SIZE(maskedXDataSize, elementSize, elementNum);
    CHECK_SEED_DATA_SIZE(seedDataSize, elementNum);
    CHECK_BIT_WIDTH_IN(bitWidthIn, elementSize);
    CHECK_KEY_DATA_SIZE(keyDataSize, bitWidthIn, elementSize, elementNum);
    CHECK_CACHE_DATA_SIZE(cache, cacheDataSize, bitWidthIn, elementSize,
                          elementNum);
    CHECK_PARTY_ID(partyId);

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
static void grottoEvalEqKernel(void*       out,
                               const void* maskedX,
                               const void* key,
                               const void* seed,
                               int         partyId,
                               size_t      bitWidthIn,
                               size_t      elementNum,
                               void*       cache)
{
    std::size_t idx    = 0;
    std::size_t stride = 1;

    const GroupElement* maskedXPtr = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr    = (const std::uint8_t*)seed;
    GroupElement*       outPtr     = (GroupElement*)out;

    impl::GrottoKey<GroupElement>   keyObj;
    impl::GrottoCache<GroupElement> cacheObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        outPtr[i] = impl::grottoEvalEq(keyObj,                    //
                                       maskedXPtr[maskedXOffset], //
                                       seedPtr + seedOffset,      //
                                       partyId,                   //
                                       bitWidthIn,                //
                                       cacheObjPtr                //
        );
    }
}

int FastFss_cpu_grottoEvalEq(void*       sharedBooleanOut,
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
    CHECK_MASKED_X_DATA_SIZE(maskedXDataSize, elementSize, elementNum);
    CHECK_SEED_DATA_SIZE(seedDataSize, elementNum);
    CHECK_BIT_WIDTH_IN(bitWidthIn, elementSize);
    CHECK_KEY_DATA_SIZE(keyDataSize, bitWidthIn, elementSize, elementNum);
    CHECK_CACHE_DATA_SIZE(cache, cacheDataSize, bitWidthIn, elementSize,
                          elementNum);
    CHECK_PARTY_ID(partyId);

    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERR_CODE::INVALID_ELEMENT_SIZE; },
        [&] {
            grottoEvalEqKernel<scalar_t> //
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
    std::size_t idx    = 0;
    std::size_t stride = 1;

    const GroupElement* maskedXPtr       = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr          = (const std::uint8_t*)seed;
    GroupElement*       outPtr           = (GroupElement*)out;
    const GroupElement* leftBoundaryPtr  = (GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (GroupElement*)rightBoundary;

    impl::GrottoKey<GroupElement>   keyObj;
    impl::GrottoCache<GroupElement> cacheObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
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
    CHECK_MASKED_X_DATA_SIZE(maskedXDataSize, elementSize, elementNum);
    CHECK_SEED_DATA_SIZE(seedDataSize, elementNum);
    CHECK_BIT_WIDTH_IN(bitWidthIn, elementSize);
    CHECK_KEY_DATA_SIZE(keyDataSize, bitWidthIn, elementSize, elementNum);
    CHECK_CACHE_DATA_SIZE(cache, cacheDataSize, bitWidthIn, elementSize,
                          elementNum);
    CHECK_PARTY_ID(partyId);
    CHECK_BOUNDARY_DATA_SIZE(leftBoundaryDataSize, rightBoundaryDataSize,
                             elementSize);
    CHECK_MIC_OUT_DATA_SIZE(sharedBooleanOutDataSize,
                            CAL_INTERVAL_NUM(leftBoundaryDataSize, elementSize),
                            elementSize, elementNum);

    size_t intervalNum = CAL_INTERVAL_NUM(leftBoundaryDataSize, elementSize);

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
static void grottoIntervalLutEvalKernel(void*       outE,
                                        void*       outT,
                                        const void* maskedX,
                                        const void* key,
                                        const void* seed,
                                        int         partyId,
                                        const void* leftBoundary,
                                        const void* rightBoundary,
                                        const void* lookUpTable,
                                        size_t      intervalNum,
                                        size_t      bitWidthIn,
                                        size_t      elementNum,
                                        void*       cache)
{
    std::size_t idx    = 0;
    std::size_t stride = 1;

    const GroupElement* maskedXPtr       = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr          = (const std::uint8_t*)seed;
    GroupElement*       outEPtr          = (GroupElement*)outE;
    GroupElement*       outTPtr          = (GroupElement*)outT;
    const GroupElement* leftBoundaryPtr  = (GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (GroupElement*)rightBoundary;
    const GroupElement* lookUpTablePtr   = (GroupElement*)lookUpTable;

    impl::GrottoKey<GroupElement>   keyObj;
    impl::GrottoCache<GroupElement> cacheObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
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
            outTPtr + i,               //
            keyObj,                    //
            maskedXPtr[maskedXOffset], //
            seedPtr + seedOffset,      //
            partyId,                   //
            leftBoundaryPtr,           //
            rightBoundaryPtr,          //
            lookUpTablePtr,            //
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
    CHECK_MASKED_X_DATA_SIZE(maskedXDataSize, elementSize, elementNum);
    CHECK_SEED_DATA_SIZE(seedDataSize, elementNum);
    CHECK_BIT_WIDTH_IN(bitWidthIn, elementSize);
    CHECK_KEY_DATA_SIZE(keyDataSize, bitWidthIn, elementSize, elementNum);
    CHECK_CACHE_DATA_SIZE(cache, cacheDataSize, bitWidthIn, elementSize,
                          elementNum);
    CHECK_PARTY_ID(partyId);
    CHECK_BOUNDARY_DATA_SIZE(leftBoundaryDataSize, rightBoundaryDataSize,
                             elementSize);

    CHECK_LUT_DATA_SIZE(lookUpTableDataSize,
                        CAL_INTERVAL_NUM(leftBoundaryDataSize, elementSize),
                        elementSize);

    std::size_t intervalNum =
        CAL_INTERVAL_NUM(leftBoundaryDataSize, elementSize);

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
    CHECK_BIT_WIDTH_IN(bitWidthIn, elementSize);
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
    CHECK_BIT_WIDTH_IN(bitWidthIn, elementSize);
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
    *cacheDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (size_t)0; },
        [&] {
            return impl::grottoGetCacheDataSize<scalar_t>(bitWidthIn,
                                                          elementNum);
        });
    return ERR_CODE::SUCCESS;
}