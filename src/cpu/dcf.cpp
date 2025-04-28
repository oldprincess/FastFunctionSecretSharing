#include <FastFss/cpu/dcf.h>

#include <cassert>
#include <memory>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../impl/dcf.h"

#define FSS_ASSERT(cond, errCode) \
    if (!(cond)) return errCode

using namespace FastFss;

enum ERROR_CODE
{
    SUCCESS                            = 0,
    RUNTIME_ERROR                      = -1,
    INVALID_KEY_DATA_SIZE_ERROR        = -2,
    INVALID_ALPHA_DATA_SIZE_ERROR      = -3,
    INVALID_BETA_DATA_SIZE_ERROR       = -4,
    INVALID_SEED_DATA_SIZE_ERROR       = -5,
    INVALID_BOUNDARY_DATA_SIZE_ERROR   = -6,
    INVALID_Z_DATA_SIZE_ERROR          = -7,
    INVALID_SHARED_OUT_DATA_SIZE_ERROR = -8,
    INVLIAD_MASKED_X_DATA_SIZE_ERROR   = -9,
    INVALID_BITWIDTH_ERROR             = -10,
    INVALID_ELEMENT_SIZE_ERROR         = -11,
    INVALID_PARTY_ID_ERROR             = -12,
};

template <typename GroupElement>
static void dcfKeyGenKernel(void*       key,
                            const void* alpha,
                            const void* beta,
                            const void* seed0,
                            const void* seed1,
                            std::size_t bitWidthIn,
                            std::size_t bitWidthOut,
                            std::size_t elementNum)
{
    std::size_t idx    = 0;
    std::size_t stride = 1;

    const GroupElement* alphaPtr = (const GroupElement*)alpha;
    const GroupElement* betaPtr  = (const GroupElement*)beta;
    const std::uint8_t* seed0Ptr = (const std::uint8_t*)seed0;
    const std::uint8_t* seed1Ptr = (const std::uint8_t*)seed1;

    impl::DcfKey<GroupElement> keyObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, i, elementNum);
        impl::dcfKeyGen(keyObj,                                   //
                        alphaPtr[i],                              //
                        (betaPtr) ? betaPtr[i] : (GroupElement)1, //
                        seed0Ptr + 16 * i,                        //
                        seed1Ptr + 16 * i,                        //
                        bitWidthIn,                               //
                        bitWidthOut                               //
        );
    }
}

int FastFss_cpu_dcfKeyGen(void*       key,
                          size_t      keyDataSize,
                          const void* alpha,
                          size_t      alphaDataSize,
                          const void* beta,
                          size_t      betaDataSize,
                          const void* seed0,
                          size_t      seedDataSize0,
                          const void* seed1,
                          size_t      seedDataSize1,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      elementSize,
                          size_t      elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERROR_CODE::INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dcfKeyGenKernel<scalar_t>(
                key, alpha, (betaDataSize) ? beta : nullptr, seed0, seed1,
                bitWidthIn, bitWidthOut, elementNum);

            return ERROR_CODE::SUCCESS;
        });
}

template <typename GroupElement>
static void dcfEvalKernel(void*       sharedOut,
                          const void* maskedX,
                          const void* key,
                          const void* seed,
                          int         partyId,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      elementNum,
                          void*       cache)
{
    std::size_t idx    = 0;
    std::size_t stride = 1;

    GroupElement*       sharedOutPtr = (GroupElement*)sharedOut;
    const GroupElement* maskedXPtr   = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr      = (const std::uint8_t*)seed;

    impl::DcfKey<GroupElement> keyObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::dcfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, i, elementNum);
        sharedOutPtr[i] = impl::dcfEval(keyObj,           //
                                        maskedXPtr[i],    //
                                        seedPtr + 16 * i, //
                                        partyId,          //
                                        bitWidthIn,       //
                                        bitWidthOut);
    }
}

int FastFss_cpu_dcfEval(void*       sharedOut,
                        const void* maskedX,
                        size_t      maskedXDataSize,
                        const void* key,
                        size_t      keyDataSize,
                        const void* seed,
                        size_t      seedDataSize,
                        int         partyId,
                        size_t      bitWidthIn,
                        size_t      bitWidthOut,
                        size_t      elementSize,
                        size_t      elementNum,
                        void*       cache,
                        size_t      cacheDataSize)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return ERROR_CODE::INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            dcfEvalKernel<scalar_t>(sharedOut, maskedX, key, seed, partyId,
                                    bitWidthIn, bitWidthOut, elementNum, cache);
            return ERROR_CODE::SUCCESS;
        });
}

int FastFss_cpu_dcfKeyZip(void*       zippedKey,
                          size_t      zippedKeyDataSize,
                          const void* key,
                          size_t      keyDataSize,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      elementSize,
                          size_t      elementNum)
{
    return ERROR_CODE::RUNTIME_ERROR;
}

int FastFss_cpu_dcfKeyUnzip(void*       key,
                            size_t      keyDataSize,
                            const void* zippedKey,
                            size_t      zippedKeyDataSize,
                            size_t      bitWidthIn,
                            size_t      bitWidthOut,
                            size_t      elementSize,
                            size_t      elementNum)
{
    return ERROR_CODE::RUNTIME_ERROR;
}

int FastFss_cpu_dcfGetKeyDataSize(size_t* keyDataSize,
                                  size_t  bitWidthIn,
                                  size_t  bitWidthOut,
                                  size_t  elementSize,
                                  size_t  elementNum)
{
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dcfGetKeyDataSize<scalar_t>(bitWidthIn, bitWidthOut,
                                                     elementNum);
        });
    return ERROR_CODE::SUCCESS;
}

int FastFss_cpu_dcfGetZippedKeyDataSize(size_t* keyDataSize,
                                        size_t  bitWidthIn,
                                        size_t  bitWidthOut,
                                        size_t  elementSize,
                                        size_t  elementNum)
{
    *keyDataSize = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (std::size_t)0; },
        [&] {
            return impl::dcfGetZippedKeyDataSize<scalar_t>(
                bitWidthIn, bitWidthOut, elementNum);
        });
    return ERROR_CODE::SUCCESS;
}

int FastFss_cpu_dcfGetCacheDataSize(size_t* cacheDataSize,
                                    size_t  bitWidthIn,
                                    size_t  bitWidthOut,
                                    size_t  elementSize,
                                    size_t  elementNum)
{
    return ERROR_CODE::RUNTIME_ERROR;
}