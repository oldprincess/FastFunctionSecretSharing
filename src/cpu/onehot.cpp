#include "../impl/onehot.h"

#include <FastFss/cpu/config.h>
#include <FastFss/cpu/onehot.h>
#include <omp.h>

using namespace FastFss;

#define FSS_ASSERT(cond, errCode) \
    if (!(cond)) return errCode

enum ERR_CODE
{
    SUCCESS                        = 0,
    RUNTIME_ERROR                  = -1,
    INVALID_BIT_WIDTH_IN           = -2,
    INVALID_ELEMENT_SIZE           = -3,
    INVALID_KEY_DATA_SIZE          = -4,
    INVALID_ALPHA_DATA_SIZE        = -5,
    INVALID_LOOKUP_TABLE_DATA_SIZE = -6,
    INVALID_MASKED_X_DATA_SIZE     = -7,
};

template <typename GroupElement>
static void onehotKeyGenKernel(void*       key,
                               const void* alpha,
                               std::size_t bitWidthIn,
                               std::size_t elementNum)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* alphaPtr = (const GroupElement*)alpha;
    std::uint8_t*       keyPtr   = (std::uint8_t*)key;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        std::size_t keyOffset = i * (1ULL << bitWidthIn) / 8;
        impl::onehotKeyGen(keyPtr + keyOffset, alphaPtr[i], bitWidthIn);
    }
}

template <typename GroupElement>
static void onehotLutEvalKernel(void*       sharedOutE,
                                void*       sharedOutT,
                                const void* maskedX,
                                const void* key,
                                const void* lut,
                                int         partyId,
                                std::size_t bitWidthIn,
                                std::size_t elementNum)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement* maskedXPtr    = (const GroupElement*)maskedX;
    const std::uint8_t* keyPtr        = (const std::uint8_t*)key;
    const GroupElement* lutPtr        = (const GroupElement*)lut;
    GroupElement*       sharedOutEPtr = (GroupElement*)sharedOutE;
    GroupElement*       sharedOutTPtr = (GroupElement*)sharedOutT;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        std::size_t keyOffset = i * (1ULL << bitWidthIn) / 8; //

        impl::onehotLutEval<GroupElement>( //
            sharedOutEPtr + i,             //
            sharedOutTPtr + i,             //
            maskedXPtr[i],                 //
            keyPtr + keyOffset,            //
            lutPtr,                        //
            partyId,                       //
            bitWidthIn);
    }
}

int FastFss_cpu_onehotKeyGen(void*       key,
                             size_t      keyDataSize,
                             const void* alpha,
                             size_t      alphaDataSize,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    using namespace impl;
    FSS_ASSERT(bitWidthIn >= 3, ERR_CODE::INVALID_BIT_WIDTH_IN);

    std::size_t needKeyDataSize = onehotGetKeyDataSize( //
        bitWidthIn, elementNum                          //
    );                                                  //
    FSS_ASSERT(needKeyDataSize == keyDataSize, ERR_CODE::INVALID_KEY_DATA_SIZE);

    FSS_ASSERT(alphaDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_ALPHA_DATA_SIZE);

    auto ret = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize,                                //
        { return ERR_CODE::INVALID_ELEMENT_SIZE; }, //
        [&] {
            onehotKeyGenKernel<scalar_t>(key, alpha, bitWidthIn, elementNum);
            return ERR_CODE::SUCCESS;
        });
    return ret;
}

int FastFss_cpu_onehotLutEval(void*       sharedOutE,
                              void*       sharedOutT,
                              const void* maskedX,
                              size_t      maskedXDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              int         partyId,
                              const void* lookUpTable,
                              size_t      lookUpTableDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum)
{
    using namespace impl;

    FSS_ASSERT(bitWidthIn >= 3, ERR_CODE::INVALID_BIT_WIDTH_IN);

    std::size_t needKeyDataSize = onehotGetKeyDataSize( //
        bitWidthIn, elementNum                          //
    );                                                  //
    FSS_ASSERT(needKeyDataSize == keyDataSize, ERR_CODE::INVALID_KEY_DATA_SIZE);

    FSS_ASSERT(maskedXDataSize == elementSize * elementNum,
               ERR_CODE::INVALID_MASKED_X_DATA_SIZE);
    FSS_ASSERT(lookUpTableDataSize == elementSize * (1ULL << bitWidthIn),
               ERR_CODE::INVALID_LOOKUP_TABLE_DATA_SIZE);

    auto ret = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize,                                //
        { return ERR_CODE::INVALID_ELEMENT_SIZE; }, //
        [&] {
            onehotLutEvalKernel<scalar_t>( //
                sharedOutE,                //
                sharedOutT,                //
                maskedX,                   //
                key,                       //
                lookUpTable,               //
                partyId,                   //
                bitWidthIn,                //
                elementNum);               //

            return ERR_CODE::SUCCESS;
        });
    return ret;
}

int FastFss_cpu_onehotGetKeyDataSize(size_t* keyDataSize,
                                     size_t  bitWidthIn,
                                     size_t  elementNum)
{
    FSS_ASSERT(bitWidthIn >= 3, ERR_CODE::INVALID_BIT_WIDTH_IN);
    *keyDataSize = impl::onehotGetKeyDataSize(bitWidthIn, elementNum);
    return (int)ERR_CODE::SUCCESS;
}