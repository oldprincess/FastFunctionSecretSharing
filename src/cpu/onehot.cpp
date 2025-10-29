#include "../impl/onehot.h"

#include <FastFss/cpu/config.h>
#include <FastFss/cpu/onehot.h>
#include <omp.h>

#include "../helper/onehot_helper.h"

using namespace FastFss;

template <typename GroupElement>
static void onehotKeyGenKernel(void       *key,
                               const void *alpha,
                               std::size_t bitWidthIn,
                               std::size_t elementNum)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *alphaPtr = (const GroupElement *)alpha;
    std::uint8_t       *keyPtr   = (std::uint8_t *)key;

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = idx; i < (std::int64_t)elementNum; i += stride)
    {
        std::size_t keyOffset = i * (1ULL << bitWidthIn) / 8;
        impl::onehotKeyGen(keyPtr + keyOffset, alphaPtr[i], bitWidthIn);
    }
}

template <typename GroupElement>
static void onehotLutEvalKernel(void       *sharedOutE,
                                void       *sharedOutT,
                                const void *maskedX,
                                const void *key,
                                const void *lut,
                                int         partyId,
                                std::size_t bitWidthIn,
                                std::size_t elementNum)
{
    std::int64_t idx    = 0;
    std::int64_t stride = 1;

    const GroupElement *maskedXPtr    = (const GroupElement *)maskedX;
    const std::uint8_t *keyPtr        = (const std::uint8_t *)key;
    const GroupElement *lutPtr        = (const GroupElement *)lut;
    GroupElement       *sharedOutEPtr = (GroupElement *)sharedOutE;
    GroupElement       *sharedOutTPtr = (GroupElement *)sharedOutT;

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

int FastFss_cpu_onehotKeyGen(void       *key,
                             size_t      keyDataSize,
                             const void *alpha,
                             size_t      alphaDataSize,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    int ret = FastFss_helper_checkOnehotKeyGenParams(
        keyDataSize, alphaDataSize, bitWidthIn, elementSize, elementNum,
        FastFss_cpu_onehotGetKeyDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    ret = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize,                                     //
        { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; }, //
        [&] {
            onehotKeyGenKernel<scalar_t>(          //
                key, alpha, bitWidthIn, elementNum //
            );                                     //
            return FAST_FSS_SUCCESS;
        });
    return ret;
}

int FastFss_cpu_onehotLutEval(void       *sharedOutE,
                              size_t      sharedOutEDataSize,
                              void       *sharedOutT,
                              size_t      sharedOutTDataSize,
                              const void *maskedX,
                              size_t      maskedXDataSize,
                              const void *key,
                              size_t      keyDataSize,
                              int         partyId,
                              const void *lookUpTable,
                              size_t      lookUpTableDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum)
{
    int ret = FastFss_helper_checkOnehotLutEvalParams(
        sharedOutEDataSize, sharedOutTDataSize, maskedXDataSize, keyDataSize,
        partyId, lookUpTableDataSize, bitWidthIn, elementSize, elementNum,
        FastFss_cpu_onehotGetKeyDataSize);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    ret = FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize,                                     //
        { return FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; }, //
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

            return FAST_FSS_SUCCESS;
        });
    return ret;
}

int FastFss_cpu_onehotGetKeyDataSize(size_t *keyDataSize,
                                     size_t  bitWidthIn,
                                     size_t  elementNum)
{
    if (!(3 <= bitWidthIn))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    *keyDataSize = impl::onehotGetKeyDataSize(bitWidthIn, elementNum);
    return FAST_FSS_SUCCESS;
}