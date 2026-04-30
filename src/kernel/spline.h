#ifndef SRC_KERNEL_SPLINE_H
#define SRC_KERNEL_SPLINE_H

#include <FastFss/errors.h>

#include "../impl/spline.h"

namespace FastFss::kernel {

template <typename GroupElement>
struct DcfSplineKeyGenTask
{
    void       *key;
    size_t      keyDataSize;
    void       *e;
    size_t      eDataSize;
    void       *beta;
    size_t      betaDataSize;
    const void *alpha;
    size_t      alphaDataSize;
    const void *seed0;
    size_t      seedDataSize0;
    const void *seed1;
    size_t      seedDataSize1;
    const void *coefficients;
    size_t      coefficientsDataSize;
    size_t      degree;
    const void *leftEndpoints;
    size_t      leftEndpointsDataSize;
    const void *rightEndpoints;
    size_t      rightEndpointsDataSize;
    size_t      intervalNum;
    size_t      bitWidthIn;
    size_t      bitWidthOut;
    size_t      elementSize;
    size_t      elementNum;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        std::size_t coeffNum  = degree + 1;
        std::size_t groupSize = intervalNum * coeffNum;
        std::size_t needKeyDataSize =
            impl::dcfSplineGetKeyDataSize<GroupElement>(bitWidthIn, bitWidthOut, intervalNum, degree, elementNum);

        if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (intervalNum == 0)
        {
            return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
        }
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (eDataSize != elementNum * elementSize * groupSize)
        {
            return FAST_FSS_INVALID_Z_DATA_SIZE_ERROR;
        }
        if (betaDataSize != elementNum * elementSize * groupSize)
        {
            return FAST_FSS_INVALID_BETA_DATA_SIZE_ERROR;
        }
        if (alphaDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_ALPHA_DATA_SIZE_ERROR;
        }
        if (seedDataSize0 != elementNum * 16 || seedDataSize1 != elementNum * 16)
        {
            return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
        }
        if (leftEndpointsDataSize != intervalNum * elementSize)
        {
            return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
        }
        if (rightEndpointsDataSize != intervalNum * elementSize)
        {
            return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
        }
        if (coefficientsDataSize != intervalNum * coeffNum * elementSize)
        {
            return FAST_FSS_INVALID_BETA_DATA_SIZE_ERROR;
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        std::size_t         coeffNum          = degree + 1;
        std::size_t         groupSize         = intervalNum * coeffNum;
        GroupElement       *ePtr              = (GroupElement *)e;
        GroupElement       *betaPtr           = (GroupElement *)beta;
        const GroupElement *alphaPtr          = (const GroupElement *)alpha;
        const std::uint8_t *seed0Ptr          = (const std::uint8_t *)seed0;
        const std::uint8_t *seed1Ptr          = (const std::uint8_t *)seed1;
        const GroupElement *coefficientsPtr   = (const GroupElement *)coefficients;
        const GroupElement *leftEndpointsPtr  = (const GroupElement *)leftEndpoints;
        const GroupElement *rightEndpointsPtr = (const GroupElement *)rightEndpoints;

        impl::DcfKey<GroupElement> keyObj;
        impl::dcfKeySetPtr<GroupElement>(keyObj, key, bitWidthIn, bitWidthOut, groupSize, i, elementNum);
        impl::dcfSplineKeyGen<GroupElement>(keyObj, ePtr + groupSize * i, betaPtr + groupSize * i, alphaPtr[i],
                                            coefficientsPtr, seed0Ptr + 16 * i, seed1Ptr + 16 * i, leftEndpointsPtr,
                                            rightEndpointsPtr, intervalNum, degree, bitWidthIn, bitWidthOut);
    }
};

template <typename GroupElement>
struct DcfSplineEvalTask
{
    void       *sharedOut;
    size_t      sharedOutDataSize;
    const void *maskedX;
    size_t      maskedXDataSize;
    const void *key;
    size_t      keyDataSize;
    const void *sharedE;
    size_t      sharedEDataSize;
    const void *sharedBeta;
    size_t      sharedBetaDataSize;
    const void *seed;
    size_t      seedDataSize;
    int         partyId;
    const void *leftEndpoints;
    size_t      leftEndpointsDataSize;
    const void *rightEndpoints;
    size_t      rightEndpointsDataSize;
    size_t      intervalNum;
    size_t      degree;
    size_t      bitWidthIn;
    size_t      bitWidthOut;
    size_t      elementSize;
    size_t      elementNum;
    void       *cache;
    size_t      cacheDataSize;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        std::size_t coeffNum  = degree + 1;
        std::size_t groupSize = intervalNum * coeffNum;
        std::size_t needKeyDataSize =
            impl::dcfSplineGetKeyDataSize<GroupElement>(bitWidthIn, bitWidthOut, intervalNum, degree, elementNum);
        std::size_t needCacheDataSize =
            impl::dcfSplineGetCacheDataSize<GroupElement>(bitWidthIn, bitWidthOut, intervalNum, degree, elementNum);

        if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (partyId != 0 && partyId != 1)
        {
            return FAST_FSS_INVALID_PARTY_ID_ERROR;
        }
        if (intervalNum == 0)
        {
            return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
        }
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (sharedOutDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
        }
        if (maskedXDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_MASKED_X_DATA_SIZE_ERROR;
        }
        if (sharedEDataSize != elementNum * elementSize * groupSize)
        {
            return FAST_FSS_INVALID_Z_DATA_SIZE_ERROR;
        }
        if (sharedBetaDataSize != elementNum * elementSize * groupSize)
        {
            return FAST_FSS_INVALID_BETA_DATA_SIZE_ERROR;
        }
        if (seedDataSize != elementNum * 16)
        {
            return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
        }
        if (leftEndpointsDataSize != intervalNum * elementSize)
        {
            return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
        }
        if (rightEndpointsDataSize != intervalNum * elementSize)
        {
            return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
        }
        if (cache == nullptr || cacheDataSize != needCacheDataSize)
        {
            return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        std::size_t         coeffNum          = degree + 1;
        std::size_t         groupSize         = intervalNum * coeffNum;
        GroupElement       *sharedOutPtr      = (GroupElement *)sharedOut;
        const GroupElement *maskedXPtr        = (const GroupElement *)maskedX;
        const GroupElement *sharedEPtr        = (const GroupElement *)sharedE;
        const GroupElement *sharedBetaPtr     = (const GroupElement *)sharedBeta;
        const std::uint8_t *seedPtr           = (const std::uint8_t *)seed;
        const GroupElement *leftEndpointsPtr  = (const GroupElement *)leftEndpoints;
        const GroupElement *rightEndpointsPtr = (const GroupElement *)rightEndpoints;

        impl::DcfKey<GroupElement>         keyObj;
        impl::DcfSplineCache<GroupElement> cacheObj;
        impl::dcfKeySetPtr<GroupElement>(keyObj, key, bitWidthIn, bitWidthOut, groupSize, i, elementNum);
        impl::dcfSplineCacheSetPtr<GroupElement>(cacheObj, cache, bitWidthIn, groupSize, coeffNum, i, elementNum);
        sharedOutPtr[i] = impl::dcfSplineEval<GroupElement>(
            keyObj, maskedXPtr[i], sharedBetaPtr + groupSize * i, sharedEPtr + groupSize * i, seedPtr + 16 * i, partyId,
            leftEndpointsPtr, rightEndpointsPtr, intervalNum, degree, bitWidthIn, bitWidthOut, cacheObj);
    }
};

} // namespace FastFss::kernel

#endif
