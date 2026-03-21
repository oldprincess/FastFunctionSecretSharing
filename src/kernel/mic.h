#ifndef SRC_KERNEL_MIC_H
#define SRC_KERNEL_MIC_H

#include "../impl/mic.h"
#include "parallel_execute.h"

namespace FastFss::kernel {

template <typename GroupElement>
struct DcfMICKeyGenTask
{
    void       *key;
    size_t      keyDataSize;
    void       *z;
    size_t      zDataSize;
    const void *alpha;
    size_t      alphaDataSize;
    const void *seed0;
    size_t      seedDataSize0;
    const void *seed1;
    size_t      seedDataSize1;
    const void *leftEndpoints;
    size_t      leftEndpointsDataSize;
    const void *rightEndpoints;
    size_t      rightEndpointsDataSize;
    size_t      bitWidthIn;
    size_t      bitWidthOut;
    size_t      elementSize;
    size_t      elementNum;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        const std::size_t intervalNum = leftEndpointsDataSize / elementSize;
        const std::size_t needKeyDataSize =
            impl::dcfGetKeyDataSize<GroupElement>(bitWidthIn, bitWidthOut, 1,
                                                  elementNum);

        if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (zDataSize != elementNum * elementSize * intervalNum)
        {
            return FAST_FSS_INVALID_Z_DATA_SIZE_ERROR;
        }
        if (alphaDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_ALPHA_DATA_SIZE_ERROR;
        }
        if (seedDataSize0 != elementNum * 16 || seedDataSize1 != elementNum * 16)
        {
            return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
        }
        if (leftEndpointsDataSize != intervalNum * elementSize ||
            rightEndpointsDataSize != intervalNum * elementSize)
        {
            return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        const std::size_t  intervalNum       = leftEndpointsDataSize / elementSize;
        GroupElement      *zPtr              = (GroupElement *)z;
        const GroupElement *alphaPtr         = (const GroupElement *)alpha;
        const std::uint8_t *seed0Ptr         = (const std::uint8_t *)seed0;
        const std::uint8_t *seed1Ptr         = (const std::uint8_t *)seed1;
        const GroupElement *leftEndpointsPtr  = (const GroupElement *)leftEndpoints;
        const GroupElement *rightEndpointsPtr = (const GroupElement *)rightEndpoints;

        impl::DcfKey<GroupElement> keyObj;
        impl::dcfKeySetPtr<GroupElement>(keyObj, key, bitWidthIn, bitWidthOut, 1, i,
                                         elementNum);
        impl::dcfMICKeyGen<GroupElement>(
            keyObj, zPtr + intervalNum * i, alphaPtr[i], seed0Ptr + 16 * i,
            seed1Ptr + 16 * i, leftEndpointsPtr, rightEndpointsPtr, intervalNum,
            bitWidthIn, bitWidthOut);
    }
};

template <typename GroupElement>
struct DcfMICEvalTask
{
    void       *sharedOut;
    size_t      sharedOutDataSize;
    const void *maskedX;
    size_t      maskedXDataSize;
    const void *key;
    size_t      keyDataSize;
    const void *sharedZ;
    size_t      sharedZDataSize;
    const void *seed;
    size_t      seedDataSize;
    int         partyId;
    const void *leftEndpoints;
    size_t      leftEndpointsDataSize;
    const void *rightEndpoints;
    size_t      rightEndpointsDataSize;
    size_t      bitWidthIn;
    size_t      bitWidthOut;
    size_t      elementSize;
    size_t      elementNum;
    void       *cache;
    size_t      cacheDataSize;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        const std::size_t intervalNum = leftEndpointsDataSize / elementSize;
        const std::size_t needKeyDataSize =
            impl::dcfGetKeyDataSize<GroupElement>(bitWidthIn, bitWidthOut, 1,
                                                  elementNum);
        const std::size_t needCacheDataSize =
            impl::dcfGetCacheDataSize<GroupElement>(bitWidthIn, 1, elementNum);

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
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (sharedOutDataSize != elementNum * elementSize * intervalNum)
        {
            return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
        }
        if (maskedXDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_MASKED_X_DATA_SIZE_ERROR;
        }
        if (sharedZDataSize != elementNum * elementSize * intervalNum)
        {
            return FAST_FSS_INVALID_Z_DATA_SIZE_ERROR;
        }
        if (seedDataSize != elementNum * 16)
        {
            return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
        }
        if (leftEndpointsDataSize != intervalNum * elementSize ||
            rightEndpointsDataSize != intervalNum * elementSize)
        {
            return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
        }
        if (cacheDataSize != 0 && cacheDataSize != needCacheDataSize)
        {
            return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        const std::size_t  intervalNum       = leftEndpointsDataSize / elementSize;
        GroupElement      *sharedOutPtr      = (GroupElement *)sharedOut;
        const GroupElement *maskedXPtr       = (const GroupElement *)maskedX;
        const GroupElement *sharedZPtr       = (const GroupElement *)sharedZ;
        const std::uint8_t *seedPtr          = (const std::uint8_t *)seed;
        const GroupElement *leftEndpointsPtr  = (const GroupElement *)leftEndpoints;
        const GroupElement *rightEndpointsPtr = (const GroupElement *)rightEndpoints;

        impl::DcfKey<GroupElement>    keyObj;
        impl::DcfCache<GroupElement>  cacheObj;
        impl::DcfCache<GroupElement> *cachePtr = nullptr;

        impl::dcfKeySetPtr<GroupElement>(keyObj, key, bitWidthIn, bitWidthOut, 1, i,
                                         elementNum);
        if (cache != nullptr)
        {
            impl::dcfCacheSetPtr<GroupElement>(cacheObj, cache, bitWidthIn, 1, i,
                                               elementNum);
            cachePtr = &cacheObj;
        }
        impl::dcfMICEval<GroupElement>(
            sharedOutPtr + intervalNum * i, maskedXPtr[i], keyObj,
            sharedZPtr + intervalNum * i, seedPtr + 16 * i, partyId, leftEndpointsPtr,
            rightEndpointsPtr, intervalNum, bitWidthIn, bitWidthOut, cachePtr);
    }
};

} // namespace FastFss::kernel

#endif
