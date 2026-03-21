#ifndef SRC_KERNEL_DPF_H
#define SRC_KERNEL_DPF_H

#include "../impl/dpf.h"
#include "parallel_execute.h"

namespace FastFss::kernel {

template <typename GroupElement>
struct DpfKeyGenTask
{
    void       *key;
    size_t      keyDataSize;
    const void *alpha;
    size_t      alphaDataSize;
    const void *beta;
    size_t      betaDataSize;
    const void *seed0;
    size_t      seedDataSize0;
    const void *seed1;
    size_t      seedDataSize1;
    size_t      bitWidthIn;
    size_t      bitWidthOut;
    size_t      groupSize;
    size_t      elementSize;
    size_t      elementNum;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        std::size_t needKeyDataSize = 0;

        if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (groupSize == 0)
        {
            return FAST_FSS_INVALID_GROUP_SIZE_ERROR;
        }

        needKeyDataSize = impl::dpfGetKeyDataSize<GroupElement>(
            bitWidthIn, bitWidthOut, groupSize, elementNum);
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (alphaDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_ALPHA_DATA_SIZE_ERROR;
        }
        if (betaDataSize != 0)
        {
            if (betaDataSize != elementNum * elementSize * groupSize)
            {
                return FAST_FSS_INVALID_BETA_DATA_SIZE_ERROR;
            }
        }
        else if (groupSize != 1)
        {
            return FAST_FSS_INVALID_BETA_DATA_SIZE_ERROR;
        }
        if (seedDataSize0 != elementNum * 16 ||
            seedDataSize1 != elementNum * 16)
        {
            return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        const GroupElement ONE = 1;

        const GroupElement *alphaPtr = (const GroupElement *)alpha;
        const GroupElement *betaPtr  = (const GroupElement *)beta;
        const std::uint8_t *seed0Ptr = (const std::uint8_t *)seed0;
        const std::uint8_t *seed1Ptr = (const std::uint8_t *)seed1;

        impl::DpfKey<GroupElement> keyObj;
        impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, groupSize, i,
                           elementNum);
        const GroupElement *ptr = &ONE;
        if (betaPtr != nullptr) ptr = betaPtr + groupSize * i;
        impl::dpfKeyGen(keyObj, alphaPtr[i], ptr, seed0Ptr + 16 * i,
                        seed1Ptr + 16 * i, bitWidthIn, bitWidthOut, groupSize);
    }
};

template <typename GroupElement>
struct DpfEvalTask
{
    void       *sharedOut;
    size_t      sharedOutDataSize;
    const void *maskedX;
    size_t      maskedXDataSize;
    const void *key;
    size_t      keyDataSize;
    const void *seed;
    size_t      seedDataSize;
    int         partyId;
    size_t      bitWidthIn;
    size_t      bitWidthOut;
    size_t      groupSize;
    size_t      elementSize;
    size_t      elementNum;
    void       *cache;
    size_t      cacheDataSize;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        std::size_t needKeyDataSize   = 0;
        std::size_t needCacheDataSize = 0;

        if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (groupSize == 0)
        {
            return FAST_FSS_INVALID_GROUP_SIZE_ERROR;
        }
        if (partyId != 0 && partyId != 1)
        {
            return FAST_FSS_INVALID_PARTY_ID_ERROR;
        }

        needKeyDataSize = impl::dpfGetKeyDataSize<GroupElement>(
            bitWidthIn, bitWidthOut, groupSize, elementNum);
        needCacheDataSize =
            impl::dpfGetCacheDataSize<GroupElement>(bitWidthIn, elementNum);
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (seedDataSize != elementNum * 16)
        {
            return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
        }
        if (maskedXDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_MASKED_X_DATA_SIZE_ERROR;
        }
        if (sharedOutDataSize != elementNum * elementSize * groupSize)
        {
            return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
        }
        if (cacheDataSize != 0 && cacheDataSize != needCacheDataSize)
        {
            return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        GroupElement       *sharedOutPtr = (GroupElement *)sharedOut;
        const GroupElement *maskedXPtr   = (const GroupElement *)maskedX;
        const std::uint8_t *seedPtr      = (const std::uint8_t *)seed;

        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement> *cacheObjPtr = nullptr;
        impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, groupSize, i,
                           elementNum);
        if (cache != nullptr)
        {
            impl::dpfCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        impl::dpfEval(sharedOutPtr + i * groupSize, keyObj, maskedXPtr[i],
                      seedPtr + 16 * i, partyId, bitWidthIn, bitWidthOut,
                      groupSize, cacheObjPtr);
    }
};

template <typename GroupElement>
struct DpfEvalAllTask
{
    void       *sharedOut;
    size_t      sharedOutDataSize;
    const void *maskedX;
    size_t      maskedXDataSize;
    const void *key;
    size_t      keyDataSize;
    const void *seed;
    size_t      seedDataSize;
    int         partyId;
    size_t      bitWidthIn;
    size_t      bitWidthOut;
    size_t      groupSize;
    size_t      elementSize;
    size_t      elementNum;
    void       *cache;
    size_t      cacheDataSize;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        std::size_t needKeyDataSize   = 0;
        std::size_t needCacheDataSize = 0;

        if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (groupSize == 0)
        {
            return FAST_FSS_INVALID_GROUP_SIZE_ERROR;
        }
        if (partyId != 0 && partyId != 1)
        {
            return FAST_FSS_INVALID_PARTY_ID_ERROR;
        }

        needKeyDataSize = impl::dpfGetKeyDataSize<GroupElement>(
            bitWidthIn, bitWidthOut, groupSize, elementNum);
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (sharedOutDataSize !=
            elementNum * elementSize * (1ULL << bitWidthIn) * groupSize)
        {
            return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
        }
        if (maskedXDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_MASKED_X_DATA_SIZE_ERROR;
        }
        if (seedDataSize != elementNum * 16)
        {
            return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
        }
        if (cacheDataSize != 0)
        {
            needCacheDataSize =
                impl::dpfGetCacheDataSize<GroupElement>(bitWidthIn, elementNum);
            if (cacheDataSize != needCacheDataSize)
            {
                return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
            }
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        GroupElement       *sharedOutPtr = (GroupElement *)sharedOut;
        const GroupElement *maskedXPtr   = (const GroupElement *)maskedX;
        const std::uint8_t *seedPtr      = (const std::uint8_t *)seed;

        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement> *cacheObjPtr = nullptr;
        impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, groupSize, i,
                           elementNum);
        if (cache != nullptr)
        {
            impl::dpfCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        std::size_t size = (std::size_t)(1ULL << bitWidthIn);
        for (std::size_t j = 0; j < size; ++j)
        {
            std::size_t k =
                groupSize * (std::size_t)((maskedXPtr[i] - j) % size);
            impl::dpfEval(sharedOutPtr + size * i * groupSize + k, keyObj,
                          (GroupElement)j, seedPtr + 16 * i, partyId,
                          bitWidthIn, bitWidthOut, groupSize, cacheObjPtr);
        }
    }
};

template <typename GroupElement>
struct DpfEvalMultiTask
{
    void       *sharedOut;
    size_t      sharedOutDataSize;
    const void *maskedX;
    size_t      maskedXDataSize;
    const void *key;
    size_t      keyDataSize;
    const void *seed;
    size_t      seedDataSize;
    int         partyId;
    const void *point;
    size_t      pointDataSize;
    size_t      bitWidthIn;
    size_t      bitWidthOut;
    size_t      groupSize;
    size_t      elementSize;
    size_t      elementNum;
    void       *cache;
    size_t      cacheDataSize;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        std::size_t needKeyDataSize   = 0;
        std::size_t needCacheDataSize = 0;

        if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (groupSize == 0)
        {
            return FAST_FSS_INVALID_GROUP_SIZE_ERROR;
        }
        if (partyId != 0 && partyId != 1)
        {
            return FAST_FSS_INVALID_PARTY_ID_ERROR;
        }
        if (pointDataSize % elementSize != 0)
        {
            return FAST_FSS_INVALID_POINT_DATA_SIZE_ERROR;
        }

        needKeyDataSize = impl::dpfGetKeyDataSize<GroupElement>(
            bitWidthIn, bitWidthOut, groupSize, elementNum);
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (sharedOutDataSize != elementNum * elementSize *
                                     (pointDataSize / elementSize) * groupSize)
        {
            return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
        }
        if (maskedXDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_MASKED_X_DATA_SIZE_ERROR;
        }
        if (seedDataSize != elementNum * 16)
        {
            return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
        }
        if (cacheDataSize != 0)
        {
            needCacheDataSize =
                impl::dpfGetCacheDataSize<GroupElement>(bitWidthIn, elementNum);
            if (cacheDataSize != needCacheDataSize)
            {
                return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
            }
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        const std::size_t pointNum = pointDataSize / elementSize;

        GroupElement       *sharedOutPtr = (GroupElement *)sharedOut;
        const GroupElement *maskedXPtr   = (const GroupElement *)maskedX;
        const std::uint8_t *seedPtr      = (const std::uint8_t *)seed;
        const GroupElement *pointPtr     = (const GroupElement *)point;

        impl::DpfKey<GroupElement>    keyObj;
        impl::DpfCache<GroupElement>  cacheObj;
        impl::DpfCache<GroupElement> *cacheObjPtr = nullptr;
        impl::dpfKeySetPtr(keyObj, key, bitWidthIn, bitWidthOut, groupSize, i,
                           elementNum);
        if (cache != nullptr)
        {
            impl::dpfCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        for (std::size_t j = 0; j < pointNum; ++j)
        {
            GroupElement tmp = maskedXPtr[i] - pointPtr[j];
            impl::dpfEval(
                sharedOutPtr + pointNum * i * groupSize + j * groupSize, keyObj,
                tmp, seedPtr + 16 * i, partyId, bitWidthIn, bitWidthOut,
                groupSize, cacheObjPtr);
        }
    }
};

} // namespace FastFss::kernel

#endif
