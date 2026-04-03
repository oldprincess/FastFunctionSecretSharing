#ifndef SRC_KERNEL_GROTTO_H
#define SRC_KERNEL_GROTTO_H

#include <FastFss/errors.h>

#include "../impl/grotto.h"

namespace FastFss::kernel {

template <typename GroupElement>
struct GrottoKeyGenTask
{
    void       *key;
    size_t      keyDataSize;
    const void *alpha;
    size_t      alphaDataSize;
    const void *seed0;
    size_t      seedDataSize0;
    const void *seed1;
    size_t      seedDataSize1;
    size_t      bitWidthIn;
    size_t      elementSize;
    size_t      elementNum;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (alphaDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_ALPHA_DATA_SIZE_ERROR;
        }
        if (seedDataSize0 != elementNum * 16 || seedDataSize1 != elementNum * 16)
        {
            return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
        }

        std::size_t needKeyDataSize = impl::grottoGetKeyDataSize<GroupElement>(bitWidthIn, elementNum);
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        impl::GrottoKey<GroupElement> keyObj;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        impl::grottoKeyGen(keyObj, ((const GroupElement *)alpha)[i], ((const std::uint8_t *)seed0) + 16 * i,
                           ((const std::uint8_t *)seed1) + 16 * i, bitWidthIn);
    }
};

template <typename GroupElement>
struct GrottoEqEvalTask
{
    void       *sharedBooleanOut;
    size_t      sharedOutDataSize;
    const void *maskedX;
    size_t      maskedXDataSize;
    const void *key;
    size_t      keyDataSize;
    const void *seed;
    size_t      seedDataSize;
    int         partyId;
    size_t      bitWidthIn;
    size_t      elementSize;
    size_t      elementNum;
    void       *cache;
    size_t      cacheDataSize;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (sharedOutDataSize != elementNum * elementSize)
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
        if (partyId != 0 && partyId != 1)
        {
            return FAST_FSS_INVALID_PARTY_ID_ERROR;
        }

        std::size_t needKeyDataSize = impl::grottoGetKeyDataSize<GroupElement>(bitWidthIn, elementNum);
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (cacheDataSize != 0)
        {
            std::size_t needCacheDataSize = impl::grottoGetCacheDataSize<GroupElement>(bitWidthIn, elementNum);
            if (cacheDataSize != needCacheDataSize)
            {
                return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
            }
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
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

        ((GroupElement *)sharedBooleanOut)[i] =
            impl::grottoEqEval(keyObj, ((const GroupElement *)maskedX)[i], ((const std::uint8_t *)seed) + 16 * i,
                               partyId, bitWidthIn, cacheObjPtr);
    }
};

template <typename GroupElement>
struct GrottoEvalTask
{
    void       *sharedBooleanOut;
    size_t      sharedOutDataSize;
    const void *maskedX;
    size_t      maskedXDataSize;
    const void *key;
    size_t      keyDataSize;
    const void *seed;
    size_t      seedDataSize;
    bool        equalBound;
    int         partyId;
    size_t      bitWidthIn;
    size_t      elementSize;
    size_t      elementNum;
    void       *cache;
    size_t      cacheDataSize;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (sharedOutDataSize != elementNum * elementSize)
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
        if (partyId != 0 && partyId != 1)
        {
            return FAST_FSS_INVALID_PARTY_ID_ERROR;
        }

        std::size_t needKeyDataSize = impl::grottoGetKeyDataSize<GroupElement>(bitWidthIn, elementNum);
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (cacheDataSize != 0)
        {
            std::size_t needCacheDataSize = impl::grottoGetCacheDataSize<GroupElement>(bitWidthIn, elementNum);
            if (cacheDataSize != needCacheDataSize)
            {
                return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
            }
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
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

        ((GroupElement *)sharedBooleanOut)[i] =
            impl::grottoEval(keyObj, ((const GroupElement *)maskedX)[i], ((const std::uint8_t *)seed) + 16 * i, partyId,
                             bitWidthIn, equalBound, cacheObjPtr);
    }
};

template <typename GroupElement>
struct GrottoMICEvalTask
{
    void       *sharedBooleanOut;
    size_t      sharedBooleanOutDataSize;
    const void *maskedX;
    size_t      maskedXDataSize;
    const void *key;
    size_t      keyDataSize;
    const void *seed;
    size_t      seedDataSize;
    int         partyId;
    const void *leftEndpoints;
    size_t      leftEndpointsDataSize;
    const void *rightEndpoints;
    size_t      rightEndpointsDataSize;
    size_t      bitWidthIn;
    size_t      elementSize;
    size_t      elementNum;
    void       *cache;
    size_t      cacheDataSize;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        std::size_t intervalNum = leftEndpointsDataSize / elementSize;
        if (intervalNum * elementSize != leftEndpointsDataSize || intervalNum * elementSize != rightEndpointsDataSize)
        {
            return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
        }
        if (sharedBooleanOutDataSize != elementNum * elementSize * intervalNum)
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
        if (partyId != 0 && partyId != 1)
        {
            return FAST_FSS_INVALID_PARTY_ID_ERROR;
        }

        std::size_t needKeyDataSize = impl::grottoGetKeyDataSize<GroupElement>(bitWidthIn, elementNum);
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (cacheDataSize != 0)
        {
            std::size_t needCacheDataSize = impl::grottoGetCacheDataSize<GroupElement>(bitWidthIn, elementNum);
            if (cacheDataSize != needCacheDataSize)
            {
                return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
            }
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        std::size_t intervalNum = leftEndpointsDataSize / elementSize;

        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement> *cacheObjPtr = nullptr;

        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }

        impl::grottoMICEval(((GroupElement *)sharedBooleanOut) + intervalNum * i, keyObj,
                            ((const GroupElement *)maskedX)[i], ((const std::uint8_t *)seed) + 16 * i, partyId,
                            (const GroupElement *)leftEndpoints, (const GroupElement *)rightEndpoints, intervalNum,
                            bitWidthIn, cacheObjPtr);
    }
};

template <typename GroupElement>
struct GrottoIntervalLutEvalTask
{
    void       *sharedOutE;
    size_t      sharedOutEDataSize;
    void       *sharedOutT;
    size_t      sharedOutTDataSize;
    const void *maskedX;
    size_t      maskedXDataSize;
    const void *key;
    size_t      keyDataSize;
    const void *seed;
    size_t      seedDataSize;
    int         partyId;
    const void *leftEndpoints;
    size_t      leftEndpointsDataSize;
    const void *rightEndpoints;
    size_t      rightEndpointsDataSize;
    const void *lookUpTable;
    size_t      lookUpTableDataSize;
    size_t      bitWidthIn;
    size_t      bitWidthOut;
    size_t      elementSize;
    size_t      elementNum;
    void       *cache;
    size_t      cacheDataSize;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (!(leftEndpointsDataSize == rightEndpointsDataSize && leftEndpointsDataSize % elementSize == 0))
        {
            return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
        }
        std::size_t intervalNum = leftEndpointsDataSize / elementSize;
        if (lookUpTableDataSize % (elementSize * intervalNum) != 0)
        {
            return FAST_FSS_INVALID_LOOKUP_TABLE_DATA_SIZE_ERROR;
        }
        std::size_t lutNum = lookUpTableDataSize / (elementSize * intervalNum);
        if (sharedOutEDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
        }
        if (sharedOutTDataSize != elementNum * elementSize * lutNum)
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
        if (partyId != 0 && partyId != 1)
        {
            return FAST_FSS_INVALID_PARTY_ID_ERROR;
        }

        std::size_t needKeyDataSize = impl::grottoGetKeyDataSize<GroupElement>(bitWidthIn, elementNum);
        if (keyDataSize != needKeyDataSize)
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (cacheDataSize != 0)
        {
            std::size_t needCacheDataSize = impl::grottoGetCacheDataSize<GroupElement>(bitWidthIn, elementNum);
            if (cacheDataSize != needCacheDataSize)
            {
                return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
            }
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        std::size_t intervalNum = leftEndpointsDataSize / elementSize;
        std::size_t lutNum      = lookUpTableDataSize / leftEndpointsDataSize;

        impl::GrottoKey<GroupElement>    keyObj;
        impl::GrottoCache<GroupElement>  cacheObj;
        impl::GrottoCache<GroupElement> *cacheObjPtr = nullptr;

        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }

        impl::grottoIntervalLutEval(((GroupElement *)sharedOutE) + i, ((GroupElement *)sharedOutT) + i * lutNum, keyObj,
                                    ((const GroupElement *)maskedX)[i], ((const std::uint8_t *)seed) + 16 * i, partyId,
                                    (const GroupElement *)leftEndpoints, (const GroupElement *)rightEndpoints,
                                    (const GroupElement *)lookUpTable, lutNum, intervalNum, bitWidthIn, cacheObjPtr);
    }
};

} // namespace FastFss::kernel

#endif
