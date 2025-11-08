#ifndef FAST_FSS_HELPER_DCF_HELPER_H
#define FAST_FSS_HELPER_DCF_HELPER_H

#include <cstddef>
#include <cstdint>

#include "error_code.h"

static int FastFss_helper_checkDcfKeyGenParams(
    std::size_t keyDataSize,
    std::size_t alphaDataSize,
    std::size_t betaDataSize,
    std::size_t seedDataSize0,
    std::size_t seedDataSize1,
    std::size_t bitWidthIn,
    std::size_t bitWidthOut,
    std::size_t groupSize,
    std::size_t elementSize,
    std::size_t elementNum,
    int (*getKeyDataSizeFunc)(size_t *keyDataSize,
                              size_t  bitWidthIn,
                              size_t  bitWidthOut,
                              size_t  groupSize,
                              size_t  elementSize,
                              size_t  elementNum))
{
    int         ret;
    std::size_t needKeyDataSize = 0;

    if (!(0 < bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    if (!(0 < bitWidthOut && bitWidthOut <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, bitWidthOut,
                             groupSize, elementSize, elementNum);
    if (ret != 0)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (groupSize == 0)
    {
        return FAST_FSS_INVALID_GROUP_SIZE_ERROR;
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

    if (seedDataSize0 != elementNum * 16 || seedDataSize1 != elementNum * 16)
    {
        return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
    }

    return FAST_FSS_SUCCESS;
}

static int FastFss_helper_checkDcfEvalParams(
    std::size_t sharedOutDataSize,
    std::size_t maskedXDataSize,
    std::size_t keyDataSize,
    std::size_t seedDataSize,
    int         partyId,
    std::size_t bitWidthIn,
    std::size_t bitWidthOut,
    std::size_t groupSize,
    std::size_t elementSize,
    std::size_t elementNum,
    std::size_t cacheDataSize,
    int (*getKeyDataSizeFunc)(size_t *keyDataSize,
                              size_t  bitWidthIn,
                              size_t  bitWidthOut,
                              size_t  groupSize,
                              size_t  elementSize,
                              size_t  elementNum),
    int (*getCacheDataSizeFunc)(size_t *cacheDataSize,
                                size_t  bitWidthIn,
                                size_t  bitWidthOut,
                                size_t  groupSize,
                                size_t  elementSize,
                                size_t  elementNum))
{
    int         ret;
    std::size_t needKeyDataSize   = 0;
    std::size_t needCacheDataSize = 0;

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, bitWidthOut,
                             groupSize, elementSize, elementNum);
    if (ret != 0)
    {
        return ret;
    }
    ret = getCacheDataSizeFunc(&needCacheDataSize, bitWidthIn, bitWidthOut,
                               groupSize, elementSize, elementNum);
    if (ret != 0)
    {
        return ret;
    }

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
    if (groupSize == 0)
    {
        return FAST_FSS_INVALID_GROUP_SIZE_ERROR;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    if (sharedOutDataSize != elementNum * elementSize * groupSize)
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
        if (needCacheDataSize != cacheDataSize)
        {
            return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
        }
    }

    return FAST_FSS_SUCCESS;
}

#endif // FAST_FSS_HELPER_DCF_HELPER_H