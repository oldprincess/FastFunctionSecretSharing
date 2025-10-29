#ifndef FAST_FSS_HELPER_MIC_HELPER_H
#define FAST_FSS_HELPER_MIC_HELPER_H

#include <cstddef>
#include <cstdint>

#include "error_code.h"

static int FastFss_helper_checkDcfMicKeyGenParams(
    std::size_t keyDataSize,
    std::size_t zDataSize,
    std::size_t alphaDataSize,
    std::size_t seedDataSize0,
    std::size_t seedDataSize1,
    std::size_t leftBoundaryDataSize,
    std::size_t rightBoundaryDataSize,
    std::size_t bitWidthIn,
    std::size_t bitWidthOut,
    std::size_t elementSize,
    std::size_t elementNum,
    int (*getKeyDataSizeFunc)(std::size_t *,
                              std::size_t,
                              std::size_t,
                              std::size_t,
                              std::size_t))
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
                             elementSize, elementNum);
    if (ret != 0)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    std::size_t intervalNum = leftBoundaryDataSize / elementSize;

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

    if (leftBoundaryDataSize != elementSize * intervalNum ||
        rightBoundaryDataSize != elementSize * intervalNum)
    {
        return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }

    return FAST_FSS_SUCCESS;
}

static int FastFss_helper_checkDcfMicEvalParams(
    std::size_t sharedOutDataSize,
    std::size_t maskedXDataSize,
    std::size_t keyDataSize,
    std::size_t sharedZDataSize,
    std::size_t seedDataSize,
    std::size_t leftBoundaryDataSize,
    std::size_t rightBoundaryDataSize,
    int         partyId,
    std::size_t bitWidthIn,
    std::size_t bitWidthOut,
    std::size_t elementSize,
    std::size_t elementNum,
    std::size_t cacheDataSize,
    int (*getKeyDataSizeFunc)(std::size_t *,
                              std::size_t,
                              std::size_t,
                              std::size_t,
                              std::size_t),
    int (*getCacheDataSizeFunc)(std::size_t *,
                                std::size_t,
                                std::size_t,
                                std::size_t,
                                std::size_t))
{
    int         ret;
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

    if (partyId != 0 && partyId != 1)
    {
        return FAST_FSS_INVALID_PARTY_ID_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, bitWidthOut,
                             elementSize, elementNum);
    if (ret != 0)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    std::size_t intervalNum = leftBoundaryDataSize / elementSize;
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
    if (leftBoundaryDataSize != intervalNum * elementSize ||
        rightBoundaryDataSize != intervalNum * elementSize)
    {
        return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }

    if (cacheDataSize != 0)
    {
        ret = getCacheDataSizeFunc(&needCacheDataSize, bitWidthIn, bitWidthOut,
                                   elementSize, elementNum);
        if (ret != 0)
        {
            return ret;
        }
        if (cacheDataSize != needCacheDataSize)
        {
            return FAST_FSS_INVALID_CACHE_DATA_SIZE_ERROR;
        }
    }

    return FAST_FSS_SUCCESS;
}

#endif // FAST_FSS_HELPER_MIC_HELPER_H