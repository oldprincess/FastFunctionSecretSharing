#ifndef FAST_FSS_HELPER_GROTTO_HELPER_H
#define FAST_FSS_HELPER_GROTTO_HELPER_H

#include <cstddef>
#include <cstdint>

#include "error_code.h"

// int FastFss_cpu_grottoGetKeyDataSize(size_t *keyDataSize,
//                                      size_t  bitWidthIn,
//                                      size_t  elementSize,
//                                      size_t  elementNum);

// int FastFss_cpu_grottoGetCacheDataSize(size_t *cacheDataSize,
//                                        size_t  bitWidthIn,
//                                        size_t  elementSize,
//                                        size_t  elementNum)
static int FastFss_helper_checkGrottoKeyGenParams(
    size_t keyDataSize,
    size_t alphaDataSize,
    size_t seedDataSize0,
    size_t seedDataSize1,
    size_t bitWidthIn,
    size_t elementSize,
    size_t elementNum,
    int (*getKeyDataSizeFunc)(size_t *, size_t, size_t, size_t))
{
    int    ret             = 0;
    size_t needKeyDataSize = 0;

    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    if (alphaDataSize != elementSize * elementNum)
    {
        return FAST_FSS_INVALID_ALPHA_DATA_SIZE_ERROR;
    }

    if (seedDataSize0 != elementNum * 16 || seedDataSize1 != elementNum * 16)
    {
        return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, elementSize,
                             elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    return FAST_FSS_SUCCESS;
}

static int FastFss_helper_checkGrottoEvalParams(
    size_t sharedOutDataSize,
    size_t maskedXDataSize,
    size_t keyDataSize,
    size_t seedDataSize,
    int    partyId,
    size_t bitWidthIn,
    size_t elementSize,
    size_t elementNum,
    size_t cacheDataSize,
    int (*getKeyDataSizeFunc)(size_t *, size_t, size_t, size_t),
    int (*getCacheDataSizeFunc)(size_t *, size_t, size_t, size_t))
{
    int    ret               = 0;
    size_t needKeyDataSize   = 0;
    size_t needCacheDataSize = 0;

    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    if (sharedOutDataSize != elementSize * elementNum)
    {
        return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
    }

    if (maskedXDataSize != elementSize * elementNum)
    {
        return FAST_FSS_INVALID_MASKED_X_DATA_SIZE_ERROR;
    }

    if (seedDataSize != elementNum * 16)
    {
        return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
    }

    if (!(partyId == 0 || partyId == 1))
    {
        return FAST_FSS_INVALID_PARTY_ID_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, elementSize,
                             elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    if (cacheDataSize != 0)
    {
        ret = getCacheDataSizeFunc(&needCacheDataSize, bitWidthIn, elementSize,
                                   elementNum);
        if (ret != FAST_FSS_SUCCESS)
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

static int FastFss_helper_checkGrottoEqEvalParams(
    size_t sharedOutDataSize,
    size_t maskedXDataSize,
    size_t keyDataSize,
    size_t seedDataSize,
    int    partyId,
    size_t bitWidthIn,
    size_t elementSize,
    size_t elementNum,
    size_t cacheDataSize,
    int (*getKeyDataSizeFunc)(size_t *, size_t, size_t, size_t),
    int (*getCacheDataSizeFunc)(size_t *, size_t, size_t, size_t))
{
    return FastFss_helper_checkGrottoEvalParams(
        sharedOutDataSize, maskedXDataSize, keyDataSize, seedDataSize, partyId,
        bitWidthIn, elementSize, elementNum, cacheDataSize, getKeyDataSizeFunc,
        getCacheDataSizeFunc);
}

static int FastFss_helper_checkGrottoEqMultiEvalParams(
    size_t sharedOutDataSize,
    size_t maskedXDataSize,
    size_t keyDataSize,
    size_t seedDataSize,
    int    partyId,
    size_t pointDataSize,
    size_t bitWidthIn,
    size_t elementSize,
    size_t elementNum,
    size_t cacheDataSize,
    int (*getKeyDataSizeFunc)(size_t *, size_t, size_t, size_t),
    int (*getCacheDataSizeFunc)(size_t *, size_t, size_t, size_t))
{
    int    ret               = 0;
    size_t needKeyDataSize   = 0;
    size_t needCacheDataSize = 0;
    size_t pointNum          = pointDataSize / elementSize;

    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    if (!(pointDataSize == elementSize * pointNum))
    {
        return FAST_FSS_INVALID_POINT_DATA_SIZE_ERROR;
    }

    if (sharedOutDataSize != elementSize * elementNum * pointNum)
    {
        return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
    }

    if (maskedXDataSize != elementSize * elementNum)
    {
        return FAST_FSS_INVALID_MASKED_X_DATA_SIZE_ERROR;
    }

    if (seedDataSize != elementNum * 16)
    {
        return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
    }

    if (!(partyId == 0 || partyId == 1))
    {
        return FAST_FSS_INVALID_PARTY_ID_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, elementSize,
                             elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    if (cacheDataSize != 0)
    {
        ret = getCacheDataSizeFunc(&needCacheDataSize, bitWidthIn, elementSize,
                                   elementNum);
        if (ret != FAST_FSS_SUCCESS)
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

static int FastFss_helper_checkGrottoMICEvalParams(
    size_t sharedOutDataSize,
    size_t maskedXDataSize,
    size_t keyDataSize,
    size_t seedDataSize,
    int    partyId,
    size_t leftBoundaryDataSize,
    size_t rightBoundaryDataSize,
    size_t bitWidthIn,
    size_t elementSize,
    size_t elementNum,
    size_t cacheDataSize,
    int (*getKeyDataSizeFunc)(size_t *, size_t, size_t, size_t),
    int (*getCacheDataSizeFunc)(size_t *, size_t, size_t, size_t))
{
    int    ret               = 0;
    size_t needKeyDataSize   = 0;
    size_t needCacheDataSize = 0;
    size_t intervalNum       = leftBoundaryDataSize / elementSize;

    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    if (intervalNum * elementSize != leftBoundaryDataSize ||
        intervalNum * elementSize != rightBoundaryDataSize)
    {
        return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }

    if (sharedOutDataSize != elementNum * elementSize * intervalNum)
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

    if (!(partyId == 0 || partyId == 1))
    {
        return FAST_FSS_INVALID_PARTY_ID_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, elementSize,
                             elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    if (cacheDataSize != 0)
    {
        ret = getCacheDataSizeFunc(&needCacheDataSize, bitWidthIn, elementSize,
                                   elementNum);
        if (ret != FAST_FSS_SUCCESS)
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

static int FastFss_helper_checkGrottoLutEval_exParams(
    size_t sharedOutEDataSize,
    size_t sharedOutTDataSize,
    size_t maskedXDataSize,
    size_t keyDataSize,
    size_t seedDataSize,
    int    partyId,
    size_t lookUpTableDataSize,
    size_t lutBitWidth,
    size_t bitWidthIn,
    size_t bitWidthOut,
    size_t elementSize,
    size_t elementNum,
    size_t cacheDataSize,
    int (*getKeyDataSizeFunc)(size_t *, size_t, size_t, size_t),
    int (*getCacheDataSizeFunc)(size_t *, size_t, size_t, size_t))
{
    int    ret               = 0;
    size_t needKeyDataSize   = 0;
    size_t needCacheDataSize = 0;

    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    if (lutBitWidth > bitWidthIn)
    {
        return FAST_FSS_INVALID_LOOKUP_TABLE_DATA_SIZE_ERROR;
    }

    size_t lutSize = 1ULL << lutBitWidth;
    size_t lutNum  = lookUpTableDataSize / (elementSize * lutSize);
    if (lookUpTableDataSize != lutSize * elementSize * lutNum)
    {
        return FAST_FSS_INVALID_LOOKUP_TABLE_DATA_SIZE_ERROR;
    }

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

    if (!(partyId == 0 || partyId == 1))
    {
        return FAST_FSS_INVALID_PARTY_ID_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, elementSize,
                             elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    if (cacheDataSize != 0)
    {
        ret = getCacheDataSizeFunc(&needCacheDataSize, bitWidthIn, elementSize,
                                   elementNum);
        if (ret != FAST_FSS_SUCCESS)
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
static int FastFss_helper_checkGrottoLutEvalParams(
    size_t sharedOutEDataSize,
    size_t sharedOutTDataSize,
    size_t maskedXDataSize,
    size_t keyDataSize,
    size_t seedDataSize,
    int    partyId,
    size_t lookUpTableDataSize,
    size_t bitWidthIn,
    size_t bitWidthOut,
    size_t elementSize,
    size_t elementNum,
    size_t cacheDataSize,
    int (*getKeyDataSizeFunc)(size_t *, size_t, size_t, size_t),
    int (*getCacheDataSizeFunc)(size_t *, size_t, size_t, size_t))
{
    return FastFss_helper_checkGrottoLutEval_exParams(
        sharedOutEDataSize, sharedOutTDataSize, maskedXDataSize, keyDataSize,
        seedDataSize, partyId, lookUpTableDataSize, bitWidthIn, bitWidthIn,
        bitWidthOut, elementSize, elementNum, cacheDataSize, getKeyDataSizeFunc,
        getCacheDataSizeFunc);
}
static int FastFss_helper_checkGrottoLutEval_ex2Params(
    size_t sharedOutEDataSize,
    size_t sharedOutTDataSize,
    size_t maskedXDataSize,
    size_t keyDataSize,
    size_t seedDataSize,
    int    partyId,
    size_t pointsDataSize,
    size_t lookUpTableDataSize,
    size_t bitWidthIn,
    size_t bitWidthOut,
    size_t elementSize,
    size_t elementNum,
    size_t cacheDataSize,
    int (*getKeyDataSizeFunc)(size_t *, size_t, size_t, size_t),
    int (*getCacheDataSizeFunc)(size_t *, size_t, size_t, size_t))
{
    int    ret               = 0;
    size_t needKeyDataSize   = 0;
    size_t needCacheDataSize = 0;

    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    if (pointsDataSize % elementSize != 0)
    {
        return FAST_FSS_INVALID_POINT_DATA_SIZE_ERROR;
    }

    if (lookUpTableDataSize % pointsDataSize != 0)
    {
        return FAST_FSS_INVALID_LOOKUP_TABLE_DATA_SIZE_ERROR;
    }
    size_t lutNum = lookUpTableDataSize / pointsDataSize;
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

    if (!(partyId == 0 || partyId == 1))
    {
        return FAST_FSS_INVALID_PARTY_ID_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, elementSize,
                             elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    if (cacheDataSize != 0)
    {
        ret = getCacheDataSizeFunc(&needCacheDataSize, bitWidthIn, elementSize,
                                   elementNum);
        if (ret != FAST_FSS_SUCCESS)
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

static int FastFss_helper_checkGrottoIntervalLutEvalParams(
    size_t sharedOutEDataSize,
    size_t sharedOutTDataSize,
    size_t maskedXDataSize,
    size_t keyDataSize,
    size_t seedDataSize,
    int    partyId,
    size_t leftBoundaryDataSize,
    size_t rightBoundaryDataSize,
    size_t lookUpTableDataSize,
    size_t bitWidthIn,
    size_t bitWidthOut,
    size_t elementSize,
    size_t elementNum,
    size_t cacheDataSize,
    int (*getKeyDataSizeFunc)(size_t *, size_t, size_t, size_t),
    int (*getCacheDataSizeFunc)(size_t *, size_t, size_t, size_t))
{
    int    ret               = 0;
    size_t needKeyDataSize   = 0;
    size_t needCacheDataSize = 0;
    size_t intervalNum       = leftBoundaryDataSize / elementSize;

    if (!(6 <= bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    if (!(leftBoundaryDataSize == rightBoundaryDataSize &&
          leftBoundaryDataSize % elementSize == 0))
    {
        return FAST_FSS_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }

    if (lookUpTableDataSize % (elementSize * intervalNum) != 0)
    {
        return FAST_FSS_INVALID_LOOKUP_TABLE_DATA_SIZE_ERROR;
    }
    size_t lutNum = lookUpTableDataSize / (elementSize * intervalNum);
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

    if (!(partyId == 0 || partyId == 1))
    {
        return FAST_FSS_INVALID_PARTY_ID_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, elementSize,
                             elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    if (cacheDataSize != 0)
    {
        ret = getCacheDataSizeFunc(&needCacheDataSize, bitWidthIn, elementSize,
                                   elementNum);
        if (ret != FAST_FSS_SUCCESS)
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

#endif // FAST_FSS_HELPER_GROTTO_HELPER_H