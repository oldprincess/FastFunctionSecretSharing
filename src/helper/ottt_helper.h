#ifndef FAST_FSS_HELPER_OTTT_HELPER_H
#define FAST_FSS_HELPER_OTTT_HELPER_H

#include <cstddef>
#include <cstdint>

#include "error_code.h"

static int FastFss_helper_checkOtttKeyGenParams(
    size_t keyDataSize,
    size_t alphaDataSize,
    size_t bitWidthIn,
    size_t elementSize,
    size_t elementNum,
    int (*getKeyDataSizeFunc)(size_t *keyDataSize,
                              size_t  bitWidthIn,
                              size_t  elementNum))
{
    int         ret;
    std::size_t needKeyDataSize = 0;

    if (!(3 <= bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, elementNum);
    if (ret != 0)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    if (alphaDataSize != elementNum * elementSize)
    {
        return FAST_FSS_INVALID_ALPHA_DATA_SIZE_ERROR;
    }

    return FAST_FSS_SUCCESS;
}

static int FastFss_helper_checkOtttLutEvalParams(
    size_t sharedOutEDataSize,
    size_t sharedOutTDataSize,
    size_t maskedXDataSize,
    size_t keyDataSize,
    int    partyId,
    size_t lookUpTableDataSize,
    size_t bitWidthIn,
    size_t elementSize,
    size_t elementNum,
    int (*getKeyDataSizeFunc)(size_t *keyDataSize,
                              size_t  bitWidthIn,
                              size_t  elementNum))
{
    int         ret;
    std::size_t needKeyDataSize = 0;

    if (!(3 <= bitWidthIn && bitWidthIn <= elementSize * 8))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }

    if (partyId != 0 && partyId != 1)
    {
        return FAST_FSS_INVALID_PARTY_ID_ERROR;
    }

    ret = getKeyDataSizeFunc(&needKeyDataSize, bitWidthIn, elementNum);
    if (ret != 0)
    {
        return ret;
    }
    if (keyDataSize != needKeyDataSize)
    {
        return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
    }

    std::size_t eachLookUpTableSize = elementSize * (1ULL << bitWidthIn);
    if (lookUpTableDataSize % eachLookUpTableSize != 0)
    {
        return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
    }

    std::size_t lookUpTableNum = lookUpTableDataSize / eachLookUpTableSize;

    if (sharedOutEDataSize != elementNum * elementSize)
    {
        return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
    }

    if (sharedOutTDataSize != elementNum * elementSize * lookUpTableNum)
    {
        return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
    }

    if (maskedXDataSize != elementSize * elementNum)
    {
        return FAST_FSS_INVALID_MASKED_X_DATA_SIZE_ERROR;
    }

    return FAST_FSS_SUCCESS;
}

#endif // FAST_FSS_HELPER_Ottt_HELPER_H