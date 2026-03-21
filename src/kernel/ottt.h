#ifndef SRC_KERNEL_OTTT_H
#define SRC_KERNEL_OTTT_H

#include "../impl/ottt.h"
#include "parallel_execute.h"

namespace FastFss::kernel {

template <typename GroupElement>
struct OtttKeyGenTask
{
    void       *key;
    size_t      keyDataSize;
    const void *alpha;
    size_t      alphaDataSize;
    size_t      bitWidthIn;
    size_t      elementSize;
    size_t      elementNum;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        if (!(3 <= bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (keyDataSize != impl::otttGetKeyDataSize(bitWidthIn, elementNum))
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }
        if (alphaDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_ALPHA_DATA_SIZE_ERROR;
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        const auto *alphaPtr = (const GroupElement *)alpha;
        auto       *keyPtr   = (std::uint8_t *)key;
        const auto  keyOffset = i * (1ULL << bitWidthIn) / 8;
        impl::otttKeyGen<GroupElement>(keyPtr + keyOffset, alphaPtr[i], bitWidthIn);
    }
};

template <typename GroupElement>
struct OtttLutEvalTask
{
    void       *sharedOutE;
    size_t      sharedOutEDataSize;
    void       *sharedOutT;
    size_t      sharedOutTDataSize;
    const void *maskedX;
    size_t      maskedXDataSize;
    const void *key;
    size_t      keyDataSize;
    int         partyId;
    const void *lookUpTable;
    size_t      lookUpTableDataSize;
    size_t      bitWidthIn;
    size_t      elementSize;
    size_t      elementNum;
    void       *cudaStreamPtr = nullptr;

    int check() noexcept
    {
        if (!(3 <= bitWidthIn && bitWidthIn <= elementSize * 8))
        {
            return FAST_FSS_INVALID_BITWIDTH_ERROR;
        }
        if (partyId != 0 && partyId != 1)
        {
            return FAST_FSS_INVALID_PARTY_ID_ERROR;
        }
        if (keyDataSize != impl::otttGetKeyDataSize(bitWidthIn, elementNum))
        {
            return FAST_FSS_INVALID_KEY_DATA_SIZE_ERROR;
        }

        const std::size_t eachLookUpTableSize = elementSize * (1ULL << bitWidthIn);
        if (lookUpTableDataSize % eachLookUpTableSize != 0)
        {
            return FAST_FSS_INVALID_SEED_DATA_SIZE_ERROR;
        }
        const std::size_t lookUpTableNum = lookUpTableDataSize / eachLookUpTableSize;
        if (sharedOutEDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
        }
        if (sharedOutTDataSize != elementNum * elementSize * lookUpTableNum)
        {
            return FAST_FSS_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
        }
        if (maskedXDataSize != elementNum * elementSize)
        {
            return FAST_FSS_INVALID_MASKED_X_DATA_SIZE_ERROR;
        }
        return FAST_FSS_SUCCESS;
    }

    FAST_FSS_DEVICE void operator()(std::size_t i) const noexcept
    {
        const auto *maskedXPtr    = (const GroupElement *)maskedX;
        const auto *keyPtr        = (const std::uint8_t *)key;
        const auto *lutPtr        = (const GroupElement *)lookUpTable;
        auto       *sharedOutEPtr = (GroupElement *)sharedOutE;
        auto       *sharedOutTPtr = (GroupElement *)sharedOutT;
        const auto  keyOffset     = i * (1ULL << bitWidthIn) / 8;

        impl::otttLutEval<GroupElement>(sharedOutEPtr + i, sharedOutTPtr + i,
                                        maskedXPtr[i], keyPtr + keyOffset, lutPtr,
                                        partyId, bitWidthIn);
    }
};

} // namespace FastFss::kernel

#endif
