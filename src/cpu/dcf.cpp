#include "dcf.hpp"

#include <FastFss/cpu/dcf.h>

#include <cassert>
#include <memory>

enum DCF_ERROR_CODE
{
    DCF_SUCCESS                            = 0,
    DCF_RUNTIME_ERROR                      = -1,
    DCF_INVALID_KEY_DATA_SIZE_ERROR        = -2,
    DCF_INVALID_ALPHA_DATA_SIZE_ERROR      = -3,
    DCF_INVALID_BETA_DATA_SIZE_ERROR       = -4,
    DCF_INVALID_SEED_DATA_SIZE_ERROR       = -5,
    DCF_INVALID_BOUNDARY_DATA_SIZE_ERROR   = -6,
    DCF_INVALID_Z_DATA_SIZE_ERROR          = -7,
    DCF_INVALID_SHARED_OUT_DATA_SIZE_ERROR = -8,
    DCF_INVLIAD_MASKED_X_DATA_SIZE_ERROR   = -9,
    DCF_INVALID_BITWIDTH_ERROR             = -10,
    DCF_INVALID_ELEMENT_SIZE_ERROR         = -11,
    DCF_INVALID_PARTY_ID_ERROR             = -12,
};

namespace FastFss::cpu {

template <typename GroupElement>
static int IMPL_FastFss_cpu_dcfKeyGen(void**      key,
                                      size_t*     keyDataSize,
                                      const void* alpha,
                                      size_t      alphaDataSize,
                                      const void* beta,
                                      size_t      betaDataSize,
                                      const void* seed0,
                                      size_t      seedDataSize0,
                                      const void* seed1,
                                      size_t      seedDataSize1,
                                      size_t      bitWidthIn,
                                      size_t      bitWidthOut,
                                      size_t      elementSize,
                                      size_t      elementNum) noexcept
{
    assert(sizeof(GroupElement) == elementSize);

    if (alphaDataSize != sizeof(GroupElement) * elementNum)
    {
        return DCF_INVALID_ALPHA_DATA_SIZE_ERROR;
    }
    if (beta != nullptr && betaDataSize != sizeof(GroupElement) * elementNum)
    {
        return DCF_INVALID_BETA_DATA_SIZE_ERROR;
    }
    if (!(seedDataSize0 == 16 * elementNum && seedDataSize1 == 16 * elementNum))
    {
        return DCF_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthOut <= elementSize * 8))
    {
        return DCF_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = dcfGetKeyDataSize<GroupElement>( //
        bitWidthIn, bitWidthOut, elementNum                        //
    );                                                             //
    if (*key != nullptr)
    {
        if (*keyDataSize != needKeyDataSize)
        {
            return DCF_INVALID_KEY_DATA_SIZE_ERROR;
        }
    }
    else
    {
        *key = std::malloc(needKeyDataSize);
        if (*key == nullptr)
        {
            return DCF_RUNTIME_ERROR;
        }
        *keyDataSize = needKeyDataSize;
    }

    // ===============================================

    const GroupElement* alphaPtr = (const GroupElement*)alpha;
    const GroupElement* betaPtr  = (const GroupElement*)beta;
    const std::uint8_t* seedPtr0 = (const std::uint8_t*)seed0;
    const std::uint8_t* seedPtr1 = (const std::uint8_t*)seed1;

    for (std::size_t i = 0; i < elementNum; i++)
    {
        DcfKey<GroupElement> dcfKey;
        std::size_t          offsetAlpha = i;
        std::size_t          offsetBeta  = i;
        std::size_t          offsetSeed0 = 16 * i;
        std::size_t          offsetSeed1 = 16 * i;
        dcfKeySetPtr<GroupElement>(dcfKey, *key, bitWidthIn, bitWidthOut, i,
                                   elementNum);
        GroupElement curBeta = (beta == nullptr) ? 1 : betaPtr[offsetBeta];
        dcfKeyGen<GroupElement>(dcfKey, alphaPtr[offsetAlpha], curBeta,
                                seedPtr0 + offsetSeed0, seedPtr1 + offsetSeed1,
                                bitWidthIn, bitWidthOut);
    }

    return DCF_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cpu_dcfEval(void*       sharedOut,
                                    const void* maskedX,
                                    size_t      maskedXDataSize,
                                    const void* key,
                                    size_t      keyDataSize,
                                    const void* seed,
                                    size_t      seedDataSize,
                                    int         partyId,
                                    size_t      bitWidthIn,
                                    size_t      bitWidthOut,
                                    size_t      elementSize,
                                    size_t      elementNum)
{
    assert(sizeof(GroupElement) == elementSize);

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return DCF_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return DCF_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthOut <= elementSize * 8))
    {
        return DCF_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = dcfGetKeyDataSize<GroupElement>( //
        bitWidthIn, bitWidthOut, elementNum                        //
    );                                                             //
    if (keyDataSize != needKeyDataSize)
    {
        return DCF_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return DCF_INVALID_PARTY_ID_ERROR;
    }

    // ===============================================

    GroupElement*       sharedOutPtr = (GroupElement*)sharedOut;
    const GroupElement* maskedXPtr   = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr      = (const std::uint8_t*)seed;

    DcfKey<GroupElement> dcfKey;
    for (std::size_t i = 0; i < elementNum; i++)
    {
        std::size_t offsetSharedOut = i;
        std::size_t offsetMaskedX   = i;
        std::size_t offsetSeed      = 16 * i;

        dcfKeySetPtr<GroupElement>(dcfKey, key, bitWidthIn, bitWidthOut, i,
                                   elementNum);
        sharedOutPtr[offsetSharedOut] = dcfEval<GroupElement>(
            dcfKey, maskedXPtr[offsetMaskedX], seedPtr + offsetSeed, partyId,
            bitWidthIn, bitWidthOut);
    }

    return DCF_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cpu_dcfKeyZip(void**      zippedKey,
                                      size_t*     zippedKeyDataSize,
                                      const void* key,
                                      size_t      keyDataSize,
                                      size_t      bitWidthIn,
                                      size_t      bitWidthOut,
                                      size_t      elementSize,
                                      size_t      elementNum)
{
    return DCF_RUNTIME_ERROR;
}

template <typename GroupElement>
static int IMPL_FastFss_cpu_dcfKeyUnzip(void**      key,
                                        size_t*     keyDataSize,
                                        const void* zippedKey,
                                        size_t      zippedKeyDataSize,
                                        size_t      bitWidthIn,
                                        size_t      bitWidthOut,
                                        size_t      elementSize,
                                        size_t      elementNum)
{
    return DCF_RUNTIME_ERROR;
}

}; // namespace FastFss::cpu

int FastFss_cpu_dcfKeyGen(void**      key,
                          size_t*     keyDataSize,
                          const void* alpha,
                          size_t      alphaDataSize,
                          const void* beta,
                          size_t      betaDataSize,
                          const void* seed0,
                          size_t      seedDataSize0,
                          const void* seed1,
                          size_t      seedDataSize1,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      elementSize,
                          size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyGen<std::uint8_t>(
                key, keyDataSize, alpha, alphaDataSize, beta, betaDataSize,
                seed0, seedDataSize0, seed1, seedDataSize1, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyGen<std::uint16_t>(
                key, keyDataSize, alpha, alphaDataSize, beta, betaDataSize,
                seed0, seedDataSize0, seed1, seedDataSize1, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyGen<std::uint32_t>(
                key, keyDataSize, alpha, alphaDataSize, beta, betaDataSize,
                seed0, seedDataSize0, seed1, seedDataSize1, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyGen<std::uint64_t>(
                key, keyDataSize, alpha, alphaDataSize, beta, betaDataSize,
                seed0, seedDataSize0, seed1, seedDataSize1, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_dcfEval(void*       sharedOut,
                        const void* maskedX,
                        size_t      maskedXDataSize,
                        const void* key,
                        size_t      keyDataSize,
                        const void* seed,
                        size_t      seedDataSize,
                        int         partyId,
                        size_t      bitWidthIn,
                        size_t      bitWidthOut,
                        size_t      elementSize,
                        size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfEval<std::uint8_t>(
                sharedOut, maskedX, maskedXDataSize, key, keyDataSize, seed,
                seedDataSize, partyId, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfEval<std::uint16_t>(
                sharedOut, maskedX, maskedXDataSize, key, keyDataSize, seed,
                seedDataSize, partyId, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfEval<std::uint32_t>(
                sharedOut, maskedX, maskedXDataSize, key, keyDataSize, seed,
                seedDataSize, partyId, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfEval<std::uint64_t>(
                sharedOut, maskedX, maskedXDataSize, key, keyDataSize, seed,
                seedDataSize, partyId, bitWidthIn, bitWidthOut, elementSize,
                elementNum);
        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_dcfKeyZip(void**      zippedKey,
                          size_t*     zippedKeyDataSize,
                          const void* key,
                          size_t      keyDataSize,
                          size_t      bitWidthIn,
                          size_t      bitWidthOut,
                          size_t      elementSize,
                          size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyZip<std::uint8_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyZip<std::uint16_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyZip<std::uint32_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyZip<std::uint64_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);

        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_dcfKeyUnzip(void**      key,
                            size_t*     keyDataSize,
                            const void* zippedKey,
                            size_t      zippedKeyDataSize,
                            size_t      bitWidthIn,
                            size_t      bitWidthOut,
                            size_t      elementSize,
                            size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyUnzip<std::uint8_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyUnzip<std::uint16_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyUnzip<std::uint32_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_dcfKeyUnzip<std::uint64_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                bitWidthOut, elementSize, elementNum);
        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_dcfGetKeyDataSize(size_t bitWidthIn,
                                  size_t bitWidthOut,
                                  size_t elementSize,
                                  size_t elementNum)
{
    switch (elementSize)
    {
        case 1:
            return (int)FastFss::cpu::dcfGetKeyDataSize<std::uint8_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 2:
            return (int)FastFss::cpu::dcfGetKeyDataSize<std::uint16_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 4:
            return (int)FastFss::cpu::dcfGetKeyDataSize<std::uint32_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 8:
            return (int)FastFss::cpu::dcfGetKeyDataSize<std::uint64_t>(
                bitWidthIn, bitWidthOut, elementNum);
        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_dcfGetZippedKeyDataSize(size_t bitWidthIn,
                                        size_t bitWidthOut,
                                        size_t elementSize,
                                        size_t elementNum)
{
    switch (elementSize)
    {
        case 1:
            return (int)FastFss::cpu::dcfGetZippedKeyDataSize<std::uint8_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 2:
            return (int)FastFss::cpu::dcfGetZippedKeyDataSize<std::uint16_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 4:
            return (int)FastFss::cpu::dcfGetZippedKeyDataSize<std::uint32_t>(
                bitWidthIn, bitWidthOut, elementNum);
        case 8:
            return (int)FastFss::cpu::dcfGetZippedKeyDataSize<std::uint64_t>(
                bitWidthIn, bitWidthOut, elementNum);
        default: return DCF_INVALID_ELEMENT_SIZE_ERROR;
    }
}