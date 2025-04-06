#include "grotto.hpp"

#include <FastFss/cpu/grotto.h>

#include <cassert>
#include <memory>

enum GROTTO_ERROR_CODE
{
    GROTTO_SUCCESS                            = 0,
    GROTTO_RUNTIME_ERROR                      = -1,
    GROTTO_INVALID_KEY_DATA_SIZE_ERROR        = -2,
    GROTTO_INVALID_ALPHA_DATA_SIZE_ERROR      = -3,
    GROTTO_INVALID_SEED_DATA_SIZE_ERROR       = -5,
    GROTTO_INVALID_BOUNDARY_DATA_SIZE_ERROR   = -6,
    GROTTO_INVALID_Z_DATA_SIZE_ERROR          = -7,
    GROTTO_INVALID_SHARED_OUT_DATA_SIZE_ERROR = -8,
    GROTTO_INVLIAD_MASKED_X_DATA_SIZE_ERROR   = -9,
    GROTTO_INVALID_BITWIDTH_ERROR             = -10,
    GROTTO_INVALID_ELEMENT_SIZE_ERROR         = -11,
    GROTTO_INVALID_PARTY_ID_ERROR             = -12,
};

namespace FastFss::cpu {

template <typename GroupElement>
static int IMPL_FastFss_cpu_grottoKeyGen(void**      key,
                                         size_t*     keyDataSize,
                                         const void* alpha,
                                         size_t      alphaDataSize,
                                         const void* seed0,
                                         size_t      seedDataSize0,
                                         const void* seed1,
                                         size_t      seedDataSize1,
                                         size_t      bitWidthIn,
                                         size_t      elementSize,
                                         size_t      elementNum) noexcept
{
    assert(sizeof(GroupElement) == elementSize);

    if (alphaDataSize != sizeof(GroupElement) * elementNum)
    {
        return GROTTO_INVALID_ALPHA_DATA_SIZE_ERROR;
    }
    if (!(seedDataSize0 == 16 * elementNum && seedDataSize1 == 16 * elementNum))
    {
        return GROTTO_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = grottoGetKeyDataSize<GroupElement>( //
        bitWidthIn, elementNum                                        //
    );                                                                //
    if (*key != nullptr)
    {
        if (*keyDataSize != needKeyDataSize)
        {
            return GROTTO_INVALID_KEY_DATA_SIZE_ERROR;
        }
    }
    else
    {
        *key = std::malloc(needKeyDataSize);
        if (*key == nullptr)
        {
            return GROTTO_RUNTIME_ERROR;
        }
        *keyDataSize = needKeyDataSize;
    }

    // ===============================================

    const GroupElement* alphaPtr = (const GroupElement*)alpha;
    const std::uint8_t* seedPtr0 = (const std::uint8_t*)seed0;
    const std::uint8_t* seedPtr1 = (const std::uint8_t*)seed1;

    for (std::size_t i = 0; i < elementNum; i++)
    {
        GrottoKey<GroupElement> grottoKey;
        std::size_t             offsetAlpha = i;
        std::size_t             offsetBeta  = i;
        std::size_t             offsetSeed0 = 16 * i;
        std::size_t             offsetSeed1 = 16 * i;
        grottoKeySetPtr<GroupElement>(grottoKey, *key, bitWidthIn, i,
                                      elementNum);
        grottoKeyGen<GroupElement>(grottoKey, alphaPtr[offsetAlpha],
                                   seedPtr0 + offsetSeed0,
                                   seedPtr1 + offsetSeed1, bitWidthIn);
    }

    return GROTTO_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cpu_grottoEval(void*       sharedOut,
                                       const void* maskedX,
                                       size_t      maskedXDataSize,
                                       const void* key,
                                       size_t      keyDataSize,
                                       const void* seed,
                                       size_t      seedDataSize,
                                       bool        equalBound,
                                       int         partyId,
                                       size_t      bitWidthIn,
                                       size_t      elementSize,
                                       size_t      elementNum)
{
    assert(sizeof(GroupElement) == elementSize);

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return GROTTO_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return GROTTO_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = grottoGetKeyDataSize<GroupElement>( //
        bitWidthIn, elementNum                                        //
    );                                                                //
    if (keyDataSize != needKeyDataSize)
    {
        return GROTTO_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return GROTTO_INVALID_PARTY_ID_ERROR;
    }

    // ===============================================

    GroupElement*       sharedOutPtr = (GroupElement*)sharedOut;
    const GroupElement* maskedXPtr   = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr      = (const std::uint8_t*)seed;

    GrottoKey<GroupElement> grottoKey;
    for (std::size_t i = 0; i < elementNum; i++)
    {
        std::size_t offsetSharedOut = i;
        std::size_t offsetMaskedX   = i;
        std::size_t offsetSeed      = 16 * i;

        grottoKeySetPtr<GroupElement>(grottoKey, key, bitWidthIn, i,
                                      elementNum);
        sharedOutPtr[offsetSharedOut] = grottoEval<GroupElement>( //
            grottoKey, maskedXPtr[offsetMaskedX], seedPtr + offsetSeed, partyId,
            bitWidthIn, equalBound);
    }

    return GROTTO_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cpu_grottoEvalEq(void*       sharedOut,
                                         const void* maskedX,
                                         size_t      maskedXDataSize,
                                         const void* key,
                                         size_t      keyDataSize,
                                         const void* seed,
                                         size_t      seedDataSize,
                                         int         partyId,
                                         size_t      bitWidthIn,
                                         size_t      elementSize,
                                         size_t      elementNum)
{
    assert(sizeof(GroupElement) == elementSize);

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return GROTTO_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return GROTTO_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = grottoGetKeyDataSize<GroupElement>( //
        bitWidthIn, elementNum                                        //
    );                                                                //
    if (keyDataSize != needKeyDataSize)
    {
        return GROTTO_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return GROTTO_INVALID_PARTY_ID_ERROR;
    }

    // ===============================================

    GroupElement*       sharedOutPtr = (GroupElement*)sharedOut;
    const GroupElement* maskedXPtr   = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr      = (const std::uint8_t*)seed;

    GrottoKey<GroupElement> grottoKey;
    for (std::size_t i = 0; i < elementNum; i++)
    {
        std::size_t offsetSharedOut = i;
        std::size_t offsetMaskedX   = i;
        std::size_t offsetSeed      = 16 * i;

        grottoKeySetPtr<GroupElement>(grottoKey, key, bitWidthIn, i,
                                      elementNum);
        sharedOutPtr[offsetSharedOut] = grottoEvalEq<GroupElement>( //
            grottoKey, maskedXPtr[offsetMaskedX], seedPtr + offsetSeed, partyId,
            bitWidthIn);
    }

    return GROTTO_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cpu_grottoMICEval(void*       sharedBooleanOut,
                                          size_t      sharedBooleanOutDataSize,
                                          const void* maskedX,
                                          size_t      maskedXDataSize,
                                          const void* key,
                                          size_t      keyDataSize,
                                          const void* seed,
                                          size_t      seedDataSize,
                                          int         partyId,
                                          const void* leftBoundary,
                                          size_t      leftBoundaryDataSize,
                                          const void* rightBoundary,
                                          size_t      rightBoundaryDataSize,
                                          size_t      bitWidthIn,
                                          size_t      elementSize,
                                          size_t      elementNum) noexcept
{
    assert(sizeof(GroupElement) == elementSize);
    if (leftBoundaryDataSize != rightBoundaryDataSize ||
        leftBoundaryDataSize % sizeof(GroupElement) != 0)
    {
        return GROTTO_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }
    std::size_t intervalNum = leftBoundaryDataSize / sizeof(GroupElement);
    if (sharedBooleanOutDataSize !=
        sizeof(GroupElement) * elementNum * intervalNum)
    {
        return GROTTO_INVALID_SHARED_OUT_DATA_SIZE_ERROR;
    }

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return GROTTO_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return GROTTO_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = grottoGetKeyDataSize<GroupElement>( //
        bitWidthIn, elementNum                                        //
    );                                                                //
    if (keyDataSize != needKeyDataSize)
    {
        return GROTTO_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return GROTTO_INVALID_PARTY_ID_ERROR;
    }

    GroupElement*       sharedBooleanOutPtr = (GroupElement*)sharedBooleanOut;
    const GroupElement* maskedXPtr          = (const GroupElement*)maskedX;
    const GroupElement* leftBoundaryPtr     = (const GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (const GroupElement*)rightBoundary;
    const std::uint8_t* seedPtr          = (const std::uint8_t*)seed;

    GrottoKey<GroupElement> grottoKey;
    for (std::size_t i = 0; i < elementNum; i++)
    {
        std::size_t offsetSharedOut = i * intervalNum;
        std::size_t offsetMaskedX   = i;
        std::size_t offsetSeed      = 16 * i;

        grottoKeySetPtr<GroupElement>(grottoKey, key, bitWidthIn, i,
                                      elementNum);
        grottoMICEval<GroupElement>(
            sharedBooleanOutPtr + offsetSharedOut, maskedXPtr[offsetMaskedX],
            grottoKey, seedPtr + offsetSeed, partyId, leftBoundaryPtr,
            rightBoundaryPtr, intervalNum, bitWidthIn);
    }
    return GROTTO_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cpu_grottoIntervalLutEval( //
    void*       sharedOutE,
    void*       sharedOutT,
    const void* maskedX,
    size_t      maskedXDataSize,
    const void* key,
    size_t      keyDataSize,
    const void* seed,
    size_t      seedDataSize,
    int         partyId,
    const void* leftBoundary,
    size_t      leftBoundaryDataSize,
    const void* rightBoundary,
    size_t      rightBoundaryDataSize,
    const void* lookUpTable,
    size_t      lookUpTableDataSize,
    size_t      bitWidthIn,
    size_t      bitWidthOut,
    size_t      elementSize,
    size_t      elementNum) noexcept
{
    assert(sizeof(GroupElement) == elementSize);
    if (leftBoundaryDataSize != rightBoundaryDataSize ||
        lookUpTableDataSize != rightBoundaryDataSize ||
        leftBoundaryDataSize % sizeof(GroupElement) != 0)
    {
        return GROTTO_INVALID_BOUNDARY_DATA_SIZE_ERROR;
    }
    std::size_t intervalNum = leftBoundaryDataSize / sizeof(GroupElement);

    if (maskedXDataSize != sizeof(GroupElement) * elementNum)
    {
        return GROTTO_INVLIAD_MASKED_X_DATA_SIZE_ERROR;
    }
    if (seedDataSize != 16 * elementNum)
    {
        return GROTTO_INVALID_SEED_DATA_SIZE_ERROR;
    }
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6 &&
          bitWidthOut <= elementSize * 8))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    std::size_t needKeyDataSize = grottoGetKeyDataSize<GroupElement>( //
        bitWidthIn, elementNum                                        //
    );                                                                //
    if (keyDataSize != needKeyDataSize)
    {
        return GROTTO_INVALID_KEY_DATA_SIZE_ERROR;
    }
    if (!(partyId == 0 || partyId == 1))
    {
        return GROTTO_INVALID_PARTY_ID_ERROR;
    }

    GroupElement*       sharedOutEPtr    = (GroupElement*)sharedOutE;
    GroupElement*       sharedOutTPtr    = (GroupElement*)sharedOutT;
    const GroupElement* maskedXPtr       = (const GroupElement*)maskedX;
    const GroupElement* leftBoundaryPtr  = (const GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (const GroupElement*)rightBoundary;
    const GroupElement* lookUpTablePtr   = (const GroupElement*)lookUpTable;
    const std::uint8_t* seedPtr          = (const std::uint8_t*)seed;

    GrottoKey<GroupElement> grottoKey;
    for (std::size_t i = 0; i < elementNum; i++)
    {
        std::size_t offsetSharedOut = i;
        std::size_t offsetMaskedX   = i;
        std::size_t offsetSeed      = 16 * i;

        grottoKeySetPtr<GroupElement>(grottoKey, key, bitWidthIn, i,
                                      elementNum);
        grottoIntervalLutEval<GroupElement>(
            sharedOutEPtr + offsetSharedOut, sharedOutTPtr + offsetSharedOut,
            maskedXPtr[offsetMaskedX], grottoKey, seedPtr + offsetSeed, partyId,
            leftBoundaryPtr, rightBoundaryPtr, lookUpTablePtr, intervalNum,
            bitWidthIn);
    }
    return GROTTO_SUCCESS;
}

template <typename GroupElement>
static int IMPL_FastFss_cpu_grottoKeyZip(void**      zippedKey,
                                         size_t*     zippedKeyDataSize,
                                         const void* key,
                                         size_t      keyDataSize,
                                         size_t      bitWidthIn,
                                         size_t      elementSize,
                                         size_t      elementNum)
{
    return GROTTO_RUNTIME_ERROR;
}

template <typename GroupElement>
static int IMPL_FastFss_cpu_grottoKeyUnzip(void**      key,
                                           size_t*     keyDataSize,
                                           const void* zippedKey,
                                           size_t      zippedKeyDataSize,
                                           size_t      bitWidthIn,
                                           size_t      elementSize,
                                           size_t      elementNum)
{
    return GROTTO_RUNTIME_ERROR;
}

}; // namespace FastFss::cpu

int FastFss_cpu_grottoKeyGen(void**      key,
                             size_t*     keyDataSize,
                             const void* alpha,
                             size_t      alphaDataSize,
                             const void* seed0,
                             size_t      seedDataSize0,
                             const void* seed1,
                             size_t      seedDataSize1,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyGen<std::uint8_t>(
                key, keyDataSize, alpha, alphaDataSize, seed0, seedDataSize0,
                seed1, seedDataSize1, bitWidthIn, elementSize, elementNum);
        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyGen<std::uint16_t>(
                key, keyDataSize, alpha, alphaDataSize, seed0, seedDataSize0,
                seed1, seedDataSize1, bitWidthIn, elementSize, elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyGen<std::uint32_t>(
                key, keyDataSize, alpha, alphaDataSize, seed0, seedDataSize0,
                seed1, seedDataSize1, bitWidthIn, elementSize, elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyGen<std::uint64_t>(
                key, keyDataSize, alpha, alphaDataSize, seed0, seedDataSize0,
                seed1, seedDataSize1, bitWidthIn, elementSize, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_grottoEval(void*       sharedBooleanOut,
                           const void* maskedX,
                           size_t      maskedXDataSize,
                           const void* key,
                           size_t      keyDataSize,
                           const void* seed,
                           size_t      seedDataSize,
                           bool        equalBound,
                           int         partyId,
                           size_t      bitWidthIn,
                           size_t      elementSize,
                           size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoEval<std::uint8_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, equalBound, partyId, bitWidthIn,
                elementSize, elementNum);
        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoEval<std::uint16_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, equalBound, partyId, bitWidthIn,
                elementSize, elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoEval<std::uint32_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, equalBound, partyId, bitWidthIn,
                elementSize, elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoEval<std::uint64_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, equalBound, partyId, bitWidthIn,
                elementSize, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_grottoEvalEq(void*       sharedBooleanOut,
                             const void* maskedX,
                             size_t      maskedXDataSize,
                             const void* key,
                             size_t      keyDataSize,
                             const void* seed,
                             size_t      seedDataSize,
                             int         partyId,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoEvalEq<std::uint8_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, partyId, bitWidthIn, elementSize,
                elementNum);
        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoEvalEq<std::uint16_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, partyId, bitWidthIn, elementSize,
                elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoEvalEq<std::uint32_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, partyId, bitWidthIn, elementSize,
                elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoEvalEq<std::uint64_t>(
                sharedBooleanOut, maskedX, maskedXDataSize, key, keyDataSize,
                seed, seedDataSize, partyId, bitWidthIn, elementSize,
                elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_grottoMICEval(void*       sharedBooleanOut,
                              size_t      sharedBooleanOutDataSize,
                              const void* maskedX,
                              size_t      maskedXDataSize,
                              const void* key,
                              size_t      keyDataSize,
                              const void* seed,
                              size_t      seedDataSize,
                              int         partyId,
                              const void* leftBoundary,
                              size_t      leftBoundaryDataSize,
                              const void* rightBoundary,
                              size_t      rightBoundaryDataSize,
                              size_t      bitWidthIn,
                              size_t      elementSize,
                              size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoMICEval<std::uint8_t>(
                sharedBooleanOut, sharedBooleanOutDataSize, maskedX,
                maskedXDataSize, key, keyDataSize, seed, seedDataSize, partyId,
                leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, elementSize, elementNum);
        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoMICEval<std::uint16_t>(
                sharedBooleanOut, sharedBooleanOutDataSize, maskedX,
                maskedXDataSize, key, keyDataSize, seed, seedDataSize, partyId,
                leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, elementSize, elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoMICEval<std::uint32_t>(
                sharedBooleanOut, sharedBooleanOutDataSize, maskedX,
                maskedXDataSize, key, keyDataSize, seed, seedDataSize, partyId,
                leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, elementSize, elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoMICEval<std::uint64_t>(
                sharedBooleanOut, sharedBooleanOutDataSize, maskedX,
                maskedXDataSize, key, keyDataSize, seed, seedDataSize, partyId,
                leftBoundary, leftBoundaryDataSize, rightBoundary,
                rightBoundaryDataSize, bitWidthIn, elementSize, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_grottoIntervalLutEval(void*       sharedOutE,
                                      void*       sharedOutT,
                                      const void* maskedX,
                                      size_t      maskedXDataSize,
                                      const void* key,
                                      size_t      keyDataSize,
                                      const void* seed,
                                      size_t      seedDataSize,
                                      int         partyId,
                                      const void* leftBoundary,
                                      size_t      leftBoundaryDataSize,
                                      const void* rightBoundary,
                                      size_t      rightBoundaryDataSize,
                                      const void* lookUpTable,
                                      size_t      lookUpTableDataSize,
                                      size_t      bitWidthIn,
                                      size_t      bitWidthOut,
                                      size_t      elementSize,
                                      size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoIntervalLutEval<
                std::uint8_t>(sharedOutE, sharedOutT, maskedX, maskedXDataSize,
                              key, keyDataSize, seed, seedDataSize, partyId,
                              leftBoundary, leftBoundaryDataSize, rightBoundary,
                              rightBoundaryDataSize, lookUpTable,
                              lookUpTableDataSize, bitWidthIn, bitWidthOut,
                              elementSize, elementNum);

        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoIntervalLutEval<
                std::uint16_t>(sharedOutE, sharedOutT, maskedX, maskedXDataSize,
                               key, keyDataSize, seed, seedDataSize, partyId,
                               leftBoundary, leftBoundaryDataSize,
                               rightBoundary, rightBoundaryDataSize,
                               lookUpTable, lookUpTableDataSize, bitWidthIn,
                               bitWidthOut, elementSize, elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoIntervalLutEval<
                std::uint32_t>(sharedOutE, sharedOutT, maskedX, maskedXDataSize,
                               key, keyDataSize, seed, seedDataSize, partyId,
                               leftBoundary, leftBoundaryDataSize,
                               rightBoundary, rightBoundaryDataSize,
                               lookUpTable, lookUpTableDataSize, bitWidthIn,
                               bitWidthOut, elementSize, elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoIntervalLutEval<
                std::uint64_t>(sharedOutE, sharedOutT, maskedX, maskedXDataSize,
                               key, keyDataSize, seed, seedDataSize, partyId,
                               leftBoundary, leftBoundaryDataSize,
                               rightBoundary, rightBoundaryDataSize,
                               lookUpTable, lookUpTableDataSize, bitWidthIn,
                               bitWidthOut, elementSize, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_grottoKeyZip(void**      zippedKey,
                             size_t*     zippedKeyDataSize,
                             const void* key,
                             size_t      keyDataSize,
                             size_t      bitWidthIn,
                             size_t      elementSize,
                             size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyZip<std::uint8_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                elementSize, elementNum);
        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyZip<std::uint16_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                elementSize, elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyZip<std::uint32_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                elementSize, elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyZip<std::uint64_t>(
                zippedKey, zippedKeyDataSize, key, keyDataSize, bitWidthIn,
                elementSize, elementNum);

        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_grottoKeyUnzip(void**      key,
                               size_t*     keyDataSize,
                               const void* zippedKey,
                               size_t      zippedKeyDataSize,
                               size_t      bitWidthIn,
                               size_t      elementSize,
                               size_t      elementNum)
{
    switch (elementSize)
    {
        case 1:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyUnzip<std::uint8_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                elementSize, elementNum);
        case 2:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyUnzip<std::uint16_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                elementSize, elementNum);
        case 4:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyUnzip<std::uint32_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                elementSize, elementNum);
        case 8:
            return FastFss::cpu::IMPL_FastFss_cpu_grottoKeyUnzip<std::uint64_t>(
                key, keyDataSize, zippedKey, zippedKeyDataSize, bitWidthIn,
                elementSize, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_grottoGetKeyDataSize(size_t bitWidthIn,
                                     size_t elementSize,
                                     size_t elementNum)
{
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    switch (elementSize)
    {
        case 1:
            return (int)FastFss::cpu::grottoGetKeyDataSize<std::uint8_t>(
                bitWidthIn, elementNum);
        case 2:
            return (int)FastFss::cpu::grottoGetKeyDataSize<std::uint16_t>(
                bitWidthIn, elementNum);
        case 4:
            return (int)FastFss::cpu::grottoGetKeyDataSize<std::uint32_t>(
                bitWidthIn, elementNum);
        case 8:
            return (int)FastFss::cpu::grottoGetKeyDataSize<std::uint64_t>(
                bitWidthIn, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}

int FastFss_cpu_grottoGetZippedKeyDataSize(size_t bitWidthIn,
                                           size_t elementSize,
                                           size_t elementNum)
{
    if (!(bitWidthIn <= elementSize * 8 && bitWidthIn > 6))
    {
        return GROTTO_INVALID_BITWIDTH_ERROR;
    }
    switch (elementSize)
    {
        case 1:
            return (int)FastFss::cpu::grottoGetZippedKeyDataSize<std::uint8_t>(
                bitWidthIn, elementNum);
        case 2:
            return (int)FastFss::cpu::grottoGetZippedKeyDataSize<std::uint16_t>(
                bitWidthIn, elementNum);
        case 4:
            return (int)FastFss::cpu::grottoGetZippedKeyDataSize<std::uint32_t>(
                bitWidthIn, elementNum);
        case 8:
            return (int)FastFss::cpu::grottoGetZippedKeyDataSize<std::uint64_t>(
                bitWidthIn, elementNum);
        default: return GROTTO_INVALID_ELEMENT_SIZE_ERROR;
    }
}