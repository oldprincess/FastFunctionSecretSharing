#pragma once
#ifndef SRC_IMPL_DPF_H
#define SRC_IMPL_DPF_H

#include "def.h"
#include "number.h"

#if !defined(AES_IMPL)
#include "aes.h"
#endif

namespace FastFss::impl {

// Function secret sharing: Improvements and extensions
template <typename GroupElement>
struct DpfKey
{
    std::uint64_t (*sCW)[2] = nullptr; // 127 bits, [1: 127]. bitWidthIn length
    GroupElement* tLCW      = nullptr; // bitWidthIn bits; 1 length
    GroupElement* tRCW      = nullptr; // bitWidthIn bits; 1 length
    GroupElement* lastCW    = nullptr; // bitWidthOut bits; 1 length
};

template <typename GroupElement>
struct DpfCache
{
    // S: 127 bits, [1: 127]. bitWidthIn length
    // t: 1   bits, [0]     . bitWidthIn length
    std::uint64_t (*stCache)[2] = nullptr;

    GroupElement  preMaskedX;
    std::uint16_t preTo; // 16bit preTo
};

template <typename GroupElement>
static inline std::size_t dpfGetKeyDataSize(std::size_t bitWidthIn,
                                            std::size_t bitWidthOut,
                                            size_t      elementNum) noexcept
{
    return elementNum * (16 * bitWidthIn +      //
                         sizeof(GroupElement) + //
                         sizeof(GroupElement) + //
                         sizeof(GroupElement)   //
                        );
}

template <typename GroupElement>
inline std::size_t dpfGetZippedKeyDataSize(std::size_t bitWidthIn,
                                           std::size_t bitWidthOut,
                                           size_t      elementNum) noexcept
{
    return 0;
}

template <typename GroupElement>
static inline std::size_t dpfGetCacheDataSize(std::size_t bitWidthIn,
                                              std::size_t elementNum) noexcept
{
    return elementNum * (16 * bitWidthIn);
}

template <typename GroupElement>
FAST_FSS_DEVICE static inline GroupElement dpfConvert(const std::uint64_t s[2],
                                                      int bitWidthOut) noexcept
{
    static_assert(sizeof(GroupElement) <= 16,
                  "GroupElement must be 128 bits or less");
    return *((GroupElement*)s);
}

template <typename GroupElement>
FAST_FSS_DEVICE static inline void dpfKeySetPtr(DpfKey<GroupElement>& dpfKey,
                                                const void*           keyData,
                                                std::size_t bitWidthIn,
                                                std::size_t bitWidthOut,
                                                std::size_t idx,
                                                std::size_t elementNum) noexcept
{
    const char* curKeyData = nullptr;

    std::size_t offsetSCW = idx * (16 * bitWidthIn);
    std::size_t offsetVCW =
        elementNum * (16 * bitWidthIn) + idx * (sizeof(GroupElement) + //
                                                sizeof(GroupElement) + //
                                                sizeof(GroupElement)   //
                                               );

    curKeyData  = (const char*)keyData + offsetSCW;
    dpfKey.sCW  = (std::uint64_t (*)[2])(curKeyData);
    curKeyData  = (const char*)keyData + offsetVCW;
    dpfKey.tLCW = (GroupElement*)(curKeyData);
    curKeyData += sizeof(GroupElement);
    dpfKey.tRCW = (GroupElement*)(curKeyData);
    curKeyData += sizeof(GroupElement);
    dpfKey.lastCW = (GroupElement*)(curKeyData);
    curKeyData += sizeof(GroupElement);
}

template <typename GroupElement>
FAST_FSS_DEVICE static inline void dpfCacheSetPtr(
    DpfCache<GroupElement>& dpfCache,
    const void*             cacheData,
    std::size_t             bitWidthIn,
    std::size_t             bitWidthOut,
    std::size_t             idx,
    std::size_t             elementNum) noexcept
{
    const char* curCacheData = (const char*)cacheData;
    std::size_t offsetSCW    = idx * (16 * bitWidthIn);
    dpfCache.stCache         = (std::uint64_t (*)[2])(curCacheData + offsetSCW);
    dpfCache.preMaskedX      = 0;
    dpfCache.preTo           = 0;
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void dpfKeyGen(
    DpfKey<GroupElement>&      key,
    GroupElement               alpha,
    GroupElement               beta,
    const void*                seed0,
    const void*                seed1,
    std::size_t                bitWidthIn,
    std::size_t                bitWidthOut,
    const AES128GlobalContext* aesCtx = nullptr) noexcept
{
    constexpr std::uint64_t    MASK_MSB63   = 0xFFFF'FFFF'FFFF'FFFEULL;
    static const std::uint64_t PLAINTEXT[4] = {0, 1, 2, 3};

    key.tLCW[0] = 0;
    key.tRCW[0] = 0;

    // copy seed
    std::uint64_t curS0[2] = {
        ((const std::uint64_t*)seed0)[0] & MASK_MSB63,
        ((const std::uint64_t*)seed0)[1],
    };
    std::uint64_t curS1[2] = {
        ((const std::uint64_t*)seed1)[0] & MASK_MSB63,
        ((const std::uint64_t*)seed1)[1],
    };
    int curT0 = 0, curT1 = 1;

    std::uint64_t sL0tL0sR0tR0[4];
    std::uint64_t sL1tL1sR1tR1[4];
    for (std::size_t i = 0; i < bitWidthIn; i++)
    {
        // sL0, tL0, sR0, tR0 <- G(s0)
        // sL1, tL1, sR1, tR1 <- G(s1)
        AES128::aes128_enc2_block(sL0tL0sR0tR0, PLAINTEXT, curS0, aesCtx);
        AES128::aes128_enc2_block(sL1tL1sR1tR1, PLAINTEXT, curS1, aesCtx);

        std::uint64_t* sL0 = sL0tL0sR0tR0 + 0;
        std::uint64_t* sR0 = sL0tL0sR0tR0 + 2;

        std::uint64_t* sL1 = sL1tL1sR1tR1 + 0;
        std::uint64_t* sR1 = sL1tL1sR1tR1 + 2;

        int tL0 = sL0[0] & 1;
        int tR0 = sR0[0] & 1;
        sL0[0] &= MASK_MSB63;
        sR0[0] &= MASK_MSB63;

        int tL1 = sL1[0] & 1;
        int tR1 = sR1[0] & 1;
        sL1[0] &= MASK_MSB63;
        sR1[0] &= MASK_MSB63;

        // if alphaI = 0,   keep <- L, lose <- R
        // else             keep <- R, lose <- L
        //                  1 means R, 0 means L
        int alphaI = (alpha >> (bitWidthIn - i - 1)) & 1;
        int keep   = alphaI;
        int lose   = 1 - alphaI;

        std::uint64_t *sLose0 = nullptr, *sLose1 = nullptr;
        std::uint64_t *sKeep0 = nullptr, *sKeep1 = nullptr;
        if (lose == 0) // lose = L, keep = R
        {
            sLose0 = sL0, sLose1 = sL1;
            sKeep0 = sR0, sKeep1 = sR1;
        }
        else // lose = R, keep = L
        {
            sLose0 = sR0, sLose1 = sR1;
            sKeep0 = sL0, sKeep1 = sL1;
        }

        // sCW <- sLose0 xor sLose1
        std::uint64_t sCW[2] = {
            sLose0[0] ^ sLose1[0],
            sLose0[1] ^ sLose1[1],
        };

        GroupElement tLCW = tL0 ^ tL1 ^ alphaI ^ 1;
        GroupElement tRCW = tR0 ^ tR1 ^ alphaI;
        // CW(i) <- sCW || tLCW || tRCW
        key.sCW[i][0] = sCW[0];
        key.sCW[i][1] = sCW[1];
        key.tLCW[0]   = (tLCW << i) | (key.tLCW[0]);
        key.tRCW[0]   = (tRCW << i) | (key.tRCW[0]);

        // s(b) <- sKeep(b) ^ t(b) * sCW
        // t(b) <- tKeep(b) ^ t(b) * tKeepCW
        curS0[0] = (curT0 == 0) ? sKeep0[0] : (sKeep0[0] ^ sCW[0]);
        curS0[1] = (curT0 == 0) ? sKeep0[1] : (sKeep0[1] ^ sCW[1]);
        curS1[0] = (curT1 == 0) ? sKeep1[0] : (sKeep1[0] ^ sCW[0]);
        curS1[1] = (curT1 == 0) ? sKeep1[1] : (sKeep1[1] ^ sCW[1]);
        if (keep == 0) // keep = L
        {
            curT0 = (tLCW == 1) ? (tL0 ^ curT0) : tL0;
            curT1 = (tLCW == 1) ? (tL1 ^ curT1) : tL1;
        }
        else
        {
            curT0 = (tRCW == 1) ? (tR0 ^ curT0) : tR0;
            curT1 = (tRCW == 1) ? (tR1 ^ curT1) : tR1;
        }
    }
    // CW(n+1) <- (-1)^t1 * (beta - convert(s0) + convert(s1))
    key.lastCW[0] = ((GroupElement)(-2) * curT1 + 1) *
                    (beta - dpfConvert<GroupElement>(curS0, bitWidthOut) +
                     dpfConvert<GroupElement>(curS1, bitWidthOut));
}

template <typename GroupElement>
FAST_FSS_DEVICE inline GroupElement dpfEval(
    const DpfKey<GroupElement>& key,
    GroupElement                maskedX,
    const void*                 seed,
    int                         partyId,
    std::size_t                 bitWidthIn,
    std::size_t                 bitWidthOut,
    DpfCache<GroupElement>*     cache  = nullptr,
    const AES128GlobalContext*  aesCtx = nullptr) noexcept
{
    constexpr std::uint64_t    MASK_MSB63   = 0xFFFF'FFFF'FFFF'FFFEULL;
    static const std::uint64_t PLAINTEXT[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    std::uint64_t curS[2] = {
        ((const std::uint64_t*)seed)[0] & MASK_MSB63,
        ((const std::uint64_t*)seed)[1],
    };
    int curT = partyId;

    maskedX = modBits<GroupElement>(maskedX, bitWidthIn);

    std::size_t idx_from = 0;
    if (cache != nullptr)
    {
        idx_from = clz<GroupElement>(maskedX ^ cache->preMaskedX, bitWidthIn);
        idx_from = (idx_from < cache->preTo) ? idx_from : cache->preTo;
        if (0 < idx_from && idx_from <= bitWidthIn)
        {
            curT    = cache->stCache[idx_from - 1][0] & 1;
            curS[0] = cache->stCache[idx_from - 1][0] & MASK_MSB63;
            curS[1] = cache->stCache[idx_from - 1][1];
        }
    }

    std::uint64_t sLtLsRtR[4];
    for (std::size_t i = idx_from; i < bitWidthIn; i++)
    {
        AES128::aes128_enc2_block(sLtLsRtR, PLAINTEXT, curS, aesCtx);

        std::uint64_t* sL = sLtLsRtR + 0;
        std::uint64_t* sR = sLtLsRtR + 2;

        int tL = sL[0] & 1;
        int tR = sR[0] & 1;
        sL[0] &= MASK_MSB63;
        sR[0] &= MASK_MSB63;

        //
        if (curT)
        {
            sL[0] ^= key.sCW[i][0], sL[1] ^= key.sCW[i][1];
            tL ^= (key.tLCW[0] >> i) & 1;
            sR[0] ^= key.sCW[i][0], sR[1] ^= key.sCW[i][1];
            tR ^= (key.tRCW[0] >> i) & 1;
        }
        //
        int xI = (maskedX >> (bitWidthIn - i - 1)) & 1;

        if (xI == 0)
        {
            curS[0] = sL[0], curS[1] = sL[1];
            curT = tL;
        }
        else
        {
            curS[0] = sR[0], curS[1] = sR[1];
            curT = tR;
        }

        if (cache != nullptr)
        {
            cache->stCache[i][0] = curS[0] & MASK_MSB63;
            cache->stCache[i][1] = curS[1];
            cache->stCache[i][0] ^= curT;

            cache->preTo      = i + 1;
            cache->preMaskedX = maskedX;
        }
    }
    return ((GroupElement)(-2) * partyId + 1) *
           (dpfConvert<GroupElement>(curS, bitWidthOut) + curT * key.lastCW[0]);
}

} // namespace FastFss::impl

#endif