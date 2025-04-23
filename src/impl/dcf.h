#pragma once
#ifndef SRC_IMPL_DCF_H
#define SRC_IMPL_DCF_H

#include "def.h"
#include "number.h"

#if !defined(AES_IMPL)
#include "aes.h"
#endif

namespace FastFss::impl {

// Distributed Comparison Function,
// Function secret sharing for mixed-mode and fixed-point secure computation
template <typename GroupElement>
struct DcfKey
{
    std::uint64_t (*sCW)[2] = nullptr; // 127 bits, [1: 127]. bitWidthIn length
    GroupElement* vCW       = nullptr; // bitWidthOut bits; bitWidthIn length
    GroupElement* tLCW      = nullptr; // bitWidthIn bits; 1 length
    GroupElement* tRCW      = nullptr; // bitWidthIn bits; 1 length
    GroupElement* lastCW    = nullptr; // bitWidthOut bits; 1 length
};

template <typename GroupElement>
static inline std::size_t dcfGetKeyDataSize(std::size_t bitWidthIn,
                                            std::size_t bitWidthOut,
                                            size_t      elementNum) noexcept
{
    return elementNum * (16 * bitWidthIn +                   //
                         sizeof(GroupElement) * bitWidthIn + //
                         sizeof(GroupElement) +              //
                         sizeof(GroupElement) +              //
                         sizeof(GroupElement)                //
                        );
}

template <typename GroupElement>
inline std::size_t dcfGetZippedKeyDataSize(std::size_t bitWidthIn,
                                           std::size_t bitWidthOut,
                                           size_t      elementNum) noexcept
{
    return -1;
}

template <typename GroupElement>
FAST_FSS_DEVICE static inline GroupElement convert(const std::uint64_t s[2],
                                                   int bitWidthOut) noexcept
{
    static_assert(sizeof(GroupElement) <= 8,
                  "GroupElement must be 64 bits or less");
    return *((GroupElement*)s);
}

template <typename GroupElement>
FAST_FSS_DEVICE static inline void dcfKeySetPtr(DcfKey<GroupElement>& dcfKey,
                                                const void*           keyData,
                                                std::size_t bitWidthIn,
                                                std::size_t bitWidthOut,
                                                std::size_t idx,
                                                std::size_t elementNum) noexcept
{
    const char* curKeyData = nullptr;

    std::size_t offsetSCW = idx * (16 * bitWidthIn);
    std::size_t offsetVCW = elementNum * (16 * bitWidthIn) +
                            idx * (sizeof(GroupElement) * bitWidthIn + //
                                   sizeof(GroupElement) +              //
                                   sizeof(GroupElement) +              //
                                   sizeof(GroupElement)                //
                                  );

    curKeyData = (const char*)keyData + offsetSCW;
    dcfKey.sCW = (std::uint64_t (*)[2])(curKeyData);
    curKeyData = (const char*)keyData + offsetVCW;
    dcfKey.vCW = (GroupElement*)(curKeyData);
    curKeyData += sizeof(GroupElement) * bitWidthIn;
    dcfKey.tLCW = (GroupElement*)(curKeyData);
    curKeyData += sizeof(GroupElement);
    dcfKey.tRCW = (GroupElement*)(curKeyData);
    curKeyData += sizeof(GroupElement);
    dcfKey.lastCW = (GroupElement*)(curKeyData);
    curKeyData += sizeof(GroupElement);
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void dcfKeyGen(DcfKey<GroupElement>& key,
                                      GroupElement          alpha,
                                      GroupElement          beta,
                                      const void*           seed0,
                                      const void*           seed1,
                                      std::size_t           bitWidthIn,
                                      std::size_t bitWidthOut) noexcept
{
    constexpr std::uint64_t    MASK_MSB63   = 0xFFFF'FFFF'FFFF'FFFEULL;
    static const std::uint64_t PLAINTEXT[8] = {0, 1, 2, 3, 4, 5, 6, 7};

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
    GroupElement vAlpha = 0, curT0 = 0, curT1 = 1;

    std::uint64_t sL0vL0sR0vR0[8];
    std::uint64_t sL1vL1sR1vR1[8];
    for (std::size_t i = 0; i < bitWidthIn; i++)
    {
        // sL0, tL0, vL0, sR0, tR0, vR0 <- G(s0)
        // sL1, tL1, vL1, sR1, tR1, vR1 <- G(s1)
        AES128::aes128_enc4_block(sL0vL0sR0vR0, PLAINTEXT, curS0);
        AES128::aes128_enc4_block(sL1vL1sR1vR1, PLAINTEXT, curS1);

        std::uint64_t* sL0 = sL0vL0sR0vR0 + 0;
        std::uint64_t* vL0 = sL0vL0sR0vR0 + 2;
        std::uint64_t* sR0 = sL0vL0sR0vR0 + 4;
        std::uint64_t* vR0 = sL0vL0sR0vR0 + 6;

        std::uint64_t* sL1 = sL1vL1sR1vR1 + 0;
        std::uint64_t* vL1 = sL1vL1sR1vR1 + 2;
        std::uint64_t* sR1 = sL1vL1sR1vR1 + 4;
        std::uint64_t* vR1 = sL1vL1sR1vR1 + 6;

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
        std::uint64_t *vLose0 = nullptr, *vLose1 = nullptr;
        std::uint64_t *vKeep0 = nullptr, *vKeep1 = nullptr;
        if (lose == 0) // lose = L, keep = R
        {
            sLose0 = sL0, sLose1 = sL1, vLose0 = vL0, vLose1 = vL1;
            sKeep0 = sR0, sKeep1 = sR1, vKeep0 = vR0, vKeep1 = vR1;
        }
        else // lose = R, keep = L
        {
            sLose0 = sR0, sLose1 = sR1, vLose0 = vR0, vLose1 = vR1;
            sKeep0 = sL0, sKeep1 = sL1, vKeep0 = vL0, vKeep1 = vL1;
        }

        // sCW <- sLose0 xor sLose1
        std::uint64_t sCW[2] = {
            sLose0[0] ^ sLose1[0],
            sLose0[1] ^ sLose1[1],
        };
        // vCW <- (-1)^t1 * ( convert(vLose1) - convert(vLose0) - vAlpha)
        GroupElement vCW =
            ((GroupElement)(-2) * curT1 + 1) *
            (convert<GroupElement>(vLose1, bitWidthOut) -
             convert<GroupElement>(vLose0, bitWidthOut) - vAlpha);
        // if Lose = L then vCW <- vCW + (-1)^t1 * beta
        if (lose == 0)
        {
            vCW = vCW + ((GroupElement)(-2) * curT1 + 1) * beta;
        }
        // vAlpha += -convert(vKeep1) + convert(vKeep0) + (-1)^t1 * vCW
        vAlpha = vAlpha - convert<GroupElement>(vKeep1, bitWidthOut) +
                 convert<GroupElement>(vKeep0, bitWidthOut) +
                 ((GroupElement)(-2) * curT1 + 1) * vCW;
        //
        GroupElement tLCW = tL0 ^ tL1 ^ alphaI ^ 1;
        GroupElement tRCW = tR0 ^ tR1 ^ alphaI;
        // CW(i) <- sCW || vCW || tLCW || tRCW
        key.sCW[i][0] = sCW[0];
        key.sCW[i][1] = sCW[1];
        key.vCW[i]    = vCW;
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
    // CW(n+1) <- (-1)^t1 * ( convert(s1) - convert(s0) - vAlpha )
    key.lastCW[0] = ((GroupElement)(-2) * curT1 + 1) *
                    (convert<GroupElement>(curS1, bitWidthOut) -
                     convert<GroupElement>(curS0, bitWidthOut) - vAlpha);
}

template <typename GroupElement>
FAST_FSS_DEVICE inline GroupElement dcfEval(const DcfKey<GroupElement>& key,
                                            GroupElement                maskedX,
                                            const void*                 seed,
                                            int                         partyId,
                                            std::size_t bitWidthIn,
                                            std::size_t bitWidthOut) noexcept
{
    constexpr std::uint64_t    MASK_MSB63   = 0xFFFF'FFFF'FFFF'FFFEULL;
    static const std::uint64_t PLAINTEXT[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    std::uint64_t curS[2] = {
        ((const std::uint64_t*)seed)[0] & MASK_MSB63,
        ((const std::uint64_t*)seed)[1],
    };
    int          curT = partyId;
    GroupElement v    = 0;

    std::uint64_t sLvLsRvR[8];
    for (std::size_t i = 0; i < bitWidthIn; i++)
    {
        AES128::aes128_enc4_block(sLvLsRvR, PLAINTEXT, curS);

        std::uint64_t* sL = sLvLsRvR + 0;
        std::uint64_t* vL = sLvLsRvR + 2;
        std::uint64_t* sR = sLvLsRvR + 4;
        std::uint64_t* vR = sLvLsRvR + 6;

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
            v += ((GroupElement)(-2) * partyId + 1) *
                 (convert<GroupElement>(vL, bitWidthOut) + curT * key.vCW[i]);
            curS[0] = sL[0], curS[1] = sL[1];
            curT = tL;
        }
        else
        {
            v += ((GroupElement)(-2) * partyId + 1) *
                 (convert<GroupElement>(vR, bitWidthOut) + curT * key.vCW[i]);
            curS[0] = sR[0], curS[1] = sR[1];
            curT = tR;
        }
    }
    v += ((GroupElement)(-2) * partyId + 1) *
         (convert<GroupElement>(curS, bitWidthOut) + curT * key.lastCW[0]);
    return v;
}

} // namespace FastFss::impl

#endif