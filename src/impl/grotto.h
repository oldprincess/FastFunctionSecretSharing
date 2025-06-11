#ifndef SRC_IMPL_GROTTO_H
#define SRC_IMPL_GROTTO_H

#include "def.h"
#include "number.h"

#if !defined(AES_IMPL)
#include "aes.h"
#define AES_IMPL
#endif

namespace FastFss::impl {

template <typename GroupElement>
struct GrottoKey
{
    // 127 bits, [1: 127]. bitWidthIn - 6 length
    std::uint64_t (*sCW)[2] = nullptr;
    std::uint64_t* lastCW   = nullptr; // 64 bits; 1 length
    GroupElement*  tLCW     = nullptr; // bitWidthIn bits; 1 length
    GroupElement*  tRCW     = nullptr; // bitWidthIn bits; 1 length
};

template <typename GroupElement>
struct GrottoCache
{
    // S: 127 bits, [1: 127]. bitWidthIn - 6 length
    // t: 1   bits, [0]     . bitWidthIn - 6 length
    std::uint64_t (*stCache)[2] = nullptr;

    std::int8_t* parityCache = nullptr; // bitWidthIn - 6 length

    GroupElement  preMaskedX;
    std::uint16_t preTo; // 16bit preTo
};

template <typename GroupElement>
static inline std::size_t grottoGetKeyDataSize(std::size_t bitWidthIn,
                                               std::size_t elementNum) noexcept
{
    return elementNum * (16 * (bitWidthIn - 6) + //
                         sizeof(std::uint64_t) + //
                         sizeof(GroupElement) +  //
                         sizeof(GroupElement)    //
                        );
}

template <typename GroupElement>
static inline std::size_t grottoGetZippedKeyDataSize(
    std::size_t bitWidthIn,
    std::size_t elementNum) noexcept
{
    return (std::size_t)(-1);
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void grottoKeySetPtr(GrottoKey<GroupElement>& grottoKey,
                                            const void*              keyData,
                                            std::size_t              bitWidthIn,
                                            std::size_t              idx,
                                            std::size_t elementNum) noexcept
{
    const char* curKeyData = nullptr;

    std::size_t offsetSCW    = idx * (16 * (bitWidthIn - 6));         //
    std::size_t offsetLastCW = elementNum * (16 * (bitWidthIn - 6)) + //
                               idx * 8;                               //
    std::size_t offsetLRtCW = elementNum * (16 * (bitWidthIn - 6)) +  //
                              elementNum * 8 +                        //
                              idx * sizeof(GroupElement) * 2;         //

    curKeyData       = (const char*)keyData + offsetSCW;
    grottoKey.sCW    = (std::uint64_t (*)[2])(curKeyData);
    curKeyData       = (const char*)keyData + offsetLastCW;
    grottoKey.lastCW = (std::uint64_t*)(curKeyData);

    curKeyData     = (const char*)keyData + offsetLRtCW;
    grottoKey.tLCW = (GroupElement*)(curKeyData);
    curKeyData += sizeof(GroupElement);
    grottoKey.tRCW = (GroupElement*)(curKeyData);
    curKeyData += sizeof(GroupElement);
}

template <typename GroupElement>
static inline std::size_t grottoGetCacheDataSize(
    std::size_t bitWidthIn,
    std::size_t elementNum) noexcept
{
    return elementNum * (16 * (bitWidthIn - 6) + (bitWidthIn - 6));
}

template <typename GroupElement>
FAST_FSS_DEVICE static inline void grottoCacheSetPtr( //
    GrottoCache<GroupElement>& cache,
    const void*                cacheData,
    std::size_t                bitWidthIn,
    std::size_t                idx,
    std::size_t                elementNum) noexcept
{
    const char* curCacheData = nullptr;

    std::size_t offsetSTCache     = idx * (16 * (bitWidthIn - 6));
    std::size_t offsetParityCache = elementNum * (16 * (bitWidthIn - 6)) + //
                                    (bitWidthIn - 6) * idx;

    curCacheData  = (const char*)cacheData + offsetSTCache;
    cache.stCache = (std::uint64_t (*)[2])(curCacheData);

    curCacheData      = (const char*)cacheData + offsetParityCache;
    cache.parityCache = (std::int8_t*)curCacheData;

    cache.preMaskedX = 0;
    cache.preTo      = 0;
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void grottoKeyGen(GrottoKey<GroupElement>& key,
                                         GroupElement             alpha,
                                         const void*              seed0,
                                         const void*              seed1,
                                         std::size_t bitWidthIn) noexcept
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
    GroupElement curT0 = 0, curT1 = 1;

    std::uint64_t sL0tL0sR0tR0[4];
    std::uint64_t sL1tL1sR1tR1[4];
    for (std::size_t i = 0; i < bitWidthIn - 6; i++)
    {
        // sL0, tL0, sR0, tR0 <- G(s0)
        // sL1, tL1, sR1, tR1 <- G(s1)
        // bits [127: 1] is seed, bits [1: 0] is tag
        AES128::aes128_enc2_block(sL0tL0sR0tR0, PLAINTEXT, curS0);
        AES128::aes128_enc2_block(sL1tL1sR1tR1, PLAINTEXT, curS1);

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

        // sCW[i] = seed0[lose] ^ seed1[lose]
        key.sCW[i][0] = (lose == 0) ? sL0[0] ^ sL1[0] : sR0[0] ^ sR1[0];
        key.sCW[i][1] = (lose == 0) ? sL0[1] ^ sL1[1] : sR0[1] ^ sR1[1];

        int tLCW = tL0 ^ tL1 ^ lose;
        int tRCW = tR0 ^ tR1 ^ keep;

        key.tLCW[0] |= (GroupElement)tLCW << i;
        key.tRCW[0] |= (GroupElement)tRCW << i;

        // seed0    = seed0[keep]           if cur_t0 == 0
        //          = seed0[keep] ^ cw[i]   if cur_t0 == 1
        curS0[0] = (keep == 0) ? sL0[0] : sR0[0];
        curS0[1] = (keep == 0) ? sL0[1] : sR0[1];

        curS0[0] = (curT0 == 0) ? curS0[0] : curS0[0] ^ key.sCW[i][0];
        curS0[1] = (curT0 == 0) ? curS0[1] : curS0[1] ^ key.sCW[i][1];

        // seed1    = seed1[keep]           if cur_t1 == 0
        //          = seed1[keep] ^ cw[i]   if cur_t1 == 1
        curS1[0] = (keep == 0) ? sL1[0] : sR1[0];
        curS1[1] = (keep == 0) ? sL1[1] : sR1[1];

        curS1[0] = (curT1 == 0) ? curS1[0] : curS1[0] ^ key.sCW[i][0];
        curS1[1] = (curT1 == 0) ? curS1[1] : curS1[1] ^ key.sCW[i][1];

        // t0   =   ti0[keep]               if t0 == 0
        //      =   ti0[keep] ^ cw_t[keep]  if t0 == 1
        curT0 = (curT0 == 0) ? ((keep == 0) ? tL0 : tR0)
                             : ((keep == 0) ? tL0 ^ tLCW : tR0 ^ tRCW);

        // t1   =   ti1[keep]               if t1 == 0
        //      =   ti1[keep] ^ cw_t[keep]  if t1 == 1
        curT1 = (curT1 == 0) ? ((keep == 0) ? tL1 : tR1)
                             : ((keep == 0) ? tL1 ^ tLCW : tR1 ^ tRCW);
    }
    // leaf = s0 ^ s1 ^ (1 << alpha_lsb6)
    key.lastCW[0] = curS0[1] ^ curS1[1] ^ (1ULL << (int)(alpha & 0b111111));
}

template <typename GroupElement>
FAST_FSS_DEVICE inline GroupElement grottoEvalEq( //
    const GrottoKey<GroupElement>& key,
    GroupElement                   maskedX,
    const void*                    seed,
    int                            partyId,
    std::size_t                    bitWidthIn,
    GrottoCache<GroupElement>*     cache = nullptr) noexcept
{
    constexpr std::uint64_t    MASK_MSB63   = 0xFFFF'FFFF'FFFF'FFFEULL;
    static const std::uint64_t PLAINTEXT[4] = {0, 1, 2, 3};

    std::uint64_t curS[2] = {
        ((const std::uint64_t*)seed)[0] & MASK_MSB63,
        ((const std::uint64_t*)seed)[1],
    };
    int curT = partyId;

    std::size_t idx_from = 0;
    if (cache != nullptr)
    {
        idx_from = clz<GroupElement>((maskedX ^ cache->preMaskedX) | 0b111111,
                                     bitWidthIn);
        idx_from = (idx_from < cache->preTo) ? idx_from : cache->preTo;
        if (0 < idx_from && idx_from <= bitWidthIn - 6)
        {
            curT    = cache->stCache[idx_from - 1][0] & 1;
            curS[0] = cache->stCache[idx_from - 1][0] & MASK_MSB63;
            curS[1] = cache->stCache[idx_from - 1][1];
        }
    }

    std::uint64_t s[2];
    for (std::size_t i = idx_from; i < bitWidthIn - 6; i++)
    {
        int bitI = (maskedX >> (bitWidthIn - 1 - i)) & 1;
        AES128::aes128_enc1_block(s, PLAINTEXT + bitI * 2, curS);

        // separate s, t
        std::uint64_t* si;
        int            ti;

        si    = s;
        ti    = si[0] & 1;
        si[0] = si[0] & MASK_MSB63;

        // seed     = s             if t == 0
        //          = s ^ cw[i]     if t == 1
        curS[0] = (curT == 0) ? si[0] : si[0] ^ key.sCW[i][0];
        curS[1] = (curT == 0) ? si[1] : si[1] ^ key.sCW[i][1];

        // t    =   ti                  if  t == 0
        //      =   ti ^ cw_t[bit_i]    if  t == 1
        GroupElement cwT = (bitI == 0) ? key.tLCW[0] : key.tRCW[0];
        curT             = (curT == 0) ? ti : ti ^ ((int)(cwT >> i) & 1);

        if (cache != nullptr)
        {
            cache->stCache[i][0] = curS[0] & MASK_MSB63;
            cache->stCache[i][1] = curS[1];
            cache->stCache[i][0] ^= curT;

            cache->preTo      = i + 1;
            cache->preMaskedX = maskedX;
        }
    }
    std::uint64_t u = (curT == 0) ? curS[1] : curS[1] ^ (key.lastCW[0]);
    return (u >> (maskedX & 0b111111)) & 1;
}

template <typename GroupElement>
FAST_FSS_DEVICE inline GroupElement grottoEval( //
    const GrottoKey<GroupElement>& key,
    GroupElement                   maskedX,
    const void*                    seed,
    int                            partyId,
    std::size_t                    bitWidthIn,
    bool                           equalBound = false,
    GrottoCache<GroupElement>*     cache      = nullptr) noexcept
{
    constexpr std::uint64_t    MASK_MSB63   = 0xFFFF'FFFF'FFFF'FFFEULL;
    static const std::uint64_t PLAINTEXT[4] = {0, 1, 2, 3};

    maskedX = modBits<GroupElement>(maskedX, bitWidthIn);

    std::uint64_t curS[2] = {
        ((const std::uint64_t*)seed)[0] & MASK_MSB63,
        ((const std::uint64_t*)seed)[1],
    };
    int curT   = partyId;
    int parity = 0;

    std::size_t idx_from = 0;
    if (cache != nullptr)
    {
        idx_from = clz<GroupElement>((maskedX ^ cache->preMaskedX) | 0b111111,
                                     bitWidthIn);
        idx_from = (idx_from < cache->preTo) ? idx_from : cache->preTo;
        if (0 < idx_from && idx_from <= bitWidthIn - 6)
        {
            curT    = cache->stCache[idx_from - 1][0] & 1;
            curS[0] = cache->stCache[idx_from - 1][0] & MASK_MSB63;
            curS[1] = cache->stCache[idx_from - 1][1];
            parity  = cache->parityCache[idx_from - 1];
        }
    }

    std::uint64_t s[2];
    for (std::size_t i = idx_from; i < bitWidthIn - 6; i++)
    {
        if (equalBound)
        {
            // early drop
            GroupElement mask = ((GroupElement)(-1) >>
                                 (sizeof(GroupElement) * 8 - (bitWidthIn - i)));
            if ((maskedX & mask) == 0)
            {
                return parity ^ partyId;
            }
        }

        int maskedXI = (maskedX >> (bitWidthIn - 1 - i)) & 1;

        AES128::aes128_enc1_block(s, PLAINTEXT + 2 * maskedXI, curS);

        // separate si and ti
        std::uint64_t* si;
        int            ti;

        si    = s;
        ti    = si[0] & 1;
        si[0] = si[0] & MASK_MSB63;

        // path[i+1].s  =   si              if path[i].t == 0
        //              =   si ^ cw[i].s    if path[i].t == 1
        curS[0] = (curT == 0) ? si[0] : si[0] ^ key.sCW[i][0];
        curS[1] = (curT == 0) ? si[1] : si[1] ^ key.sCW[i][1];

        // path[i+1].t  =   ti                          if path[i].t == 0
        //              =   ti ^ cw[i].t[nxt_dir]       if path[i].t == 1
        GroupElement cwT  = (maskedXI == 0) ? key.tLCW[0] : key.tRCW[0];
        int          nxtT = (curT == 0) ? ti : ti ^ ((int)(cwT >> i) & 1);

        // nxt_parity = parity[i]       if nxt_dir == 0
        //              parity[i] ^ (path[i].t ^ path[i + 1].t)
        //                              if nxt_dir == 1
        parity = (maskedXI == 0) ? parity : parity ^ (curT ^ nxtT);

        curT = nxtT;

        if (cache != nullptr)
        {
            cache->stCache[i][0] = curS[0] & MASK_MSB63;
            cache->stCache[i][1] = curS[1];
            cache->stCache[i][0] ^= curT;
            cache->parityCache[i] = parity;

            cache->preTo      = i + 1;
            cache->preMaskedX = maskedX;
        }
    }

    std::uint64_t u         = (curT == 0) ? curS[1] : curS[1] ^ key.lastCW[0];
    std::uint8_t  prefixLen = (std::uint8_t)maskedX & 0b111111;
    if (equalBound)
    {
        if (prefixLen != 0)
        {
            parity ^= parityU64(u & ((std::uint64_t)(-1) >> (64 - prefixLen)));
        }
    }
    else
    {
        parity ^= parityU64(u & ((std::uint64_t)(-1) >> (63 - prefixLen)));
    }
    return parity ^ partyId;
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void grottoMICEval(    //
    GroupElement*                  sharedOut, // intervalNum
    const GrottoKey<GroupElement>& key,
    GroupElement                   maskedX,
    const void*                    seed,
    int                            partyId,
    const GroupElement*            leftBoundary,
    const GroupElement*            rightBoundary,
    size_t                         intervalNum,
    std::size_t                    bitWidthIn,
    GrottoCache<GroupElement>*     cache = nullptr) noexcept
{
    if (intervalNum == 0)
    {
        return;
    }

    std::size_t  bitWidth = bitWidthIn;
    GroupElement sp = 0, sq = 0;
    {
        GroupElement xP = maskedX - (leftBoundary[0] - 1);
        GroupElement xQ = maskedX - rightBoundary[0];

        xP = modBits<GroupElement>(xP, bitWidth);
        xQ = modBits<GroupElement>(xQ, bitWidth);

        sp = grottoEval<GroupElement>(key, xP, seed, partyId, bitWidth, true,
                                      cache);
        sq = grottoEval<GroupElement>(key, xQ, seed, partyId, bitWidth, true,
                                      cache);

        sharedOut[0] = (sp ^ sq) ^ ((xQ > xP) ? partyId : 0);
    }
    for (std::size_t i = 1; i < intervalNum; i++)
    {
        GroupElement xP        = maskedX - (leftBoundary[i] - 1);
        GroupElement xQ        = maskedX - rightBoundary[i];
        GroupElement privQAdd1 = rightBoundary[i - 1] + 1;

        xP        = modBits<GroupElement>(xP, bitWidth);
        xQ        = modBits<GroupElement>(xQ, bitWidth);
        privQAdd1 = modBits<GroupElement>(privQAdd1, bitWidth);

        if (leftBoundary[i] == privQAdd1)
        {
            sp = sq;
        }
        else
        {
            sp = grottoEval<GroupElement>(key, xP, seed, partyId, bitWidth,
                                          true, cache);
        }
        sq = grottoEval<GroupElement>(key, xQ, seed, partyId, bitWidth, true,
                                      cache);

        sharedOut[i] = (sp ^ sq) ^ ((xQ > xP) ? partyId : 0);
    }
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void grottoIntervalLutEval( //
    GroupElement*                  sharedOutE,
    GroupElement*                  sharedOutT,
    const GrottoKey<GroupElement>& key,
    GroupElement                   maskedX,
    const void*                    seed,
    int                            partyId,
    const GroupElement*            leftBoundary,
    const GroupElement*            rightBoundary,
    const GroupElement*            lookUpTable,
    std::size_t                    lutNum,
    std::size_t                    intervalNum,
    std::size_t                    bitWidthIn,
    GrottoCache<GroupElement>*     cache = nullptr) noexcept
{
    if (intervalNum == 0)
    {
        return;
    }

    sharedOutE[0] = 0;
    for (std::size_t j = 0; j < lutNum; j++)
    {
        sharedOutT[j] = 0;
    }

    std::size_t  bitWidth = bitWidthIn;
    GroupElement sp = 0, sq = 0;
    {
        GroupElement xP = maskedX - (leftBoundary[0] - 1);
        GroupElement xQ = maskedX - rightBoundary[0];

        xP = modBits<GroupElement>(xP, bitWidth);
        xQ = modBits<GroupElement>(xQ, bitWidth);

        sp = grottoEval<GroupElement>(key, xP, seed, partyId, bitWidth, true,
                                      cache);
        sq = grottoEval<GroupElement>(key, xQ, seed, partyId, bitWidth, true,
                                      cache);

        GroupElement tmp = (sp ^ sq) ^ ((xQ > xP) ? partyId : 0);

        tmp &= 1;
        sharedOutE[0] += tmp;
        for (std::size_t j = 0; j < lutNum; j++)
        {
            sharedOutT[j] += tmp * lookUpTable[0 + j * intervalNum];
        }
    }
    for (std::size_t i = 1; i < intervalNum; i++)
    {
        GroupElement xP        = maskedX - (leftBoundary[i] - 1);
        GroupElement xQ        = maskedX - rightBoundary[i];
        GroupElement privQAdd1 = rightBoundary[i - 1] + 1;

        xP        = modBits<GroupElement>(xP, bitWidth);
        xQ        = modBits<GroupElement>(xQ, bitWidth);
        privQAdd1 = modBits<GroupElement>(privQAdd1, bitWidth);

        if (leftBoundary[i] == privQAdd1)
        {
            sp = sq;
        }
        else
        {
            sp = grottoEval<GroupElement>(key, xP, seed, partyId, bitWidth,
                                          true, cache);
        }
        sq = grottoEval<GroupElement>(key, xQ, seed, partyId, bitWidth, true,
                                      cache);

        GroupElement tmp = (sp ^ sq) ^ ((xQ > xP) ? partyId : 0);

        tmp &= 1;
        sharedOutE[0] += tmp;
        for (std::size_t j = 0; j < lutNum; j++)
        {
            sharedOutT[j] += tmp * lookUpTable[i + j * intervalNum];
        }
    }
    if (partyId)
    {
        sharedOutE[0] = (GroupElement)(-1) * sharedOutE[0];
        for (std::size_t j = 0; j < lutNum; j++)
        {
            sharedOutT[j] = (GroupElement)(-1) * sharedOutT[j];
        }
    }
    // E = 1 or -1.
    // E = ((E - 1) >> 1) & 1: 1(V need times -1) 0(V need not times -1)
    if (partyId == 0)
    {
        sharedOutE[0] -= 1;
        sharedOutE[0] = ((sharedOutE[0] >> 1) + (sharedOutE[0] & 1)) & 1;
    }
    else
    {
        sharedOutE[0] = (sharedOutE[0] >> 1) & 1;
    }
}

} // namespace FastFss::impl

#endif