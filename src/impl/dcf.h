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
    std::uint64_t (*sCW)[2]; // 127 bits, [1: 127]. bitWidthIn length
    GroupElement *vCW;       // bitWidthOut bits; bitWidthIn x groupSize length
    GroupElement *tLCW;      // bitWidthIn bits; 1 length
    GroupElement *tRCW;      // bitWidthIn bits; 1 length
    GroupElement *lastCW;    // bitWidthOut bits; groupSize length
};

template <typename GroupElement>
struct DcfCache
{
    // S: 127 bits, [1: 127]. bitWidthIn length
    // t: 1   bits, [0]     . bitWidthIn length
    std::uint64_t (*stCache)[2] = nullptr;

    GroupElement *v; // v: bitWidthOut bits; bitWidthIn x groupSize length

    GroupElement  preMaskedX;
    std::uint16_t preTo; // 16bit preTo
};

template <typename GroupElement>
static inline std::size_t dcfGetKeyDataSize(std::size_t bitWidthIn,
                                            std::size_t bitWidthOut,
                                            std::size_t groupSize,
                                            size_t      elementNum) noexcept
{
    return elementNum * (16 * bitWidthIn +                               //
                         sizeof(GroupElement) * bitWidthIn * groupSize + //
                         sizeof(GroupElement) +                          //
                         sizeof(GroupElement) +                          //
                         sizeof(GroupElement) * groupSize                //
                        );
}

template <typename GroupElement>
inline std::size_t dcfGetZippedKeyDataSize(std::size_t bitWidthIn,
                                           std::size_t bitWidthOut,
                                           std::size_t groupSize,
                                           std::size_t elementNum) noexcept
{
    return 0;
}

template <typename GroupElement>
static inline std::size_t dcfGetCacheDataSize(std::size_t bitWidthIn,
                                              std::size_t groupSize,
                                              std::size_t elementNum) noexcept
{
    return elementNum * (16 * bitWidthIn) +
           elementNum * sizeof(GroupElement) * bitWidthIn * groupSize;
}

template <typename GroupElement>
class DcfConvertCtx
{
public:
    FAST_FSS_DEVICE DcfConvertCtx(AES128 &aesCtx) : aesCtx_(aesCtx)
    {
    }

    FAST_FSS_DEVICE DcfConvertCtx(AES128                    &aesCtx,
                                  const void                *seed,
                                  std::size_t                groupSize,
                                  std::size_t                bitWidth,
                                  const AES128GlobalContext *aesGlobalCtx)
        : aesCtx_(aesCtx)
    {
        this->init(seed, groupSize, bitWidth, aesGlobalCtx);
    }

    FAST_FSS_DEVICE void init(const void                *seed,
                              std::size_t                groupSize,
                              std::size_t                bitWidth,
                              const AES128GlobalContext *aesGlobalCtx) noexcept
    {
        aesGlobalCtx_     = aesGlobalCtx;
        groupSize_        = (int)groupSize;
        groupElementSize_ = (int)((bitWidth + 7) / 8);
        if (groupElementSize_ * groupSize_ <= 16)
        {
            for (int i = 0; i < 16; i++)
            {
                buffer_[i] = ((const std::uint8_t *)seed)[i];
            }
        }
        else
        {
            aesCtx_.set_enc_key(seed, aesGlobalCtx_);
        }
        groupOffset_ = 0;
    }

    FAST_FSS_DEVICE GroupElement getNext()
    {
        GroupElement ret          = 0;
        int          bufferOffset = groupOffset_ * groupElementSize_;
        if (groupElementSize_ * groupSize_ <= 16)
        {
            for (int i = 0; i < groupElementSize_; i++)
            {
                int idx = (bufferOffset + i) % 16;
                ret |= (GroupElement)buffer_[idx] << (i * 8);
            }
        }
        else
        {
            for (int i = 0; i < groupElementSize_; i++)
            {
                int idx = (bufferOffset + i) % 16;
                if (idx == 0)
                {
                    ((std::uint64_t *)buffer_)[0] = 0;
                    ((std::uint64_t *)buffer_)[1] = (std::uint64_t)groupOffset_;
                    aesCtx_.enc_n_block<1>(buffer_, buffer_, aesGlobalCtx_);
                }
                ret |= (GroupElement)buffer_[idx] << (i * 8);
            }
        }
        groupOffset_ += 1;
        return ret;
    }

private:
    AES128                    &aesCtx_;
    const AES128GlobalContext *aesGlobalCtx_;
    std::uint8_t               buffer_[16];
    int                        groupOffset_;
    int                        groupElementSize_;
    int                        groupSize_;
};

template <typename GroupElement>
FAST_FSS_DEVICE static inline void dcfKeySetPtr(DcfKey<GroupElement> &dcfKey,
                                                const void           *keyData,
                                                std::size_t bitWidthIn,
                                                std::size_t bitWidthOut,
                                                std::size_t groupSize,
                                                std::size_t idx,
                                                std::size_t elementNum) noexcept
{
    const char *curKeyData = nullptr;

    std::size_t offsetSCW = idx * (16 * bitWidthIn);
    std::size_t offsetVCW =
        elementNum * (16 * bitWidthIn) +
        idx * (sizeof(GroupElement) * bitWidthIn * groupSize + //
               sizeof(GroupElement) +                          //
               sizeof(GroupElement) +                          //
               sizeof(GroupElement) * groupSize                //
              );

    curKeyData = (const char *)keyData + offsetSCW;
    dcfKey.sCW = (std::uint64_t (*)[2])(curKeyData);
    curKeyData = (const char *)keyData + offsetVCW;
    dcfKey.vCW = (GroupElement *)(curKeyData);
    curKeyData += sizeof(GroupElement) * bitWidthIn * groupSize;
    dcfKey.tLCW = (GroupElement *)(curKeyData);
    curKeyData += sizeof(GroupElement);
    dcfKey.tRCW = (GroupElement *)(curKeyData);
    curKeyData += sizeof(GroupElement);
    dcfKey.lastCW = (GroupElement *)(curKeyData);
}

template <typename GroupElement>
FAST_FSS_DEVICE static inline void dcfCacheSetPtr(
    DcfCache<GroupElement> &dcfCache,
    void                   *cacheData,
    std::size_t             bitWidthIn,
    std::size_t             groupSize,
    std::size_t             idx,
    std::size_t             elementNum) noexcept
{
    char       *curCacheData = (char *)cacheData;
    std::size_t offsetSCW    = idx * (16 * bitWidthIn);
    std::size_t offsetV = idx * (sizeof(GroupElement) * bitWidthIn * groupSize);
    dcfCache.stCache    = (std::uint64_t (*)[2])(curCacheData + offsetSCW);
    curCacheData        = curCacheData + 16 * elementNum * bitWidthIn;
    dcfCache.v          = (GroupElement *)(curCacheData + offsetV);
    dcfCache.preMaskedX = 0;
    dcfCache.preTo      = 0;
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void dcfKeyGen(
    DcfKey<GroupElement>      &key,
    GroupElement               alpha,
    const GroupElement        *beta,
    const void                *seed0,
    const void                *seed1,
    std::size_t                bitWidthIn,
    std::size_t                bitWidthOut,
    std::size_t                groupSize,
    const AES128GlobalContext *aesCtx = nullptr) noexcept
{
    constexpr std::uint64_t    MASK_MSB63   = 0xFFFF'FFFF'FFFF'FFFEULL;
    static const std::uint64_t PLAINTEXT[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    key.tLCW[0] = 0;
    key.tRCW[0] = 0;

    // copy seed
    std::uint64_t curS0[2] = {
        ((const std::uint64_t *)seed0)[0] & MASK_MSB63,
        ((const std::uint64_t *)seed0)[1],
    };
    std::uint64_t curS1[2] = {
        ((const std::uint64_t *)seed1)[0] & MASK_MSB63,
        ((const std::uint64_t *)seed1)[1],
    };
    int curT0 = 0, curT1 = 1;

    std::uint64_t sL0vL0sR0vR0[8];
    std::uint64_t sL1vL1sR1vR1[8];

    GroupElement *vAlphaPtr = key.lastCW;
    for (std::size_t j = 0; j < groupSize; j++)
    {
        vAlphaPtr[j] = 0;
    }
    AES128                      aes;
    DcfConvertCtx<GroupElement> dcfConvertCtx(aes);
    for (std::size_t i = 0; i < bitWidthIn; i++)
    {
        // sL0, tL0, vL0, sR0, tR0, vR0 <- G(s0)
        // sL1, tL1, vL1, sR1, tR1, vR1 <- G(s1)
        aes.set_enc_key(curS0, aesCtx);
        aes.enc_n_block<4>(sL0vL0sR0vR0, PLAINTEXT, aesCtx);
        aes.set_enc_key(curS1, aesCtx);
        aes.enc_n_block<4>(sL1vL1sR1vR1, PLAINTEXT, aesCtx);

        std::uint64_t *sL0 = sL0vL0sR0vR0 + 0;
        std::uint64_t *vL0 = sL0vL0sR0vR0 + 2;
        std::uint64_t *sR0 = sL0vL0sR0vR0 + 4;
        std::uint64_t *vR0 = sL0vL0sR0vR0 + 6;

        std::uint64_t *sL1 = sL1vL1sR1vR1 + 0;
        std::uint64_t *vL1 = sL1vL1sR1vR1 + 2;
        std::uint64_t *sR1 = sL1vL1sR1vR1 + 4;
        std::uint64_t *vR1 = sL1vL1sR1vR1 + 6;

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
        dcfConvertCtx.init(vLose1, groupSize, bitWidthOut, aesCtx);
        for (std::size_t j = 0; j < groupSize; j++)
        {
            key.vCW[i * groupSize + j] = dcfConvertCtx.getNext();
        }
        dcfConvertCtx.init(vLose0, groupSize, bitWidthOut, aesCtx);
        for (std::size_t j = 0; j < groupSize; j++)
        {
            key.vCW[i * groupSize + j] -= dcfConvertCtx.getNext();
        }
        for (std::size_t j = 0; j < groupSize; j++)
        {
            key.vCW[i * groupSize + j] -= vAlphaPtr[j];
        }
        for (std::size_t j = 0; j < groupSize; j++)
        {
            key.vCW[i * groupSize + j] *= ((GroupElement)(-2) * curT1 + 1);
        }
        // if Lose = L then vCW <- vCW + (-1)^t1 * beta
        if (lose == 0)
        {
            for (std::size_t j = 0; j < groupSize; j++)
            {
                key.vCW[i * groupSize + j] +=
                    ((GroupElement)(-2) * curT1 + 1) * beta[j];
            }
        }
        // vAlpha += -convert(vKeep1) + convert(vKeep0) + (-1)^t1 * vCW
        dcfConvertCtx.init(vKeep1, groupSize, bitWidthOut, aesCtx);
        for (std::size_t j = 0; j < groupSize; j++)
        {
            vAlphaPtr[j] -= dcfConvertCtx.getNext();
        }
        dcfConvertCtx.init(vKeep0, groupSize, bitWidthOut, aesCtx);
        for (std::size_t j = 0; j < groupSize; j++)
        {
            vAlphaPtr[j] += dcfConvertCtx.getNext();
        }
        for (std::size_t j = 0; j < groupSize; j++)
        {
            vAlphaPtr[j] +=
                ((GroupElement)(-2) * curT1 + 1) * key.vCW[i * groupSize + j];
        }
        //
        GroupElement tLCW = tL0 ^ tL1 ^ alphaI ^ 1;
        GroupElement tRCW = tR0 ^ tR1 ^ alphaI;
        // CW(i) <- sCW || vCW || tLCW || tRCW
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
    // CW(n+1) <- (-1)^t1 * ( convert(s1) - convert(s0) - vAlpha )
    for (std::size_t j = 0; j < groupSize; j++)
    {
        key.lastCW[j] = -vAlphaPtr[j];
    }
    dcfConvertCtx.init(curS1, groupSize, bitWidthOut, aesCtx);
    for (std::size_t j = 0; j < groupSize; j++)
    {
        key.lastCW[j] += dcfConvertCtx.getNext();
    }
    dcfConvertCtx.init(curS0, groupSize, bitWidthOut, aesCtx);
    for (std::size_t j = 0; j < groupSize; j++)
    {
        key.lastCW[j] -= dcfConvertCtx.getNext();
    }
    for (std::size_t j = 0; j < groupSize; j++)
    {
        key.lastCW[j] *= ((GroupElement)(-2) * curT1 + 1);
    }
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void dcfEval(
    GroupElement               *out,
    const DcfKey<GroupElement> &key,
    GroupElement                maskedX,
    const void                 *seed,
    int                         partyId,
    std::size_t                 bitWidthIn,
    std::size_t                 bitWidthOut,
    std::size_t                 groupSize,
    DcfCache<GroupElement>     *cache  = nullptr,
    const AES128GlobalContext  *aesCtx = nullptr) noexcept
{
    constexpr std::uint64_t    MASK_MSB63   = 0xFFFF'FFFF'FFFF'FFFEULL;
    static const std::uint64_t PLAINTEXT[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    std::uint64_t curS[2] = {
        ((const std::uint64_t *)seed)[0] & MASK_MSB63,
        ((const std::uint64_t *)seed)[1],
    };
    int curT = partyId;

    for (std::size_t j = 0; j < groupSize; j++)
    {
        out[j] = 0;
    }

    std::size_t idx_from = 0;
    if (cache != nullptr)
    {
        idx_from = clz<GroupElement>((maskedX ^ cache->preMaskedX), bitWidthIn);
        idx_from = (idx_from < cache->preTo) ? idx_from : cache->preTo;

        if (0 < idx_from && idx_from <= bitWidthIn)
        {
            curT    = cache->stCache[idx_from - 1][0] & 1;
            curS[0] = cache->stCache[idx_from - 1][0] & MASK_MSB63;
            curS[1] = cache->stCache[idx_from - 1][1];
            for (std::size_t j = 0; j < groupSize; j++)
            {
                out[j] = cache->v[(idx_from - 1) * groupSize + j];
            }
        }
    }

    std::uint64_t               sLvLsRvR[8];
    AES128                      aes;
    DcfConvertCtx<GroupElement> dcfConvertCtx(aes);
    for (std::size_t i = idx_from; i < bitWidthIn; i++)
    {
        // sL || vL || sR || vR <- G(s(i-1))
        aes.set_enc_key(curS, aesCtx);
        aes.enc_n_block<4>(sLvLsRvR, PLAINTEXT, aesCtx);

        std::uint64_t *sL = sLvLsRvR + 0;
        std::uint64_t *vL = sLvLsRvR + 2;
        std::uint64_t *sR = sLvLsRvR + 4;
        std::uint64_t *vR = sLvLsRvR + 6;

        int tL = sL[0] & 1;
        int tR = sR[0] & 1;
        sL[0] &= MASK_MSB63;
        sR[0] &= MASK_MSB63;

        // sL || vL || sR || vR <-
        //      (sL || vL || sR || vR) ^ (sCW || vCW || tLCW || tRCW) * t
        if (curT)
        {
            sL[0] ^= key.sCW[i][0], sL[1] ^= key.sCW[i][1];
            tL ^= (key.tLCW[0] >> i) & 1;
            sR[0] ^= key.sCW[i][0], sR[1] ^= key.sCW[i][1];
            tR ^= (key.tRCW[0] >> i) & 1;
        }
        // if xI = 0
        //  then    V <- V + (-1)^b [Convert(vL) + t(i-1) * vCW]
        //          s(i) <- sL, t(i) <- tL
        // else
        //  then    V <- V + (-1)^b [Convert(vR) + t(i-1) * vCW]
        //          s(i) <- sR, t(i) <- tR
        int xI      = (maskedX >> (bitWidthIn - i - 1)) & 1;
        int tmpCurT = curT;
        if (xI == 0)
        {
            dcfConvertCtx.init(vL, groupSize, bitWidthOut, aesCtx);
            curS[0] = sL[0], curS[1] = sL[1];
            curT = tL;
        }
        else
        {
            dcfConvertCtx.init(vR, groupSize, bitWidthOut, aesCtx);
            curS[0] = sR[0], curS[1] = sR[1];
            curT = tR;
        }
        for (std::size_t j = 0; j < groupSize; j++)
        {
            out[j] += ((GroupElement)(-2) * partyId + 1) *
                      (dcfConvertCtx.getNext() +
                       tmpCurT * key.vCW[i * groupSize + j]);
        }

        if (cache != nullptr)
        {
            cache->stCache[i][0] = curS[0] & MASK_MSB63;
            cache->stCache[i][1] = curS[1];
            cache->stCache[i][0] ^= curT;

            for (std::size_t j = 0; j < groupSize; j++)
            {
                cache->v[i * groupSize + j] = out[j];
            }
            cache->preTo      = i + 1;
            cache->preMaskedX = maskedX;
        }
    }
    // V <- V + (-1)^b [Convert(s(n)) + t(n) * lastCW]
    dcfConvertCtx.init(curS, groupSize, bitWidthOut, aesCtx);
    for (std::size_t j = 0; j < groupSize; j++)
    {
        out[j] += ((GroupElement)(-2) * partyId + 1) *
                  (dcfConvertCtx.getNext() + curT * key.lastCW[j]);
    }
}

} // namespace FastFss::impl

#endif