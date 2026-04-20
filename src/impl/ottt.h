#ifndef SRC_IMPL_OTTT_H
#define SRC_IMPL_OTTT_H

#include "def.h"
#include "number.h"

namespace FastFss::impl {

// bitWidthIn >= 3
static inline std::size_t otttGetKeyDataSize(std::size_t bitWidthIn, std::size_t elementNum) noexcept
{
    return elementNum * (1ULL << bitWidthIn) / 8;
}

// bitWidthIn >= 3
template <typename GroupElement>
FAST_FSS_DEVICE static void otttKeyGen(void *key, GroupElement alpha, std::size_t bitWidthIn) noexcept
{
    alpha                  = modBits<GroupElement>(alpha, bitWidthIn);
    const auto  alphaIndex = static_cast<std::size_t>(alpha);
    std::size_t idx        = alphaIndex / 8;
    std::size_t offset     = alphaIndex % 8;
    ((std::uint8_t *)key)[idx] ^= (1 << offset);
}

// bitWidthIn >= 3
template <typename GroupElement>
FAST_FSS_DEVICE static void otttLutEval(GroupElement       *sharedOutE,
                                        GroupElement       *sharedOutT,
                                        GroupElement        maskedX,
                                        const void         *key,
                                        const GroupElement *lut,
                                        std::size_t         lutNum,
                                        int                 partyId,
                                        std::size_t         bitWidthIn) noexcept
{
    std::size_t totalNum     = (1ULL << bitWidthIn);
    std::size_t mask         = totalNum - 1;
    std::size_t num          = totalNum;
    std::size_t i            = 0;
    const auto  maskedXIndex = static_cast<std::size_t>(maskedX);

    sharedOutE[0] = 0;
    for (std::size_t k = 0; k < lutNum; k++)
    {
        sharedOutT[k] = 0;
    }

    while (num >= 256)
    {
        std::uint64_t e0 = ((const std::uint64_t *)key)[0];
        std::uint64_t e1 = ((const std::uint64_t *)key)[1];
        std::uint64_t e2 = ((const std::uint64_t *)key)[2];
        std::uint64_t e3 = ((const std::uint64_t *)key)[3];
        for (int j = 0; j < 64; j++)
        {
            bool e0i = (e0 >> j) & 1;
            bool e1i = (e1 >> j) & 1;
            bool e2i = (e2 >> j) & 1;
            bool e3i = (e3 >> j) & 1;
            sharedOutE[0] += (e0i ? 1 : 0);
            sharedOutE[0] += (e1i ? 1 : 0);
            sharedOutE[0] += (e2i ? 1 : 0);
            sharedOutE[0] += (e3i ? 1 : 0);

            std::size_t lutIdx0 = (maskedXIndex - i - j) & mask;
            std::size_t lutIdx1 = (maskedXIndex - i - j - 64) & mask;
            std::size_t lutIdx2 = (maskedXIndex - i - j - 128) & mask;
            std::size_t lutIdx3 = (maskedXIndex - i - j - 192) & mask;
            for (std::size_t k = 0; k < lutNum; ++k)
            {
                sharedOutT[k] += (e0i) ? lut[lutIdx0 + k * totalNum] : (GroupElement)0;
                sharedOutT[k] += (e1i) ? lut[lutIdx1 + k * totalNum] : (GroupElement)0;
                sharedOutT[k] += (e2i) ? lut[lutIdx2 + k * totalNum] : (GroupElement)0;
                sharedOutT[k] += (e3i) ? lut[lutIdx3 + k * totalNum] : (GroupElement)0;
            }
        }
        key = (const void *)((const std::uint64_t *)key + 4);
        i += 256, num -= 256;
    }

    while (num >= 128)
    {
        std::uint64_t e0 = ((const std::uint64_t *)key)[0];
        std::uint64_t e1 = ((const std::uint64_t *)key)[1];
        for (int j = 0; j < 64; j++)
        {
            bool e0i = (e0 >> j) & 1;
            bool e1i = (e1 >> j) & 1;
            sharedOutE[0] += (e0i ? 1 : 0);
            sharedOutE[0] += (e1i ? 1 : 0);

            std::size_t lutIdx0 = (maskedXIndex - i - j) & mask;
            std::size_t lutIdx1 = (maskedXIndex - i - j - 64) & mask;
            for (std::size_t k = 0; k < lutNum; ++k)
            {
                sharedOutT[k] += (e0i) ? lut[lutIdx0 + k * totalNum] : (GroupElement)0;
                sharedOutT[k] += (e1i) ? lut[lutIdx1 + k * totalNum] : (GroupElement)0;
            }
        }
        key = (const void *)((const std::uint64_t *)key + 2);
        i += 128, num -= 128;
    }

    while (num >= 64)
    {
        std::uint64_t e = *(const std::uint64_t *)key;
        for (int j = 0; j < 64; j++, i++)
        {
            bool ei = (e >> j) & 1;
            sharedOutE[0] += (ei ? 1 : 0);
            std::size_t lutIdx = (maskedXIndex - i) & mask;
            for (std::size_t k = 0; k < lutNum; k++)
            {
                sharedOutT[k] += (ei) ? lut[lutIdx + k * totalNum] : (GroupElement)0;
            }
        }
        key = (const void *)((const std::uint64_t *)key + 1);
        num -= 64;
    }

    while (num >= 32)
    {
        std::uint32_t e = *(const std::uint32_t *)key;
        for (int j = 0; j < 32; j++, i++)
        {
            bool ei = (e >> j) & 1;
            sharedOutE[0] += (ei ? 1 : 0);
            std::size_t lutIdx = (maskedXIndex - i) & mask;
            for (std::size_t k = 0; k < lutNum; k++)
            {
                sharedOutT[k] += (ei) ? lut[lutIdx + k * totalNum] : (GroupElement)0;
            }
        }
        key = (const void *)((const std::uint32_t *)key + 1);
        num -= 32;
    }

    while (num >= 16)
    {
        std::uint16_t e = *(const std::uint16_t *)key;
        for (int j = 0; j < 16; j++, i++)
        {
            bool ei = (e >> j) & 1;
            sharedOutE[0] += (ei ? 1 : 0);
            std::size_t lutIdx = (maskedXIndex - i) & mask;
            for (std::size_t k = 0; k < lutNum; k++)
            {
                sharedOutT[k] += (ei) ? lut[lutIdx + k * totalNum] : (GroupElement)0;
            }
        }
        key = (const void *)((const std::uint16_t *)key + 1);
        num -= 16;
    }

    while (num >= 8)
    {
        std::uint8_t e = *(const std::uint8_t *)key;
        for (int j = 0; j < 8; j++, i++)
        {
            bool ei = (e >> j) & 1;
            sharedOutE[0] += (ei ? 1 : 0);
            std::size_t lutIdx = (maskedXIndex - i) & mask;
            for (std::size_t k = 0; k < lutNum; k++)
            {
                sharedOutT[k] += (ei) ? lut[lutIdx + k * totalNum] : (GroupElement)0;
            }
        }
        key = (const void *)((const std::uint8_t *)key + 1);
        num -= 8;
    }

    if (partyId == 0)
    {
        sharedOutE[0] = (GroupElement)(0 - sharedOutE[0]);
        for (std::size_t k = 0; k < lutNum; k++)
        {
            sharedOutT[k] = (GroupElement)(0 - sharedOutT[k]);
        }
    }
    // // E = 1 or -1.
    // // E = ((E - 1) >> 1) & 1: 1(V need times -1) 0(V need not times -1)
    // if (partyId == 0)
    // {
    //     sharedOutE[0] -= 1;
    //     sharedOutE[0] = modBits<GroupElement>((sharedOutE[0] >> 1) + (sharedOutE[0] & 1), 1);
    // }
    // else
    // {
    //     sharedOutE[0] = modBits<GroupElement>(sharedOutE[0] >> 1, 1);
    // }
}

} // namespace FastFss::impl

#endif
