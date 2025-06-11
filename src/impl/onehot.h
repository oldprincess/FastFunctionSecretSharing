#ifndef SRC_IMPL_ONEHOT_H
#define SRC_IMPL_ONEHOT_H

#include "def.h"
#include "number.h"

namespace FastFss::impl {

// bitWidthIn >= 3
static inline std::size_t onehotGetKeyDataSize(std::size_t bitWidthIn,
                                               std::size_t elementNum) noexcept
{
    return elementNum * (1ULL << bitWidthIn) / 8;
}

// bitWidthIn >= 3
template <typename GroupElement>
FAST_FSS_DEVICE static void onehotKeyGen(void*        key,
                                         GroupElement alpha,
                                         std::size_t  bitWidthIn) noexcept
{
    alpha              = modBits<GroupElement>(alpha, bitWidthIn);
    std::size_t idx    = alpha / 8;
    std::size_t offset = alpha % 8;
    ((std::uint8_t*)key)[idx] ^= (1 << offset);
}

// bitWidthIn >= 3
template <typename GroupElement>
FAST_FSS_DEVICE static void onehotLutEval(GroupElement*       sharedOutE,
                                          GroupElement*       sharedOutT,
                                          GroupElement        maskedX,
                                          const void*         key,
                                          const GroupElement* lut,
                                          int                 partyId,
                                          std::size_t bitWidthIn) noexcept
{
    std::size_t  totalNum = (1ULL << bitWidthIn);
    GroupElement mask     = (GroupElement)(totalNum - 1);
    std::size_t  num      = totalNum;
    std::size_t  i        = 0;

    sharedOutE[0] = 0;
    sharedOutT[0] = 0;

    while (num >= 256)
    {
        std::uint64_t e0 = ((const std::uint64_t*)key)[0];
        std::uint64_t e1 = ((const std::uint64_t*)key)[1];
        std::uint64_t e2 = ((const std::uint64_t*)key)[2];
        std::uint64_t e3 = ((const std::uint64_t*)key)[3];
        for (int j = 0; j < 64; j++)
        {
            bool e0i = (e0 >> j) & 1;
            bool e1i = (e1 >> j) & 1;
            bool e2i = (e2 >> j) & 1;
            bool e3i = (e3 >> j) & 1;
            sharedOutE[0] += e0i + e1i + e2i + e3i;

            std::size_t lutIdx0 = (maskedX - i - j) & mask;
            std::size_t lutIdx1 = (maskedX - i - j - 64) & mask;
            std::size_t lutIdx2 = (maskedX - i - j - 128) & mask;
            std::size_t lutIdx3 = (maskedX - i - j - 192) & mask;
            sharedOutT[0] += (e0i) ? lut[lutIdx0] : (GroupElement)0;
            sharedOutT[0] += (e1i) ? lut[lutIdx1] : (GroupElement)0;
            sharedOutT[0] += (e2i) ? lut[lutIdx2] : (GroupElement)0;
            sharedOutT[0] += (e3i) ? lut[lutIdx3] : (GroupElement)0;
        }
        key = (const void*)((const std::uint64_t*)key + 4);
        i += 256, num -= 256;
    }

    while (num >= 128)
    {
        std::uint64_t e0 = ((const std::uint64_t*)key)[0];
        std::uint64_t e1 = ((const std::uint64_t*)key)[1];
        for (int j = 0; j < 64; j++)
        {
            bool e0i = (e0 >> j) & 1;
            bool e1i = (e1 >> j) & 1;
            sharedOutE[0] += e0i + e1i;

            std::size_t lutIdx0 = (maskedX - i - j) & mask;
            std::size_t lutIdx1 = (maskedX - i - j - 64) & mask;
            sharedOutT[0] += (e0i) ? lut[lutIdx0] : (GroupElement)0;
            sharedOutT[0] += (e1i) ? lut[lutIdx1] : (GroupElement)0;
        }
        key = (const void*)((const std::uint64_t*)key + 2);
        i += 128, num -= 128;
    }

    while (num >= 64)
    {
        std::uint64_t e = *(const std::uint64_t*)key;
        for (int j = 0; j < 64; j++, i++)
        {
            bool ei = (e >> j) & 1;
            sharedOutE[0] += ei;
            sharedOutT[0] += (ei) ? lut[(maskedX - i) & mask] : (GroupElement)0;
        }
        key = (const void*)((const std::uint64_t*)key + 1);
        num -= 64;
    }

    while (num >= 32)
    {
        std::uint32_t e = *(const std::uint32_t*)key;
        for (int j = 0; j < 32; j++, i++)
        {
            bool ei = (e >> j) & 1;
            sharedOutE[0] += ei;
            sharedOutT[0] += (ei) ? lut[(maskedX - i) & mask] : (GroupElement)0;
        }
        key = (const void*)((const std::uint32_t*)key + 1);
        num -= 32;
    }

    while (num >= 16)
    {
        std::uint16_t e = *(const std::uint16_t*)key;
        for (int j = 0; j < 16; j++, i++)
        {
            bool ei = (e >> j) & 1;
            sharedOutE[0] += ei;
            sharedOutT[0] += (ei) ? lut[(maskedX - i) & mask] : (GroupElement)0;
        }
        key = (const void*)((const std::uint16_t*)key + 1);
        num -= 16;
    }

    while (num >= 8)
    {
        std::uint8_t e = *(const std::uint8_t*)key;
        for (int j = 0; j < 8; j++, i++)
        {
            bool ei = (e >> j) & 1;
            sharedOutE[0] += ei;
            sharedOutT[0] += (ei) ? lut[(maskedX - i) & mask] : (GroupElement)0;
        }
        key = (const void*)((const std::uint8_t*)key + 1);
        num -= 8;
    }

    if (partyId == 0)
    {
        sharedOutE[0] = (GroupElement)(0 - sharedOutE[0]);
        sharedOutT[0] = (GroupElement)(0 - sharedOutT[0]);
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