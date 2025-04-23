#pragma once
#ifndef SRC_IMPL_MIC_H
#define SRC_IMPL_MIC_H

#include "dcf.h"

namespace FastFss::impl {

template <typename GroupElement>
FAST_FSS_DEVICE inline void dcfMICKeyGen(DcfKey<GroupElement>& key,
                                         GroupElement*         z, // intervalNum
                                         GroupElement          alpha,
                                         const void*           seed0,
                                         const void*           seed1,
                                         const GroupElement*   leftBoundary,
                                         const GroupElement*   rightBoundary,
                                         std::size_t           intervalNum,
                                         std::size_t           bitWidthIn,
                                         std::size_t           bitWidthOut)
{
    dcfKeyGen<GroupElement>(key, alpha - 1, 1, seed0, seed1, bitWidthIn,
                            bitWidthOut);

    GroupElement MAX = modBits<GroupElement>((GroupElement)(-1), bitWidthIn);
    for (std::size_t i = 0; i < intervalNum; i++)
    {
        GroupElement qPrime      = rightBoundary[i] + 1;
        GroupElement alphaP      = leftBoundary[i] + alpha;
        GroupElement alphaQ      = rightBoundary[i] + alpha;
        GroupElement alphaQPrime = rightBoundary[i] + 1 + alpha;

        qPrime      = modBits<GroupElement>(qPrime, bitWidthIn);
        alphaP      = modBits<GroupElement>(alphaP, bitWidthIn);
        alphaQ      = modBits<GroupElement>(alphaQ, bitWidthIn);
        alphaQPrime = modBits<GroupElement>(alphaQPrime, bitWidthIn);

        z[i] = (alphaP > alphaQ) - (alphaP > leftBoundary[i]) +
               (alphaQPrime > qPrime) + (alphaQ == MAX);
    }
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void dcfMICEval(
    GroupElement*               sharedOut, // intervalNum
    GroupElement                maskedX,
    const DcfKey<GroupElement>& key,
    const GroupElement*         sharedZ, // intervalNum
    const void*                 seed,
    int                         partyId,
    const GroupElement*         leftBoundary,
    const GroupElement*         rightBoundary,
    std::size_t                 intervalNum,
    std::size_t                 bitWidthIn,
    std::size_t                 bitWidthOut)
{
    maskedX = modBits<GroupElement>(maskedX, bitWidthIn);

    GroupElement sp = 0, sq = 0;
    {
        GroupElement qPrime  = rightBoundary[0] + 1;
        GroupElement xP      = (maskedX - 1 - leftBoundary[0]);
        GroupElement xQPrime = (maskedX - 1 - qPrime);

        qPrime  = modBits<GroupElement>(qPrime, bitWidthIn);
        xP      = modBits<GroupElement>(xP, bitWidthIn);
        xQPrime = modBits<GroupElement>(xQPrime, bitWidthIn);

        sp = dcfEval<GroupElement>(key, xP, seed, partyId, bitWidthIn,
                                   bitWidthOut);
        sq = dcfEval<GroupElement>(key, xQPrime, seed, partyId, bitWidthIn,
                                   bitWidthOut);

        sharedOut[0] = sq - sp + sharedZ[0];
        if (partyId == 1)
        {
            sharedOut[0] += (maskedX > leftBoundary[0]) - (maskedX > qPrime);
        }
    }
    for (std::size_t i = 1; i < intervalNum; i++)
    {
        GroupElement qPrime    = rightBoundary[i] + 1;
        GroupElement xP        = (maskedX - 1 - leftBoundary[i]);
        GroupElement xQPrime   = (maskedX - 1 - qPrime);
        GroupElement privQAdd1 = rightBoundary[i - 1] + 1;

        privQAdd1 = modBits<GroupElement>(privQAdd1, bitWidthIn);
        qPrime    = modBits<GroupElement>(qPrime, bitWidthIn);
        xP        = modBits<GroupElement>(xP, bitWidthIn);
        xQPrime   = modBits<GroupElement>(xQPrime, bitWidthIn);

        if (leftBoundary[i] == privQAdd1)
        {
            sp = sq;
        }
        else
        {
            sp = dcfEval<GroupElement>(key, xP, seed, partyId, bitWidthIn,
                                       bitWidthOut);
        }
        sq = dcfEval<GroupElement>(key, xQPrime, seed, partyId, bitWidthIn,
                                   bitWidthOut);
        sharedOut[i] = sq - sp + sharedZ[i];
        if (partyId == 1)
        {
            sharedOut[i] += (maskedX > leftBoundary[i]) - (maskedX > qPrime);
        }
    }
}

} // namespace FastFss::impl

#endif