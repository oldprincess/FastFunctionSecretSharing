#pragma once
#ifndef SRC_IMPL_MIC_H
#define SRC_IMPL_MIC_H

#include "dcf.h"

namespace FastFss::impl {

template <typename GroupElement>
FAST_FSS_DEVICE inline void dcfMICKeyGen(DcfKey<GroupElement> &key,
                                         GroupElement         *z, // intervalNum
                                         GroupElement          alpha,
                                         const void           *seed0,
                                         const void           *seed1,
                                         const GroupElement   *leftEndpoints,
                                         const GroupElement   *rightEndpoints,
                                         std::size_t           intervalNum,
                                         std::size_t           bitWidthIn,
                                         std::size_t           bitWidthOut)
{
    const GroupElement ONE = 1;
    dcfKeyGen<GroupElement>(key, alpha - 1, &ONE, seed0, seed1, bitWidthIn, bitWidthOut, 1);

    GroupElement MAX = modBits<GroupElement>((GroupElement)(-1), bitWidthIn);
    for (std::size_t i = 0; i < intervalNum; i++)
    {
        GroupElement qPrime      = rightEndpoints[i] + 1;
        GroupElement alphaP      = leftEndpoints[i] + alpha;
        GroupElement alphaQ      = rightEndpoints[i] + alpha;
        GroupElement alphaQPrime = rightEndpoints[i] + 1 + alpha;
        GroupElement pi          = leftEndpoints[i];

        qPrime      = modBits<GroupElement>(qPrime, bitWidthIn);
        alphaP      = modBits<GroupElement>(alphaP, bitWidthIn);
        alphaQ      = modBits<GroupElement>(alphaQ, bitWidthIn);
        alphaQPrime = modBits<GroupElement>(alphaQPrime, bitWidthIn);
        pi          = modBits<GroupElement>(pi, bitWidthIn);

        z[i] = (alphaP > alphaQ) - (alphaP > pi) + (alphaQPrime > qPrime) + (alphaQ == MAX);
    }
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void dcfMICEval(GroupElement               *sharedOut, // intervalNum
                                       GroupElement                maskedX,
                                       const DcfKey<GroupElement> &key,
                                       const GroupElement         *sharedZ, // intervalNum
                                       const void                 *seed,
                                       int                         partyId,
                                       const GroupElement         *leftEndpoints,
                                       const GroupElement         *rightEndpoints,
                                       std::size_t                 intervalNum,
                                       std::size_t                 bitWidthIn,
                                       std::size_t                 bitWidthOut,
                                       DcfCache<GroupElement>     *cache = nullptr)
{
    maskedX = modBits<GroupElement>(maskedX, bitWidthIn);

    GroupElement sp = 0, sq = 0;
    {
        GroupElement qPrime  = rightEndpoints[0] + 1;
        GroupElement xP      = (maskedX - 1 - leftEndpoints[0]);
        GroupElement xQPrime = (maskedX - 1 - qPrime);
        GroupElement p0      = leftEndpoints[0];

        qPrime  = modBits<GroupElement>(qPrime, bitWidthIn);
        xP      = modBits<GroupElement>(xP, bitWidthIn);
        xQPrime = modBits<GroupElement>(xQPrime, bitWidthIn);
        p0      = modBits<GroupElement>(p0, bitWidthIn);

        dcfEval<GroupElement>(&sp, key, xP, seed, partyId, bitWidthIn, bitWidthOut, 1, cache);
        dcfEval<GroupElement>(&sq, key, xQPrime, seed, partyId, bitWidthIn, bitWidthOut, 1, cache);

        sharedOut[0] = sq - sp + sharedZ[0];
        if (partyId == 1)
        {
            sharedOut[0] += (maskedX > p0) - (maskedX > qPrime);
        }
    }
    for (std::size_t i = 1; i < intervalNum; i++)
    {
        GroupElement qPrime    = rightEndpoints[i] + 1;
        GroupElement xP        = (maskedX - 1 - leftEndpoints[i]);
        GroupElement xQPrime   = (maskedX - 1 - qPrime);
        GroupElement privQAdd1 = rightEndpoints[i - 1] + 1;
        GroupElement pi        = leftEndpoints[i];

        privQAdd1 = modBits<GroupElement>(privQAdd1, bitWidthIn);
        qPrime    = modBits<GroupElement>(qPrime, bitWidthIn);
        xP        = modBits<GroupElement>(xP, bitWidthIn);
        xQPrime   = modBits<GroupElement>(xQPrime, bitWidthIn);
        pi        = modBits<GroupElement>(pi, bitWidthIn);

        if (leftEndpoints[i] == privQAdd1)
        {
            sp = sq;
        }
        else
        {
            dcfEval<GroupElement>(&sp, key, xP, seed, partyId, bitWidthIn, bitWidthOut, 1, cache);
        }
        dcfEval<GroupElement>(&sq, key, xQPrime, seed, partyId, bitWidthIn, bitWidthOut, 1, cache);
        sharedOut[i] = sq - sp + sharedZ[i];
        if (partyId == 1)
        {
            sharedOut[i] += (maskedX > pi) - (maskedX > qPrime);
        }
    }
}

} // namespace FastFss::impl

#endif