#pragma once
#ifndef SRC_IMPL_SPLINE_H
#define SRC_IMPL_SPLINE_H

#include "dcf.h"

namespace FastFss::impl {

template <typename GroupElement>
struct DcfSplineEvalWorkspace
{
    GroupElement *currentDcfOut      = nullptr; // intervalNum * (degree + 1)
    GroupElement *nextDcfOut         = nullptr; // intervalNum * (degree + 1)
    GroupElement *wrapAroundSegment  = nullptr; // degree + 1
    GroupElement *coefficientScratch = nullptr; // degree + 1
};

template <typename GroupElement>
struct DcfSplineCache
{
    DcfCache<GroupElement>               dcfCache;
    DcfSplineEvalWorkspace<GroupElement> workspace;
};

template <typename GroupElement>
FAST_FSS_HD static inline std::size_t dcfSplineGetKeyDataSize(std::size_t bitWidthIn,
                                                              std::size_t bitWidthOut,
                                                              std::size_t intervalNum,
                                                              std::size_t degree,
                                                              std::size_t elementNum) noexcept
{
    return dcfGetKeyDataSize<GroupElement>(bitWidthIn, bitWidthOut, intervalNum * (degree + 1), elementNum);
}

template <typename GroupElement>
FAST_FSS_HD static inline std::size_t dcfSplineGetCacheDataSize(std::size_t bitWidthIn,
                                                                std::size_t bitWidthOut,
                                                                std::size_t intervalNum,
                                                                std::size_t degree,
                                                                std::size_t elementNum) noexcept
{
    std::size_t groupSize = intervalNum * (degree + 1);
    std::size_t coeffNum  = degree + 1;
    return dcfGetCacheDataSize<GroupElement>(bitWidthIn, groupSize, elementNum) +
           elementNum * sizeof(GroupElement) * (2 * groupSize + 2 * coeffNum);
}

template <typename GroupElement>
FAST_FSS_HD static inline void dcfSplineCacheSetPtr(DcfSplineCache<GroupElement> &cache,
                                                    void                         *cacheData,
                                                    std::size_t                   bitWidthIn,
                                                    std::size_t                   groupSize,
                                                    std::size_t                   coeffNum,
                                                    std::size_t                   idx,
                                                    std::size_t                   elementNum) noexcept
{
    dcfCacheSetPtr<GroupElement>(cache.dcfCache, cacheData, bitWidthIn, groupSize, idx, elementNum);

    char       *curCacheData   = static_cast<char *>(cacheData);
    std::size_t dcfCacheSize   = dcfGetCacheDataSize<GroupElement>(bitWidthIn, groupSize, elementNum);
    std::size_t workspacePitch = sizeof(GroupElement) * (2 * groupSize + 2 * coeffNum);
    char       *workspaceBase  = curCacheData + dcfCacheSize + idx * workspacePitch;

    cache.workspace.currentDcfOut = reinterpret_cast<GroupElement *>(workspaceBase);
    workspaceBase += sizeof(GroupElement) * groupSize;
    cache.workspace.nextDcfOut = reinterpret_cast<GroupElement *>(workspaceBase);
    workspaceBase += sizeof(GroupElement) * groupSize;
    cache.workspace.wrapAroundSegment = reinterpret_cast<GroupElement *>(workspaceBase);
    workspaceBase += sizeof(GroupElement) * coeffNum;
    cache.workspace.coefficientScratch = reinterpret_cast<GroupElement *>(workspaceBase);
}

template <typename GroupElement>
FAST_FSS_DEVICE static inline void splineShiftPolynomialCoefficients(GroupElement       *shiftedCoefficients,
                                                                     const GroupElement *coefficients,
                                                                     GroupElement        shift,
                                                                     std::size_t         degree,
                                                                     std::size_t         bitWidthOut) noexcept
{
    GroupElement negShift = modBits<GroupElement>(GroupElement(0) - shift, static_cast<int>(bitWidthOut));

    for (std::size_t i = 0; i <= degree; ++i)
    {
        shiftedCoefficients[i] = 0;
    }

    for (std::size_t i = degree + 1; i > 0; --i)
    {
        GroupElement coeff = coefficients[i - 1];
        for (std::size_t j = degree; j > 0; --j)
        {
            shiftedCoefficients[j] = shiftedCoefficients[j - 1] + shiftedCoefficients[j] * negShift;
        }
        shiftedCoefficients[0] = shiftedCoefficients[0] * negShift + coeff;
    }

    for (std::size_t i = 0; i <= degree; ++i)
    {
        shiftedCoefficients[i] = modBits<GroupElement>(shiftedCoefficients[i], static_cast<int>(bitWidthOut));
    }
}

template <typename GroupElement>
FAST_FSS_DEVICE static inline GroupElement splineEvaluatePolynomial(const GroupElement *coefficients,
                                                                    GroupElement        x,
                                                                    std::size_t         degree,
                                                                    std::size_t         bitWidthOut) noexcept
{
    GroupElement result = 0;
    GroupElement power  = 1;

    for (std::size_t i = 0; i <= degree; ++i)
    {
        result = result + coefficients[i] * power;
        power  = power * x;
    }

    return modBits<GroupElement>(result, static_cast<int>(bitWidthOut));
}

template <typename GroupElement>
FAST_FSS_DEVICE inline void dcfSplineKeyGen(DcfKey<GroupElement> &key,
                                            GroupElement         *e,
                                            GroupElement         *beta,
                                            GroupElement          alpha,
                                            const GroupElement   *coefficients,
                                            const void           *seed0,
                                            const void           *seed1,
                                            const GroupElement   *leftEndpoints,
                                            const GroupElement   *rightEndpoints,
                                            std::size_t           intervalNum,
                                            std::size_t           degree,
                                            std::size_t           bitWidthIn,
                                            std::size_t           bitWidthOut) noexcept
{
    std::size_t  coeffNum  = degree + 1;
    std::size_t  groupSize = intervalNum * coeffNum;
    GroupElement maxValue  = modBits<GroupElement>(GroupElement(-1), (int)bitWidthIn);

    // beta{i} = (f'{i,d}, ..., f'{i,0}), f'{i}(x) = f{i}(x - alpha)
    for (std::size_t i = 0; i < intervalNum; ++i)
    {
        splineShiftPolynomialCoefficients<GroupElement>(                                 //
            beta + i * coeffNum, coefficients + i * coeffNum, alpha, degree, bitWidthOut //
        );                                                                               //
    }
    // GenDcf(1^{\lambda}, alpha - 1, beta)
    dcfKeyGen<GroupElement>(key, alpha - 1, beta, seed0, seed1, bitWidthIn, bitWidthOut, groupSize);

    for (std::size_t i = 0; i < intervalNum; ++i)
    {
        GroupElement left       = modBits<GroupElement>(leftEndpoints[i], (int)(bitWidthIn));
        GroupElement right      = modBits<GroupElement>(rightEndpoints[i], (int)(bitWidthIn));
        GroupElement rightPrime = modBits<GroupElement>(right + 1, (int)(bitWidthIn));

        GroupElement alphaL      = modBits<GroupElement>(left + alpha, (int)(bitWidthIn));
        GroupElement alphaR      = modBits<GroupElement>(right + alpha, (int)(bitWidthIn));
        GroupElement alphaRPrime = modBits<GroupElement>(rightPrime + alpha, static_cast<int>(bitWidthIn));

        GroupElement correction = static_cast<GroupElement>((alphaL > alphaR) - (alphaL > left) +
                                                            (alphaRPrime > rightPrime) + (alphaR == maxValue));
        // e{i} = cr{i} * beta{i}
        for (std::size_t j = 0; j < coeffNum; ++j)
        {
            e[i * coeffNum + j] = modBits<GroupElement>(correction * beta[i * coeffNum + j], (int)(bitWidthOut));
        }
    }
}

template <typename GroupElement>
FAST_FSS_DEVICE inline GroupElement dcfSplineEval(const DcfKey<GroupElement>   &key,
                                                  GroupElement                  maskedX,
                                                  const GroupElement           *sharedBeta,
                                                  const GroupElement           *sharedE,
                                                  const void                   *seed,
                                                  int                           partyId,
                                                  const GroupElement           *leftEndpoints,
                                                  const GroupElement           *rightEndpoints,
                                                  std::size_t                   intervalNum,
                                                  std::size_t                   degree,
                                                  std::size_t                   bitWidthIn,
                                                  std::size_t                   bitWidthOut,
                                                  DcfSplineCache<GroupElement> &cache) noexcept
{
    std::size_t coeffNum  = degree + 1;
    std::size_t groupSize = intervalNum * coeffNum;

    DcfSplineEvalWorkspace<GroupElement> &workspace = cache.workspace;

    maskedX = modBits<GroupElement>(maskedX, static_cast<int>(bitWidthIn));

    if (intervalNum == 0)
    {
        return 0;
    }

    for (std::size_t j = 0; j < coeffNum; ++j)
    {
        workspace.coefficientScratch[j] = 0;
    }

    for (std::size_t i = 0; i < intervalNum; i++)
    {
        GroupElement left       = modBits<GroupElement>(leftEndpoints[i], (int)bitWidthIn);
        GroupElement right      = modBits<GroupElement>(rightEndpoints[i], (int)bitWidthIn);
        GroupElement rightPrime = modBits<GroupElement>(right + 1, (int)bitWidthIn);

        if (i > 0 && modBits<GroupElement>(rightEndpoints[i - 1] + 1, (int)bitWidthIn) == left)
        {
            for (std::size_t j = 0; j < coeffNum; j++)
            {
                workspace.currentDcfOut[j] = workspace.nextDcfOut[j];
            }
        }
        else
        {
            dcfEval<GroupElement>(workspace.currentDcfOut, key, maskedX - 1 - left, seed, partyId, bitWidthIn,
                                  bitWidthOut, groupSize, &cache.dcfCache);
        }
        dcfEval<GroupElement>(workspace.nextDcfOut, key, maskedX - 1 - rightPrime, seed, partyId, bitWidthIn,
                              bitWidthOut, groupSize, &cache.dcfCache);

        GroupElement cx = static_cast<GroupElement>((maskedX > left) - (maskedX > rightPrime));
        // w{j} = (w{d,b}, ... , w{0,b}) = c{x,i} * beta{i,b} - s{i,b} + s{i+1,b} + e{i,b}
        for (std::size_t j = 0; j < coeffNum; ++j)
        {
            GroupElement betaI = sharedBeta[i * coeffNum + j];
            GroupElement sI    = workspace.currentDcfOut[i * coeffNum + j];
            GroupElement sI1   = workspace.nextDcfOut[i * coeffNum + j];
            GroupElement w     = cx * betaI - sI + sI1 + sharedE[i * coeffNum + j];

            workspace.coefficientScratch[j] = workspace.coefficientScratch[j] + w;
        }
    }

    return splineEvaluatePolynomial<GroupElement>(workspace.coefficientScratch, maskedX, degree, bitWidthOut);
}

} // namespace FastFss::impl

#endif
