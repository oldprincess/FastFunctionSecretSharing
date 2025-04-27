#pragma once
#ifndef SRC_FAST_FSS_CUDA_MIC_CUH
#define SRC_FAST_FSS_CUDA_MIC_CUH

#include "dcf.cuh"
#include "number.cuh"

namespace FastFss::cuda {

template <typename GroupElement>
__device__ inline void dcfMICKeyGen(DcfKey<GroupElement>& key,
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

    GroupElement MAX = mod_bits<GroupElement>((GroupElement)(-1), bitWidthIn);
    for (int i = 0; i < intervalNum; i++)
    {
        GroupElement qPrime      = rightBoundary[i] + 1;
        GroupElement alphaP      = leftBoundary[i] + alpha;
        GroupElement alphaQ      = rightBoundary[i] + alpha;
        GroupElement alphaQPrime = rightBoundary[i] + 1 + alpha;

        qPrime      = mod_bits<GroupElement>(qPrime, bitWidthIn);
        alphaP      = mod_bits<GroupElement>(alphaP, bitWidthIn);
        alphaQ      = mod_bits<GroupElement>(alphaQ, bitWidthIn);
        alphaQPrime = mod_bits<GroupElement>(alphaQPrime, bitWidthIn);

        z[i] = (alphaP > alphaQ) - (alphaP > leftBoundary[i]) +
               (alphaQPrime > qPrime) + (alphaQ == MAX);
    }
}

template <typename GroupElement>
__device__ inline void dcfMICEval(GroupElement* sharedOut, // intervalNum
                                  GroupElement  maskedX,
                                  const DcfKey<GroupElement>& key,
                                  const GroupElement* sharedZ, // intervalNum
                                  const void*         seed,
                                  int                 partyId,
                                  const GroupElement* leftBoundary,
                                  const GroupElement* rightBoundary,
                                  std::size_t         intervalNum,
                                  std::size_t         bitWidthIn,
                                  std::size_t         bitWidthOut)
{
    maskedX = mod_bits<GroupElement>(maskedX, bitWidthIn);

    GroupElement sp = 0, sq = 0;
    {
        GroupElement qPrime  = rightBoundary[0] + 1;
        GroupElement xP      = (maskedX - 1 - leftBoundary[0]);
        GroupElement xQPrime = (maskedX - 1 - qPrime);

        qPrime  = mod_bits<GroupElement>(qPrime, bitWidthIn);
        xP      = mod_bits<GroupElement>(xP, bitWidthIn);
        xQPrime = mod_bits<GroupElement>(xQPrime, bitWidthIn);

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

        privQAdd1 = mod_bits<GroupElement>(privQAdd1, bitWidthIn);
        qPrime    = mod_bits<GroupElement>(qPrime, bitWidthIn);
        xP        = mod_bits<GroupElement>(xP, bitWidthIn);
        xQPrime   = mod_bits<GroupElement>(xQPrime, bitWidthIn);

        // never divergent branches.
        // because leftBoundary is same for all cuda threads
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

template <typename GroupElement>
__global__ void dcfMICKeyGenKernel(void*       deviceKey,
                                   void*       deviceZ,
                                   const void* deviceAlpha,
                                   const void* deviceSeed0,
                                   const void* deviceSeed1,
                                   const void* leftBoundary,
                                   const void* rightBoundary,
                                   std::size_t intervalNum,
                                   std::size_t bitWidthIn,
                                   std::size_t bitWidthOut,
                                   std::size_t elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement*       deviceZPtr       = (GroupElement*)deviceZ;
    const GroupElement* deviceAlphaPtr   = (const GroupElement*)deviceAlpha;
    const std::uint8_t* deviceSeedPtr0   = (const std::uint8_t*)deviceSeed0;
    const std::uint8_t* deviceSeedPtr1   = (const std::uint8_t*)deviceSeed1;
    const GroupElement* leftBoundaryPtr  = (const GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr = (const GroupElement*)rightBoundary;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        DcfKey<GroupElement> dcfKey;
        dcfKeySetPtr(dcfKey, deviceKey, bitWidthIn, bitWidthOut, i, elementNum);

        std::size_t offsetSeed0 = i * 16;
        std::size_t offsetSeed1 = i * 16;
        std::size_t offsetZ     = i * intervalNum;

        dcfMICKeyGen(dcfKey, deviceZPtr + offsetZ, deviceAlphaPtr[i],
                     deviceSeedPtr0 + offsetSeed0, deviceSeedPtr1 + offsetSeed1,
                     leftBoundaryPtr, rightBoundaryPtr, intervalNum, bitWidthIn,
                     bitWidthOut);
    }
}

template <typename GroupElement>
__global__ void dcfMICEvalKernel(void*       deviceSharedOut,
                                 const void* deviceMaskedX,
                                 const void* deviceKey,
                                 const void* deviceSharedZ,
                                 const void* deviceSeed,
                                 int         partyId,
                                 const void* leftBoundary,
                                 const void* rightBoundary,
                                 std::size_t intervalNum,
                                 std::size_t bitWidthIn,
                                 std::size_t bitWidthOut,
                                 std::size_t elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement*       deviceSharedOutPtr = (GroupElement*)deviceSharedOut;
    const GroupElement* deviceMaskedXPtr   = (const GroupElement*)deviceMaskedX;
    const GroupElement* deviceSharedZPtr   = (const GroupElement*)deviceSharedZ;
    const std::uint8_t* deviceSeedPtr      = (const std::uint8_t*)deviceSeed;
    const GroupElement* leftBoundaryPtr    = (const GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr   = (const GroupElement*)rightBoundary;

    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        DcfKey<GroupElement> dcfKey;
        dcfKeySetPtr(dcfKey, deviceKey, bitWidthIn, bitWidthOut, i, elementNum);

        std::size_t offsetSharedOut = i * intervalNum;
        std::size_t offsetMaskedX   = i;
        std::size_t offsetSharedZ   = i * intervalNum;
        std::size_t offsetSeed      = i * 16;

        dcfMICEval(deviceSharedOutPtr + offsetSharedOut,
                   deviceMaskedXPtr[offsetMaskedX], dcfKey,
                   deviceSharedZPtr + offsetSharedZ, deviceSeedPtr + offsetSeed,
                   partyId, leftBoundaryPtr, rightBoundaryPtr, intervalNum,
                   bitWidthIn, bitWidthOut);
    }
}

// parallel all Element
template <typename GroupElement>
__global__ void dcfMICEvalKernel2(void*       deviceSharedOut,
                                  const void* deviceMaskedX,
                                  const void* deviceKey,
                                  const void* deviceSharedZ,
                                  const void* deviceSeed,
                                  int         partyId,
                                  const void* leftBoundary,
                                  const void* rightBoundary,
                                  std::size_t intervalNum,
                                  std::size_t bitWidthIn,
                                  std::size_t bitWidthOut,
                                  std::size_t elementNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    GroupElement*       deviceSharedOutPtr = (GroupElement*)deviceSharedOut;
    const GroupElement* deviceMaskedXPtr   = (const GroupElement*)deviceMaskedX;
    const GroupElement* deviceSharedZPtr   = (const GroupElement*)deviceSharedZ;
    const std::uint8_t* deviceSeedPtr      = (const std::uint8_t*)deviceSeed;
    const GroupElement* leftBoundaryPtr    = (const GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr   = (const GroupElement*)rightBoundary;

    for (std::size_t i = idx; i < elementNum * intervalNum; i += stride)
    {
        DcfKey<GroupElement> dcfKey;
        std::size_t          elementIdx  = i / intervalNum;
        std::size_t          intervalIdx = i % intervalNum;

        dcfKeySetPtr(dcfKey, deviceKey, bitWidthIn, bitWidthOut, elementIdx,
                     elementNum);

        std::size_t offsetMaskedX = elementIdx;
        std::size_t offsetSeed    = elementIdx * 16;
        std::size_t offsetOut     = elementIdx * intervalNum + intervalIdx;

        GroupElement maskedX = deviceMaskedXPtr[offsetMaskedX];
        maskedX              = mod_bits<GroupElement>(maskedX, bitWidthIn);

        GroupElement sp = 0, sq = 0;
        GroupElement qPrime  = rightBoundaryPtr[intervalIdx] + 1;
        GroupElement xP      = (maskedX - 1 - leftBoundaryPtr[intervalIdx]);
        GroupElement xQPrime = (maskedX - 1 - qPrime);

        qPrime  = mod_bits<GroupElement>(qPrime, bitWidthIn);
        xP      = mod_bits<GroupElement>(xP, bitWidthIn);
        xQPrime = mod_bits<GroupElement>(xQPrime, bitWidthIn);

        sp = dcfEval<GroupElement>(dcfKey, xP, deviceSeedPtr + offsetSeed,
                                   partyId, bitWidthIn, bitWidthOut);
        sq = dcfEval<GroupElement>(dcfKey, xQPrime, deviceSeedPtr + offsetSeed,
                                   partyId, bitWidthIn, bitWidthOut);

        deviceSharedOutPtr[offsetOut] = sq - sp + deviceSharedZPtr[offsetOut];
        if (partyId == 1)
        {
            deviceSharedOutPtr[offsetOut] +=
                (maskedX > leftBoundaryPtr[intervalIdx]) - (maskedX > qPrime);
        }
    }
}

// parallel all DCF:
// (1) GRID_SIZE * BLOCK_SIZE == INTERVAL_NUM * ELEMENT_NUM * 2
// (2) BLOCK_SIZE % (INTERVAL_NUM * 2) == 0 or
//     (INTERVAL_NUM * 2) % BLOCK_SIZE == 0
template <typename GroupElement>
__global__ void dcfMICEvalKernel3(void*       deviceSharedOut,
                                  const void* deviceMaskedX,
                                  const void* deviceKey,
                                  const void* deviceSharedZ,
                                  const void* deviceSeed,
                                  int         partyId,
                                  const void* leftBoundary,
                                  const void* rightBoundary,
                                  std::size_t intervalNum,
                                  std::size_t bitWidthIn,
                                  std::size_t bitWidthOut,
                                  std::size_t elementNum)
{
    std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    GroupElement*       deviceSharedOutPtr = (GroupElement*)deviceSharedOut;
    const GroupElement* deviceMaskedXPtr   = (const GroupElement*)deviceMaskedX;
    const GroupElement* deviceSharedZPtr   = (const GroupElement*)deviceSharedZ;
    const std::uint8_t* deviceSeedPtr      = (const std::uint8_t*)deviceSeed;
    const GroupElement* leftBoundaryPtr    = (const GroupElement*)leftBoundary;
    const GroupElement* rightBoundaryPtr   = (const GroupElement*)rightBoundary;

    // elementIdx, intervalIdx, boundIdx
    std::size_t i = idx;

    DcfKey<GroupElement> dcfKey;

    std::size_t boundIdx    = i % 2;
    std::size_t elementIdx  = i / (intervalNum * 2);
    std::size_t intervalIdx = (i - elementIdx * (intervalNum * 2)) / 2;

    dcfKeySetPtr(dcfKey, deviceKey, bitWidthIn, bitWidthOut, elementIdx,
                 elementNum);

    std::size_t offsetMaskedX = elementIdx;
    std::size_t offsetSeed    = elementIdx * 16;
    std::size_t offsetOut     = elementIdx * intervalNum + intervalIdx;

    GroupElement maskedX = deviceMaskedXPtr[offsetMaskedX];
    maskedX              = mod_bits<GroupElement>(maskedX, bitWidthIn);

    GroupElement sp = 0, sq = 0;
    GroupElement qPrime  = rightBoundaryPtr[intervalIdx] + 1;
    GroupElement xP      = (maskedX - 1 - leftBoundaryPtr[intervalIdx]);
    GroupElement xQPrime = (maskedX - 1 - qPrime);

    qPrime  = mod_bits<GroupElement>(qPrime, bitWidthIn);
    xP      = mod_bits<GroupElement>(xP, bitWidthIn);
    xQPrime = mod_bits<GroupElement>(xQPrime, bitWidthIn);

    GroupElement maskedInput = 0;
    GroupElement sharedOut;
    if (boundIdx == 0)
    {
        maskedInput = xP;
    }
    else
    {
        maskedInput = xQPrime;
    }

    sharedOut = dcfEval<GroupElement>( //
        dcfKey, maskedInput, deviceSeedPtr + offsetSeed, partyId, bitWidthIn,
        bitWidthOut);

    if (boundIdx == 0)
    {
        deviceSharedOutPtr[offsetOut] = sharedOut;
    }

    __syncthreads();

    if (boundIdx == 1)
    {
        sp = deviceSharedOutPtr[offsetOut];
        sq = sharedOut;

        deviceSharedOutPtr[offsetOut] = sq - sp + deviceSharedZPtr[offsetOut];
        if (partyId == 1)
        {
            deviceSharedOutPtr[offsetOut] +=
                (maskedX > leftBoundaryPtr[intervalIdx]) - (maskedX > qPrime);
        }
    }
}

} // namespace FastFss::cuda

#endif