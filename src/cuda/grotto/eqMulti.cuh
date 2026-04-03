#ifndef SRC_CUDA_GROTTO_EQ_MULTI_CUH
#define SRC_CUDA_GROTTO_EQ_MULTI_CUH

#include "def.cuh"

template <typename GroupElement>
__global__ static void grottoEqMultiEvalParallelAllKernel(void       *out,
                                                          const void *maskedX,
                                                          const void *key,
                                                          const void *seed,
                                                          int         partyId,
                                                          const void *points,
                                                          size_t      pointsNum,
                                                          size_t      bitWidthIn,
                                                          size_t      elementNum)
{
    using namespace FastFss;

    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement *maskedXPtr = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr    = (const std::uint8_t *)seed;
    GroupElement       *outPtr     = (GroupElement *)out;
    const GroupElement *pointsPtr  = (const GroupElement *)points;

    impl::GrottoKey<GroupElement> keyObj;
    for (std::size_t i = idx; i < elementNum * pointsNum; i += stride)
    {
        std::size_t xIdx = i / pointsNum;
        std::size_t pIdx = i % pointsNum;

        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, xIdx, elementNum);

        GroupElement tmp = maskedXPtr[xIdx] - pointsPtr[pIdx];

        outPtr[i] = impl::grottoEqEval<GroupElement>(keyObj,              //
                                                     tmp,                 //
                                                     seedPtr + 16 * xIdx, //
                                                     partyId,             //
                                                     bitWidthIn,          //
                                                     nullptr              //

        );
    }
}

#endif
