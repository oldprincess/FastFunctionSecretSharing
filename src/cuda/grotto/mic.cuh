#ifndef SRC_CUDA_GROTTO_MIC_CUH
#define SRC_CUDA_GROTTO_MIC_CUH

#include "def.cuh"

template <typename GroupElement>
__global__ static void grottoMICEvalParallelAllKernel(void       *out,
                                                      const void *maskedX,
                                                      const void *key,
                                                      const void *seed,
                                                      int         partyId,
                                                      const void *leftEndpoints,
                                                      const void *rightEndpoints,
                                                      size_t      intervalNum,
                                                      size_t      bitWidthIn,
                                                      size_t      elementNum)
{
    using namespace FastFss;

    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement *maskedXPtr        = (const GroupElement *)maskedX;
    const std::uint8_t *seedPtr           = (const std::uint8_t *)seed;
    GroupElement       *outPtr            = (GroupElement *)out;
    const GroupElement *leftEndpointsPtr  = (const GroupElement *)leftEndpoints;
    const GroupElement *rightEndpointsPtr = (const GroupElement *)rightEndpoints;

    impl::GrottoKey<GroupElement> keyObj;
    for (std::size_t i = idx; i < elementNum * intervalNum; i += stride)
    {
        std::size_t xIdx = i / intervalNum;
        std::size_t iIdx = i % intervalNum;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, xIdx, elementNum);

        GroupElement xP = maskedXPtr[xIdx] - (leftEndpointsPtr[iIdx] - 1);
        GroupElement xQ = maskedXPtr[xIdx] - rightEndpointsPtr[iIdx];

        xP = impl::modBits<GroupElement>(xP, bitWidthIn);
        xQ = impl::modBits<GroupElement>(xQ, bitWidthIn);

        GroupElement sp = impl::grottoEval<GroupElement>( //
            keyObj, xP, seedPtr + xIdx * 16, partyId, bitWidthIn, true, nullptr);
        GroupElement sq = impl::grottoEval<GroupElement>( //
            keyObj, xQ, seedPtr + xIdx * 16, partyId, bitWidthIn, true, nullptr);

        outPtr[i] = (sp ^ sq) ^ ((xQ >= xP) ? partyId : 0);
    }
}

#endif
