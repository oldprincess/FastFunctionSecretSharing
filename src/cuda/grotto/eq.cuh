#ifndef SRC_CUDA_GROTTO_EQ_CUH
#define SRC_CUDA_GROTTO_EQ_CUH

#include "def.cuh"

template <typename GroupElement>
__global__ static void grottoEqEvalKernel(void*       out,
                                          const void* maskedX,
                                          const void* key,
                                          const void* seed,
                                          int         partyId,
                                          size_t      bitWidthIn,
                                          size_t      elementNum,
                                          void*       cache)
{
    using namespace FastFss;

    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* maskedXPtr = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr    = (const std::uint8_t*)seed;
    GroupElement*       outPtr     = (GroupElement*)out;

    impl::GrottoKey<GroupElement>   keyObj;
    impl::GrottoCache<GroupElement> cacheObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::GrottoCache<GroupElement>* cacheObjPtr   = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj, cache, bitWidthIn, i, elementNum);
            cacheObjPtr = &cacheObj;
        }
        outPtr[i] = impl::grottoEqEval(keyObj,                    //
                                       maskedXPtr[maskedXOffset], //
                                       seedPtr + seedOffset,      //
                                       partyId,                   //
                                       bitWidthIn,                //
                                       cacheObjPtr                //
        );
    }
}

#endif