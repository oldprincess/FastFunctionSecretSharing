#ifndef SRC_CUDA_GROTTO_LUT_CUH
#define SRC_CUDA_GROTTO_LUT_CUH

#include "def.cuh"

template <typename GroupElement>
__global__ static void grottoLutEvalKernel(void*       outE,
                                           void*       outT,
                                           const void* maskedX,
                                           const void* key,
                                           const void* seed,
                                           int         partyId,
                                           const void* lookUpTable,
                                           size_t      lutNum,
                                           size_t      bitWidthIn,
                                           size_t      elementNum,
                                           void*       cache)
{
    using namespace FastFss;

    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* maskedXPtr     = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr        = (const std::uint8_t*)seed;
    GroupElement*       outEPtr        = (GroupElement*)outE;
    GroupElement*       outTPtr        = (GroupElement*)outT;
    const GroupElement* lookUpTablePtr = (GroupElement*)lookUpTable;

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
        impl::grottoLutEval(           //
            outEPtr + i,               //
            outTPtr + i * lutNum,      //
            keyObj,                    //
            maskedXPtr[maskedXOffset], //
            seedPtr + seedOffset,      //
            partyId,                   //
            lookUpTablePtr,            //
            lutNum,                    //
            bitWidthIn,                //
            cacheObjPtr                //
        );
    }
}

#endif