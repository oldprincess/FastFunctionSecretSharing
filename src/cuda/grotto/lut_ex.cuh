#ifndef SRC_CUDA_GROTTO_LUT_EX_CUH
#define SRC_CUDA_GROTTO_LUT_EX_CUH

#include "def.cuh"

template <typename GroupElement>
__global__ static void grottoLutEvalKernel_ex(void*       outE,
                                              void*       outT,
                                              const void* maskedX,
                                              const void* key,
                                              const void* seed,
                                              int         partyId,
                                              const void* lookUpTable,
                                              size_t      lutNum,
                                              size_t      lutBitWidth,
                                              size_t      bitWidthIn,
                                              size_t      elementNum,
                                              void*       cache0,
                                              void*       cache1)
{
    using namespace FastFss;

    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* maskedXPtr     = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr        = (const std::uint8_t*)seed;
    GroupElement*       outEPtr        = (GroupElement*)outE;
    GroupElement*       outTPtr        = (GroupElement*)outT;
    const GroupElement* lookUpTablePtr = (GroupElement*)lookUpTable;

    impl::AES128GlobalContext       aesCtx;
    impl::GrottoKey<GroupElement>   keyObj;
    impl::GrottoCache<GroupElement> cacheObj0;
    impl::GrottoCache<GroupElement> cacheObj1;

    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        impl::GrottoCache<GroupElement>* cacheObjPtr0  = nullptr;
        impl::GrottoCache<GroupElement>* cacheObjPtr1  = nullptr;
        std::size_t                      maskedXOffset = i;
        std::size_t                      seedOffset    = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        if (cache0 != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj0, cache0, bitWidthIn, i,
                                    elementNum);
            cacheObjPtr0 = &cacheObj0;
        }
        if (cache1 != nullptr)
        {
            impl::grottoCacheSetPtr(cacheObj1, cache1, bitWidthIn, i,
                                    elementNum);
            cacheObjPtr1 = &cacheObj1;
        }
        if (cache0 != nullptr && cache1 != nullptr)
        {
            GroupElement v           = maskedXPtr[maskedXOffset];
            std::size_t  num         = (std::size_t)1 << lutBitWidth;
            GroupElement lowerMask   = (GroupElement)(num - 1);
            GroupElement higherMask  = ~(GroupElement)(num - 1);
            GroupElement higherPart0 = v & higherMask;
            GroupElement higherPart1 = (v - num) & higherMask;
            GroupElement lowerPart   = v & lowerMask;

            impl::grottoEqEval(keyObj, higherPart0, seedPtr + seedOffset,
                               partyId, bitWidthIn, cacheObjPtr0, &aesCtx);

            cacheObjPtr1->preMaskedX = cacheObjPtr0->preMaskedX;
            cacheObjPtr1->preTo      = cacheObjPtr0->preTo;
            for (std::size_t j = 0; j < bitWidthIn - 6; j++)
            {
                cacheObjPtr1->stCache[j][0] = cacheObjPtr0->stCache[j][0];
                cacheObjPtr1->stCache[j][1] = cacheObjPtr0->stCache[j][1];
            }

            impl::grottoEqEval(keyObj,
                               (GroupElement)(higherPart1 | (lowerPart + 1)),
                               seedPtr + seedOffset, partyId, bitWidthIn,
                               cacheObjPtr1, &aesCtx);
        }
        impl::grottoLutEval_ex(        //
            outEPtr + i,               //
            outTPtr + i * lutNum,      //
            keyObj,                    //
            maskedXPtr[maskedXOffset], //
            seedPtr + seedOffset,      //
            partyId,                   //
            lookUpTablePtr,            //
            lutNum,                    //
            lutBitWidth,               //
            bitWidthIn,                //
            cacheObjPtr0,              //
            cacheObjPtr1,              //
            &aesCtx                    //
        );
    }
}

#endif