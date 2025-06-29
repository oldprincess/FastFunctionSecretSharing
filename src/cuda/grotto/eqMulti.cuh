#ifndef SRC_CUDA_GROTTO_EQ_MULTI_CUH
#define SRC_CUDA_GROTTO_EQ_MULTI_CUH

#include "def.cuh"

template <typename GroupElement>
__global__ static void grottoEqMultiEvalKernel(void*       out,
                                               const void* maskedX,
                                               const void* key,
                                               const void* seed,
                                               int         partyId,
                                               const void* points,
                                               size_t      pointsNum,
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
    const GroupElement* pointsPtr  = (const GroupElement*)points;

    impl::AES128GlobalContext       aesCtx;
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
        for (std::size_t j = 0; j < pointsNum; j++)
        {
            GroupElement tmp = maskedXPtr[maskedXOffset] - pointsPtr[j];
            outPtr[i * pointsNum + j] =
                impl::grottoEqEval(keyObj,               //
                                   tmp,                  //
                                   seedPtr + seedOffset, //
                                   partyId,              //
                                   bitWidthIn,           //
                                   cacheObjPtr,          //
                                   &aesCtx               //
                );
        }
    }
}

template <typename GroupElement>
__global__ static void grottoEqMultiEvalParallelAllKernel(void*       out,
                                                          const void* maskedX,
                                                          const void* key,
                                                          const void* seed,
                                                          int         partyId,
                                                          const void* points,
                                                          size_t      pointsNum,
                                                          size_t bitWidthIn,
                                                          size_t elementNum)
{
    using namespace FastFss;

    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* maskedXPtr = (const GroupElement*)maskedX;
    const std::uint8_t* seedPtr    = (const std::uint8_t*)seed;
    GroupElement*       outPtr     = (GroupElement*)out;
    const GroupElement* pointsPtr  = (const GroupElement*)points;

    impl::AES128GlobalContext     aesCtx;
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
                                                     nullptr,             //
                                                     &aesCtx              //
        );
    }
}

#endif