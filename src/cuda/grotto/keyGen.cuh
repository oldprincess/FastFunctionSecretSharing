#ifndef SRC_CUDA_GROTTO_EVAL_KEY_GEN_CUH
#define SRC_CUDA_GROTTO_EVAL_KEY_GEN_CUH

#include "def.cuh"

template <typename GroupElement>
__global__ static void grottoKeyGenKernel(void*       key,
                                          const void* alpha,
                                          const void* seed0,
                                          const void* seed1,
                                          size_t      bitWidthIn,
                                          size_t      elementNum)
{
    using namespace FastFss;

    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    const GroupElement* alphaPtr = (const GroupElement*)alpha;
    const std::uint8_t* seed0Ptr = (const std::uint8_t*)seed0;
    const std::uint8_t* seed1Ptr = (const std::uint8_t*)seed1;

    impl::GrottoKey<GroupElement> keyObj;
    for (std::size_t i = idx; i < elementNum; i += stride)
    {
        std::size_t alphaOffset = i;
        std::size_t seed0Offset = 16 * i;
        std::size_t seed1Offset = 16 * i;
        impl::grottoKeySetPtr(keyObj, key, bitWidthIn, i, elementNum);
        impl::grottoKeyGen(keyObj,                 //
                           alphaPtr[alphaOffset],  //
                           seed0Ptr + seed0Offset, //
                           seed1Ptr + seed1Offset, //
                           bitWidthIn);            //
    }
}

#endif