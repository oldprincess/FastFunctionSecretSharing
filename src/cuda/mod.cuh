#pragma once
#ifndef SRC_FAST_FSS_CUDA_MOD_CUH
#define SRC_FAST_FSS_CUDA_MOD_CUD

namespace FastFss::cuda {

template <typename GroupElement>
static __device__ inline GroupElement mod_bits(GroupElement x,
                                               int          bitWidth) noexcept
{
    if (bitWidth == sizeof(GroupElement) * 8)
    {
        return x;
    }
    else
    {
        return x & (((GroupElement)1 << bitWidth) - 1);
    }
}

} // namespace FastFss::cuda

#endif