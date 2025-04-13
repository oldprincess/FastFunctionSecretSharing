#pragma once
#ifndef SRC_FAST_FSS_CUDA_NUMBER_CUH
#define SRC_FAST_FSS_CUDA_NUMBER_CUH

#include <cstddef>
#include <cstdint>

namespace FastFss::cuda {

template <typename GroupElement>
__device__ static inline GroupElement mod_bits(GroupElement x,
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

__device__ static inline int parityU64(std::uint64_t x)
{
    std::uint64_t t = x;
    t ^= t >> 1;
    t ^= t >> 2;
    t = (t & 0x1111111111111111ULL) * 0x1111111111111111ULL;
    return (t >> 60) & 1;
}

template <typename GroupElement>
__device__ static inline int clz(GroupElement x, std::size_t bitWidth)
{
    static_assert(sizeof(GroupElement) <= 8,
                  "clz is not supported for GroupElement larger than 64bit");

    if constexpr (sizeof(GroupElement) <= sizeof(std::uint32_t))
    {
        return __clz(static_cast<std::uint32_t>(x)) - (32 - bitWidth);
    }
    else
    {
        return __clzll(static_cast<std::uint64_t>(x)) - (64 - bitWidth);
    }
}

} // namespace FastFss::cuda

#endif