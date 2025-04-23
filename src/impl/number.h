#pragma once
#ifndef SRC_IMPL_NUMBER_H
#define SRC_IMPL_NUMBER_H

#include "def.h"

#if defined(_MSC_VER) && !defined(__CUDACC__)
#include <intrin.h>
#endif

namespace FastFss::impl {

template <typename GroupElement>
FAST_FSS_DEVICE static inline GroupElement modBits(GroupElement x,
                                                   int bitWidth) noexcept
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

FAST_FSS_DEVICE static inline int parityU64(std::uint64_t x)
{
    std::uint64_t t = x;
    t ^= t >> 1;
    t ^= t >> 2;
    t = (t & 0x1111111111111111ULL) * 0x1111111111111111ULL;
    return (t >> 60) & 1;
}

template <typename GroupElement>
FAST_FSS_DEVICE static inline int clz(GroupElement x, std::size_t bitWidth)
{
    static_assert(sizeof(GroupElement) <= 8,
                  "clz is not supported for GroupElement larger than 64bit");

#if defined(__CUDACC__)
    if constexpr (sizeof(GroupElement) <= sizeof(std::uint32_t))
    {
        return __clz(static_cast<std::uint32_t>(x)) - (32 - bitWidth);
    }
    else
    {
        return __clzll(static_cast<std::uint64_t>(x)) - (64 - bitWidth);
    }
#elif defined(_MSC_VER)
    if constexpr (sizeof(GroupElement) <= sizeof(std::uint32_t))
    {
        return __lzcnt(static_cast<std::uint32_t>(x)) - (32 - bitWidth);
    }
    else
    {
        return __lzcnt64(static_cast<std::uint64_t>(x)) - (64 - bitWidth);
    }
#else
    if constexpr (sizeof(GroupElement) <= sizeof(std::uint32_t))
    {
        return __builtin_clz(static_cast<std::uint32_t>(x)) - (32 - bitWidth);
    }
    else
    {
        return __builtin_clzll(static_cast<std::uint64_t>(x)) - (64 - bitWidth);
    }
#endif
}

} // namespace FastFss::impl

#endif