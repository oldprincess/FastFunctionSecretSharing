#pragma once
#ifndef SRC_IMPL_NUMBER_H
#define SRC_IMPL_NUMBER_H

#include <type_traits>

#include "def.h"
#include "wideint/bit.hpp"

#if defined(_MSC_VER) && !defined(__CUDACC__)
#include <intrin.h>
#endif

namespace FastFss::impl {

template <typename T>
struct is_wideint : std::false_type
{
};

template <std::size_t N, bool Signed>
struct is_wideint<wideint::detail::basic_int<N, Signed>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_supported_group_element_v =
    (std::is_integral_v<T> && !std::is_same_v<std::remove_cv_t<T>, bool>) ||
    is_wideint<std::remove_cv_t<T>>::value;

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
        const auto width = static_cast<unsigned int>(bitWidth);
        return x & ((GroupElement(1) << width) - GroupElement(1));
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
    static_assert(is_supported_group_element_v<GroupElement>,
                  "clz requires an integer or wideint::basic_int GroupElement");
    if (x == 0)
    {
        return bitWidth;
    }

#if defined(__CUDACC__)
    if constexpr (sizeof(GroupElement) <= sizeof(std::uint32_t))
    {
        return __clz(static_cast<std::uint32_t>(x)) - (32 - bitWidth);
    }
    else if constexpr (sizeof(GroupElement) <= sizeof(std::uint64_t))
    {
        return __clzll(static_cast<std::uint64_t>(x)) - (64 - bitWidth);
    }
    else
    {
        return static_cast<int>(wideint::countl_zero(x)) -
               (static_cast<int>(sizeof(GroupElement) * 8) -
                static_cast<int>(bitWidth));
    }
#elif defined(_MSC_VER)
    if constexpr (sizeof(GroupElement) <= sizeof(std::uint32_t))
    {
        return __lzcnt(static_cast<std::uint32_t>(x)) - (32 - bitWidth);
    }
    else if constexpr (sizeof(GroupElement) <= sizeof(std::uint64_t))
    {
        return __lzcnt64(static_cast<std::uint64_t>(x)) - (64 - bitWidth);
    }
    else
    {
        return static_cast<int>(wideint::countl_zero(x)) -
               (static_cast<int>(sizeof(GroupElement) * 8) -
                static_cast<int>(bitWidth));
    }
#else
    if constexpr (sizeof(GroupElement) <= sizeof(std::uint32_t))
    {
        return __builtin_clz(static_cast<std::uint32_t>(x)) - (32 - bitWidth);
    }
    else if constexpr (sizeof(GroupElement) <= sizeof(std::uint64_t))
    {
        return __builtin_clzll(static_cast<std::uint64_t>(x)) - (64 - bitWidth);
    }
    else
    {
        return static_cast<int>(wideint::countl_zero(x)) -
               (static_cast<int>(sizeof(GroupElement) * 8) -
                static_cast<int>(bitWidth));
    }
#endif
}

} // namespace FastFss::impl

#endif
