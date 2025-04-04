#pragma once
#ifndef SRC_FAST_FSS_CPU_MOD_HPP
#define SRC_FAST_FSS_CPU_MOD_HPP

namespace FastFss::cpu {

template <typename GroupElement>
static inline GroupElement mod_bits(GroupElement x, int bitWidth) noexcept
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
} // namespace FastFss::cpu

#endif