#ifndef FAST_FSS_TESTS_COMMON_NBIT_SIGNED_HPP
#define FAST_FSS_TESTS_COMMON_NBIT_SIGNED_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>
#include <wideint/wideint.hpp>

namespace FastFss::tests {

/** Interpret @a v as a two's-complement signed integer in @a bitWidth bits (1..64). */
template <typename T>
inline std::int64_t asSignedNBitInt64(T v, std::size_t bitWidth)
{
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "unsigned integral only");
    if (bitWidth == 0)
    {
        return 0;
    }
    if (bitWidth >= 64)
    {
        return static_cast<std::int64_t>(static_cast<std::uint64_t>(v));
    }
    const unsigned      bw   = static_cast<unsigned>(bitWidth);
    const std::uint64_t mask = (1ull << bw) - 1ull;
    const std::uint64_t u    = static_cast<std::uint64_t>(v) & mask;
    const unsigned      sh   = 64u - bw;
    return static_cast<std::int64_t>(static_cast<std::int64_t>(u << sh) >> sh);
}

/** Sign-extend @a v to full 128-bit two's-complement bit pattern (as @c uint128_t). */
inline wideint::uint128_t signExtendToU128(wideint::uint128_t v, std::size_t bitWidth)
{
    if (bitWidth == 0)
    {
        return wideint::uint128_t{};
    }
    if (bitWidth >= 128)
    {
        return v;
    }
    const unsigned           bw   = static_cast<unsigned>(bitWidth);
    wideint::uint128_t       mask = (wideint::uint128_t{1} << bw) - wideint::uint128_t{1};
    wideint::uint128_t       u    = v & mask;
    const wideint::uint128_t sign = wideint::uint128_t{1} << (bw - 1U);
    if ((u & sign) != wideint::uint128_t{})
    {
        u |= ~mask;
    }
    return u;
}

/** Unsigned compare of two 128-bit patterns as signed (two's complement). */
inline bool u128SignedLess(wideint::uint128_t a, wideint::uint128_t b)
{
    const wideint::uint128_t flip = wideint::uint128_t{1} << 127;
    return ((a ^ flip) < (b ^ flip));
}

inline bool u128SignedLeq(wideint::uint128_t a, wideint::uint128_t b)
{
    const wideint::uint128_t flip = wideint::uint128_t{1} << 127;
    return ((a ^ flip) <= (b ^ flip));
}

template <typename T>
inline bool signedNBitLess(T a, T b, std::size_t bitWidth)
{
    if constexpr (std::is_same_v<T, wideint::uint128_t>)
    {
        const wideint::uint128_t sa = signExtendToU128(a, bitWidth);
        const wideint::uint128_t sb = signExtendToU128(b, bitWidth);
        return u128SignedLess(sa, sb);
    }
    else
    {
        return asSignedNBitInt64(a, bitWidth) < asSignedNBitInt64(b, bitWidth);
    }
}

template <typename T>
inline bool signedNBitLeq(T a, T b, std::size_t bitWidth)
{
    if constexpr (std::is_same_v<T, wideint::uint128_t>)
    {
        const wideint::uint128_t sa = signExtendToU128(a, bitWidth);
        const wideint::uint128_t sb = signExtendToU128(b, bitWidth);
        return u128SignedLeq(sa, sb);
    }
    else
    {
        return asSignedNBitInt64(a, bitWidth) <= asSignedNBitInt64(b, bitWidth);
    }
}

/** True iff @a x lies in [left, right] under signed @a bitWidth-bit interpretation. */
template <typename T>
inline bool signedNBitIntervalContains(T x, T left, T right, std::size_t bitWidth)
{
    return signedNBitLeq(left, x, bitWidth) && signedNBitLeq(x, right, bitWidth);
}

/** Store @a s as an unsigned @a bitWidth-bit two's-complement pattern in @c T. */
template <typename T>
inline T unsignedBitPatternFromInt64(std::int64_t s, std::size_t bitWidth)
{
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>, "");
    if (bitWidth == 0)
    {
        return T(0);
    }
    if (bitWidth >= 64)
    {
        return static_cast<T>(static_cast<std::uint64_t>(s));
    }
    const unsigned      bw   = static_cast<unsigned>(bitWidth);
    const std::uint64_t mask = (1ull << bw) - 1ull;
    return static_cast<T>(static_cast<std::uint64_t>(s) & mask);
}

/** Low 128 bits of a @c uint256_t as @c uint128_t (little-endian limbs). */
inline wideint::uint128_t uint256_low128_to_uint128(const wideint::uint256_t& v)
{
    return wideint::uint128_t(static_cast<std::uint64_t>(v), static_cast<std::uint64_t>(v >> 64));
}

// TODO: FIXME
/**
 * Partition the signed n-bit range into @a intervalCount contiguous disjoint intervals (full cover).
 * Endpoints are stored as unsigned n-bit patterns in @a left / @a right (same convention as PRNG outputs).
 */
template <typename T>
inline void fillSignedNBitFullPartition(std::vector<T>& left,
                                        std::vector<T>& right,
                                        std::size_t     bitWidthIn,
                                        std::size_t     intervalCount)
{
    left.resize(intervalCount);
    right.resize(intervalCount);
    if (intervalCount == 0)
    {
        return;
    }

    if constexpr (std::is_same_v<T, wideint::uint128_t>)
    {
        const unsigned           bw = static_cast<unsigned>(std::min<std::size_t>(bitWidthIn, 128u));
        const wideint::uint128_t n  = wideint::uint128_t(intervalCount);

        if (bw < 128)
        {
            const wideint::uint256_t domain256 = wideint::uint256_t{1} << bw;
            const wideint::uint256_t n256(intervalCount);
            const wideint::uint128_t umin = wideint::uint128_t{1} << (bw - 1U);
            const wideint::uint128_t mask = (wideint::uint128_t{1} << bw) - wideint::uint128_t{1};

            for (std::size_t k = 0; k < intervalCount; ++k)
            {
                const wideint::uint256_t loOff256 = (domain256 * wideint::uint256_t(k)) / n256;
                const wideint::uint256_t hiOff256 =
                    (domain256 * wideint::uint256_t(k + 1)) / n256 - wideint::uint256_t{1};

                const wideint::uint128_t loOff = uint256_low128_to_uint128(loOff256);
                const wideint::uint128_t hiOff = uint256_low128_to_uint128(hiOff256);

                left[k]  = (umin + loOff) & mask;
                right[k] = (umin + hiOff) & mask;
            }
        }
        else
        {
            const wideint::uint256_t domain  = wideint::uint256_t{1} << 128;
            const wideint::uint256_t umin256 = wideint::uint256_t{1} << 127;
            const wideint::uint256_t mask128 = (wideint::uint256_t{1} << 128) - wideint::uint256_t{1};
            const wideint::uint256_t n256(intervalCount);

            for (std::size_t k = 0; k < intervalCount; ++k)
            {
                const wideint::uint256_t loOff = (domain * wideint::uint256_t(k)) / n256;
                const wideint::uint256_t hiOff = (domain * wideint::uint256_t(k + 1)) / n256 - wideint::uint256_t{1};

                left[k]  = uint256_low128_to_uint128((umin256 + loOff) & mask128);
                right[k] = uint256_low128_to_uint128((umin256 + hiOff) & mask128);
            }
        }
    }
    else
    {
        const unsigned           bw     = static_cast<unsigned>(std::min<std::size_t>(bitWidthIn, 64u));
        const wideint::uint128_t domain = (bw < 64) ? (wideint::uint128_t{1} << bw) : (wideint::uint128_t{1} << 64);
        const wideint::uint128_t umin =
            (bw < 64) ? (wideint::uint128_t{1} << (bw - 1U)) : (wideint::uint128_t{1} << 63);
        const wideint::uint128_t mask = (bw < 64) ? ((wideint::uint128_t{1} << bw) - wideint::uint128_t{1})
                                                  : ((wideint::uint128_t{1} << 64) - wideint::uint128_t{1});

        const wideint::uint128_t n(intervalCount);

        for (std::size_t k = 0; k < intervalCount; ++k)
        {
            const wideint::uint128_t loOff = (domain * wideint::uint128_t(k)) / n;
            const wideint::uint128_t hiOff = (domain * wideint::uint128_t(k + 1)) / n - wideint::uint128_t{1};

            const std::uint64_t loU = static_cast<std::uint64_t>((umin + loOff) & mask);
            const std::uint64_t hiU = static_cast<std::uint64_t>((umin + hiOff) & mask);

            left[k]  = static_cast<T>(loU);
            right[k] = static_cast<T>(hiU);
        }
    }
}
} // namespace FastFss::tests

#endif
