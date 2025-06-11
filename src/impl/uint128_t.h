
/*
uint128_t.h
An unsigned 128 bit integer type for C++

Copyright (c) 2013 - 2017 Jason Lee @ calccrypto at gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

With much help from Auston Sterling

Thanks to Stefan Deigmüller for finding
a bug in operator*.

Thanks to François Dessenne for convincing me
to do a general rewrite of this class.
*/

#ifndef SRC_IMPL_UINT128_T_H
#define SRC_IMPL_UINT128_T_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace FastFss::impl {

#if defined(__CUDACC__)

#define UINT128_DEVICE __device__ inline

#else

#define UINT128_DEVICE inline

#endif

#if defined(__BIG_ENDIAN__)
#elif defined(__LITTLE_ENDIAN__)
#else
#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN ||                 \
    defined(__BIG_ENDIAN__) || defined(__ARMEB__) || defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) ||       \
    defined(__MIBSEB__)
#ifndef __BIG_ENDIAN__
#define __BIG_ENDIAN__
#endif
#elif defined(__BYTE_ORDER) && __BYTE_ORDER == __LITTLE_ENDIAN ||         \
    defined(__LITTLE_ENDIAN__) || defined(__ARMEL__) ||                   \
    defined(__THUMBEL__) || defined(__AARCH64EL__) || defined(_MIPSEL) || \
    defined(__MIPSEL) || defined(__MIPSEL__) || defined(_WIN32) ||        \
    defined(__i386__) || defined(__x86_64__) || defined(_X86_) ||         \
    defined(_IA64_)
#ifndef __LITTLE_ENDIAN__
#define __LITTLE_ENDIAN__
#else
#error "I don't know what architecture this is!"
#endif
#endif
#endif

class uint128_t
{
public:
#ifdef __BIG_ENDIAN__
    uint64_t UPPER, LOWER;
#endif
#ifdef __LITTLE_ENDIAN__
    uint64_t LOWER, UPPER;
#endif

public:
    // Constructors
    uint128_t()                     = default;
    uint128_t(const uint128_t &rhs) = default;
    uint128_t(uint128_t &&rhs)      = default;

    UINT128_DEVICE uint128_t(const bool &b);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t(const T &rhs)
#ifdef __BIG_ENDIAN__
        : UPPER(0),
          LOWER(rhs)
#endif
#ifdef __LITTLE_ENDIAN__
        : LOWER(rhs), UPPER(0)
#endif
    {
        if constexpr (std::is_signed<T>::value)
        {
            if (rhs < 0)
            {
                UPPER = (uint64_t)(-1);
            }
        }
    }

    template <typename S,
              typename T,
              typename = typename std::enable_if<std::is_integral<S>::value &&
                                                     std::is_integral<T>::value,
                                                 void>::type>
    UINT128_DEVICE uint128_t(const S &upper_rhs, const T &lower_rhs)
#ifdef __BIG_ENDIAN__
        : UPPER(upper_rhs),
          LOWER(lower_rhs)
#endif
#ifdef __LITTLE_ENDIAN__
        : LOWER(lower_rhs), UPPER(upper_rhs)
#endif
    {
    }

    //  RHS input args only

    // Assignment Operator
    uint128_t &operator=(const uint128_t &rhs) = default;
    uint128_t &operator=(uint128_t &&rhs)      = default;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator=(const T &rhs)
    {
        UPPER = 0;

        if (std::is_signed<T>::value)
        {
            if (rhs < 0)
            {
                UPPER = (uint64_t)(-1);
            }
        }

        LOWER = rhs;
        return *this;
    }

    UINT128_DEVICE uint128_t &operator=(const bool &rhs);

    // Typecast Operators
    UINT128_DEVICE operator bool() const;
    UINT128_DEVICE operator uint8_t() const;
    UINT128_DEVICE operator uint16_t() const;
    UINT128_DEVICE operator uint32_t() const;
    UINT128_DEVICE operator uint64_t() const;

    UINT128_DEVICE operator int() const
    {
        return (int)LOWER;
    }
    UINT128_DEVICE operator long() const
    {
        return (long)LOWER;
    }
    UINT128_DEVICE operator long long() const
    {
        return (long long)LOWER;
    }

    // Bitwise Operators
    UINT128_DEVICE uint128_t operator&(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t operator&(const T &rhs) const
    {
        return uint128_t(0, LOWER & (uint64_t)rhs);
    }

    UINT128_DEVICE uint128_t &operator&=(const uint128_t &rhs);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator&=(const T &rhs)
    {
        UPPER = 0;
        LOWER &= rhs;
        return *this;
    }

    UINT128_DEVICE uint128_t operator|(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t operator|(const T &rhs) const
    {
        return uint128_t(UPPER, LOWER | (uint64_t)rhs);
    }

    UINT128_DEVICE uint128_t &operator|=(const uint128_t &rhs);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator|=(const T &rhs)
    {
        LOWER |= (uint64_t)rhs;
        return *this;
    }

    UINT128_DEVICE uint128_t operator^(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t operator^(const T &rhs) const
    {
        return uint128_t(UPPER, LOWER ^ (uint64_t)rhs);
    }

    UINT128_DEVICE uint128_t &operator^=(const uint128_t &rhs);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator^=(const T &rhs)
    {
        LOWER ^= (uint64_t)rhs;
        return *this;
    }

    UINT128_DEVICE uint128_t operator~() const;

    // Bit Shift Operators
    UINT128_DEVICE uint128_t operator<<(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t operator<<(const T &rhs) const
    {
        return *this << uint128_t(rhs);
    }

    UINT128_DEVICE uint128_t &operator<<=(const uint128_t &rhs);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator<<=(const T &rhs)
    {
        *this = *this << uint128_t(rhs);
        return *this;
    }

    UINT128_DEVICE uint128_t operator>>(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t operator>>(const T &rhs) const
    {
        return *this >> uint128_t(rhs);
    }

    UINT128_DEVICE uint128_t &operator>>=(const uint128_t &rhs);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator>>=(const T &rhs)
    {
        *this = *this >> uint128_t(rhs);
        return *this;
    }

    // Logical Operators
    UINT128_DEVICE bool operator!() const;
    UINT128_DEVICE bool operator&&(const uint128_t &rhs) const;
    UINT128_DEVICE bool operator||(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE bool operator&&(const T &rhs) const
    {
        return static_cast<bool>(*this && rhs);
    }

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE bool operator||(const T &rhs) const
    {
        return static_cast<bool>(*this || rhs);
    }

    // Comparison Operators
    UINT128_DEVICE bool operator==(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE bool operator==(const T &rhs) const
    {
        return (!UPPER && (LOWER == (uint64_t)rhs));
    }

    UINT128_DEVICE bool operator!=(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE bool operator!=(const T &rhs) const
    {
        return (UPPER | (LOWER != (uint64_t)rhs));
    }

    UINT128_DEVICE bool operator>(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE bool operator>(const T &rhs) const
    {
        return (UPPER || (LOWER > (uint64_t)rhs));
    }

    UINT128_DEVICE bool operator<(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE bool operator<(const T &rhs) const
    {
        return (!UPPER) ? (LOWER < (uint64_t)rhs) : false;
    }

    UINT128_DEVICE bool operator>=(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE bool operator>=(const T &rhs) const
    {
        return ((*this > rhs) | (*this == rhs));
    }

    UINT128_DEVICE bool operator<=(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE bool operator<=(const T &rhs) const
    {
        return ((*this < rhs) | (*this == rhs));
    }

    // Arithmetic Operators
    UINT128_DEVICE uint128_t operator+(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t operator+(const T &rhs) const
    {
        return uint128_t(UPPER + ((LOWER + (uint64_t)rhs) < LOWER),
                         LOWER + (uint64_t)rhs);
    }

    UINT128_DEVICE uint128_t &operator+=(const uint128_t &rhs);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator+=(const T &rhs)
    {
        return *this += uint128_t(rhs);
    }

    UINT128_DEVICE uint128_t operator-(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t operator-(const T &rhs) const
    {
        return uint128_t((uint64_t)(UPPER - ((LOWER - rhs) > LOWER)),
                         (uint64_t)(LOWER - rhs));
    }

    UINT128_DEVICE uint128_t &operator-=(const uint128_t &rhs);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator-=(const T &rhs)
    {
        return *this = *this - uint128_t(rhs);
    }

    UINT128_DEVICE uint128_t operator*(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t operator*(const T &rhs) const
    {
        return *this * uint128_t(rhs);
    }

    UINT128_DEVICE uint128_t &operator*=(const uint128_t &rhs);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator*=(const T &rhs)
    {
        return *this = *this * uint128_t(rhs);
    }

private:
    UINT128_DEVICE std::pair<uint128_t, uint128_t> divmod(
        const uint128_t &lhs,
        const uint128_t &rhs) const;

public:
    UINT128_DEVICE uint128_t operator/(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t operator/(const T &rhs) const
    {
        return *this / uint128_t(rhs);
    }

    UINT128_DEVICE uint128_t &operator/=(const uint128_t &rhs);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator/=(const T &rhs)
    {
        return *this = *this / uint128_t(rhs);
    }

    UINT128_DEVICE uint128_t operator%(const uint128_t &rhs) const;

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t operator%(const T &rhs) const
    {
        return *this % uint128_t(rhs);
    }

    UINT128_DEVICE uint128_t &operator%=(const uint128_t &rhs);

    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
    UINT128_DEVICE uint128_t &operator%=(const T &rhs)
    {
        return *this = *this % uint128_t(rhs);
    }

    // Increment Operator
    UINT128_DEVICE uint128_t &operator++();
    UINT128_DEVICE uint128_t  operator++(int);

    // Decrement Operator
    UINT128_DEVICE uint128_t &operator--();
    UINT128_DEVICE uint128_t  operator--(int);

    // Nothing done since promotion doesn't work here
    UINT128_DEVICE uint128_t operator+() const;

    // two's complement
    UINT128_DEVICE uint128_t operator-() const;

    // Get private values
    UINT128_DEVICE const uint64_t &upper() const;
    UINT128_DEVICE const uint64_t &lower() const;

    // Get bitsize of value
    UINT128_DEVICE uint8_t bits() const;
};

// lhs type T as first arguemnt
// If the output is not a bool, casts to type T

// Bitwise Operators
template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE uint128_t operator&(const T &lhs, const uint128_t &rhs)
{
    return rhs & lhs;
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE T &operator&=(T &lhs, const uint128_t &rhs)
{
    return lhs = static_cast<T>(rhs & lhs);
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE uint128_t operator|(const T &lhs, const uint128_t &rhs)
{
    return rhs | lhs;
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE T &operator|=(T &lhs, const uint128_t &rhs)
{
    return lhs = static_cast<T>(rhs | lhs);
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE uint128_t operator^(const T &lhs, const uint128_t &rhs)
{
    return rhs ^ lhs;
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE T &operator^=(T &lhs, const uint128_t &rhs)
{
    return lhs = static_cast<T>(rhs ^ lhs);
}

// Bitshift operators
UINT128_DEVICE uint128_t operator<<(const bool &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator<<(const uint8_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator<<(const uint16_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator<<(const uint32_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator<<(const uint64_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator<<(const int8_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator<<(const int16_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator<<(const int32_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator<<(const int64_t &lhs, const uint128_t &rhs);

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE T &operator<<=(T &lhs, const uint128_t &rhs)
{
    return lhs = static_cast<T>(uint128_t(lhs) << rhs);
}

UINT128_DEVICE uint128_t operator>>(const bool &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator>>(const uint8_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator>>(const uint16_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator>>(const uint32_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator>>(const uint64_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator>>(const int8_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator>>(const int16_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator>>(const int32_t &lhs, const uint128_t &rhs);
UINT128_DEVICE uint128_t operator>>(const int64_t &lhs, const uint128_t &rhs);

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE T &operator>>=(T &lhs, const uint128_t &rhs)
{
    return lhs = static_cast<T>(uint128_t(lhs) >> rhs);
}

// Comparison Operators
template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE bool operator==(const T &lhs, const uint128_t &rhs)
{
    return (!rhs.upper() && ((uint64_t)lhs == rhs.lower()));
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE bool operator!=(const T &lhs, const uint128_t &rhs)
{
    return (rhs.upper() | ((uint64_t)lhs != rhs.lower()));
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE bool operator>(const T &lhs, const uint128_t &rhs)
{
    return (!rhs.upper()) && ((uint64_t)lhs > rhs.lower());
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE bool operator<(const T &lhs, const uint128_t &rhs)
{
    if (rhs.upper())
    {
        return true;
    }
    return ((uint64_t)lhs < rhs.lower());
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE bool operator>=(const T &lhs, const uint128_t &rhs)
{
    if (rhs.upper())
    {
        return false;
    }
    return ((uint64_t)lhs >= rhs.lower());
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE bool operator<=(const T &lhs, const uint128_t &rhs)
{
    if (rhs.upper())
    {
        return true;
    }
    return ((uint64_t)lhs <= rhs.lower());
}

// Arithmetic Operators
template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE uint128_t operator+(const T &lhs, const uint128_t &rhs)
{
    return rhs + lhs;
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE T &operator+=(T &lhs, const uint128_t &rhs)
{
    return lhs = static_cast<T>(rhs + lhs);
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE uint128_t operator-(const T &lhs, const uint128_t &rhs)
{
    return -(rhs - lhs);
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE T &operator-=(T &lhs, const uint128_t &rhs)
{
    return lhs = static_cast<T>(-(rhs - lhs));
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE uint128_t operator*(const T &lhs, const uint128_t &rhs)
{
    return rhs * lhs;
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE T &operator*=(T &lhs, const uint128_t &rhs)
{
    return lhs = static_cast<T>(rhs * lhs);
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE uint128_t operator/(const T &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) / rhs;
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE T &operator/=(T &lhs, const uint128_t &rhs)
{
    return lhs = static_cast<T>(uint128_t(lhs) / rhs);
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE uint128_t operator%(const T &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) % rhs;
}

template <
    typename T,
    typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
UINT128_DEVICE T &operator%=(T &lhs, const uint128_t &rhs)
{
    return lhs = static_cast<T>(uint128_t(lhs) % rhs);
}

UINT128_DEVICE uint128_t::uint128_t(const bool &b) : uint128_t((uint8_t)b)
{
}

UINT128_DEVICE uint128_t &uint128_t::operator=(const bool &rhs)
{
    UPPER = 0;
    LOWER = rhs;
    return *this;
}

UINT128_DEVICE uint128_t::operator bool() const
{
    return (bool)(UPPER | LOWER);
}

UINT128_DEVICE uint128_t::operator uint8_t() const
{
    return (uint8_t)LOWER;
}

UINT128_DEVICE uint128_t::operator uint16_t() const
{
    return (uint16_t)LOWER;
}

UINT128_DEVICE uint128_t::operator uint32_t() const
{
    return (uint32_t)LOWER;
}

UINT128_DEVICE uint128_t::operator uint64_t() const
{
    return (uint64_t)LOWER;
}

UINT128_DEVICE uint128_t uint128_t::operator&(const uint128_t &rhs) const
{
    return uint128_t(UPPER & rhs.UPPER, LOWER & rhs.LOWER);
}

UINT128_DEVICE uint128_t &uint128_t::operator&=(const uint128_t &rhs)
{
    UPPER &= rhs.UPPER;
    LOWER &= rhs.LOWER;
    return *this;
}

UINT128_DEVICE uint128_t uint128_t::operator|(const uint128_t &rhs) const
{
    return uint128_t(UPPER | rhs.UPPER, LOWER | rhs.LOWER);
}

UINT128_DEVICE uint128_t &uint128_t::operator|=(const uint128_t &rhs)
{
    UPPER |= rhs.UPPER;
    LOWER |= rhs.LOWER;
    return *this;
}

UINT128_DEVICE uint128_t uint128_t::operator^(const uint128_t &rhs) const
{
    return uint128_t(UPPER ^ rhs.UPPER, LOWER ^ rhs.LOWER);
}

UINT128_DEVICE uint128_t &uint128_t::operator^=(const uint128_t &rhs)
{
    UPPER ^= rhs.UPPER;
    LOWER ^= rhs.LOWER;
    return *this;
}

UINT128_DEVICE uint128_t uint128_t::operator~() const
{
    return uint128_t(~UPPER, ~LOWER);
}

UINT128_DEVICE uint128_t uint128_t::operator<<(const uint128_t &rhs) const
{
    const uint64_t shift = rhs.LOWER;
    if (((bool)rhs.UPPER) || (shift >= 128))
    {
        return uint128_t(0, 0);
    }
    else if (shift == 64)
    {
        return uint128_t(LOWER, 0);
    }
    else if (shift == 0)
    {
        return *this;
    }
    else if (shift < 64)
    {
        return uint128_t((UPPER << shift) + (LOWER >> (64 - shift)),
                         LOWER << shift);
    }
    else if ((128 > shift) && (shift > 64))
    {
        return uint128_t(LOWER << (shift - 64), 0);
    }
    else
    {
        return uint128_t(0, 0);
    }
}

UINT128_DEVICE uint128_t &uint128_t::operator<<=(const uint128_t &rhs)
{
    *this = *this << rhs;
    return *this;
}

UINT128_DEVICE uint128_t uint128_t::operator>>(const uint128_t &rhs) const
{
    const uint64_t shift = rhs.LOWER;
    if (((bool)rhs.UPPER) || (shift >= 128))
    {
        return uint128_t(0, 0);
    }
    else if (shift == 64)
    {
        return uint128_t(0, UPPER);
    }
    else if (shift == 0)
    {
        return *this;
    }
    else if (shift < 64)
    {
        return uint128_t(UPPER >> shift,
                         (UPPER << (64 - shift)) + (LOWER >> shift));
    }
    else if ((128 > shift) && (shift > 64))
    {
        return uint128_t(0, (UPPER >> (shift - 64)));
    }
    else
    {
        return uint128_t(0, 0);
    }
}

UINT128_DEVICE uint128_t &uint128_t::operator>>=(const uint128_t &rhs)
{
    *this = *this >> rhs;
    return *this;
}

UINT128_DEVICE bool uint128_t::operator!() const
{
    return !(bool)(UPPER | LOWER);
}

UINT128_DEVICE bool uint128_t::operator&&(const uint128_t &rhs) const
{
    return ((bool)*this && rhs);
}

UINT128_DEVICE bool uint128_t::operator||(const uint128_t &rhs) const
{
    return ((bool)*this || rhs);
}

UINT128_DEVICE bool uint128_t::operator==(const uint128_t &rhs) const
{
    return ((UPPER == rhs.UPPER) && (LOWER == rhs.LOWER));
}

UINT128_DEVICE bool uint128_t::operator!=(const uint128_t &rhs) const
{
    return ((UPPER != rhs.UPPER) | (LOWER != rhs.LOWER));
}

UINT128_DEVICE bool uint128_t::operator>(const uint128_t &rhs) const
{
    if (UPPER == rhs.UPPER)
    {
        return (LOWER > rhs.LOWER);
    }
    return (UPPER > rhs.UPPER);
}

UINT128_DEVICE bool uint128_t::operator<(const uint128_t &rhs) const
{
    if (UPPER == rhs.UPPER)
    {
        return (LOWER < rhs.LOWER);
    }
    return (UPPER < rhs.UPPER);
}

UINT128_DEVICE bool uint128_t::operator>=(const uint128_t &rhs) const
{
    return ((*this > rhs) | (*this == rhs));
}

UINT128_DEVICE bool uint128_t::operator<=(const uint128_t &rhs) const
{
    return ((*this < rhs) | (*this == rhs));
}

UINT128_DEVICE uint128_t uint128_t::operator+(const uint128_t &rhs) const
{
    return uint128_t(UPPER + rhs.UPPER + ((LOWER + rhs.LOWER) < LOWER),
                     LOWER + rhs.LOWER);
}

UINT128_DEVICE uint128_t &uint128_t::operator+=(const uint128_t &rhs)
{
    UPPER += rhs.UPPER + ((LOWER + rhs.LOWER) < LOWER);
    LOWER += rhs.LOWER;
    return *this;
}

UINT128_DEVICE uint128_t uint128_t::operator-(const uint128_t &rhs) const
{
    return uint128_t(UPPER - rhs.UPPER - ((LOWER - rhs.LOWER) > LOWER),
                     LOWER - rhs.LOWER);
}

UINT128_DEVICE uint128_t &uint128_t::operator-=(const uint128_t &rhs)
{
    *this = *this - rhs;
    return *this;
}

UINT128_DEVICE uint128_t uint128_t::operator*(const uint128_t &rhs) const
{
    // split values into 4 32-bit parts
    uint64_t top[4]    = {UPPER >> 32, UPPER & 0xffffffff, LOWER >> 32,
                          LOWER & 0xffffffff};
    uint64_t bottom[4] = {rhs.UPPER >> 32, rhs.UPPER & 0xffffffff,
                          rhs.LOWER >> 32, rhs.LOWER & 0xffffffff};
    uint64_t products[4][4];

    // multiply each component of the values
    for (int y = 3; y > -1; y--)
    {
        for (int x = 3; x > -1; x--)
        {
            products[3 - x][y] = top[x] * bottom[y];
        }
    }

    // first row
    uint64_t fourth32 = (products[0][3] & 0xffffffff);
    uint64_t third32  = (products[0][2] & 0xffffffff) + (products[0][3] >> 32);
    uint64_t second32 = (products[0][1] & 0xffffffff) + (products[0][2] >> 32);
    uint64_t first32  = (products[0][0] & 0xffffffff) + (products[0][1] >> 32);

    // second row
    third32 += (products[1][3] & 0xffffffff);
    second32 += (products[1][2] & 0xffffffff) + (products[1][3] >> 32);
    first32 += (products[1][1] & 0xffffffff) + (products[1][2] >> 32);

    // third row
    second32 += (products[2][3] & 0xffffffff);
    first32 += (products[2][2] & 0xffffffff) + (products[2][3] >> 32);

    // fourth row
    first32 += (products[3][3] & 0xffffffff);

    // move carry to next digit
    third32 += fourth32 >> 32;
    second32 += third32 >> 32;
    first32 += second32 >> 32;

    // remove carry from current digit
    fourth32 &= 0xffffffff;
    third32 &= 0xffffffff;
    second32 &= 0xffffffff;
    first32 &= 0xffffffff;

    // combine components
    return uint128_t((first32 << 32) | second32, (third32 << 32) | fourth32);
}

UINT128_DEVICE uint128_t &uint128_t::operator*=(const uint128_t &rhs)
{
    *this = *this * rhs;
    return *this;
}

UINT128_DEVICE std::pair<uint128_t, uint128_t> uint128_t::divmod(
    const uint128_t &lhs,
    const uint128_t &rhs) const
{
    // Save some calculations /////////////////////
    if (rhs == uint128_t(0, 0))
    {
        return {(uint128_t)0, (uint128_t)0};
    }
    else if (rhs == uint128_t(0, 1))
    {
        return std::pair<uint128_t, uint128_t>(lhs, uint128_t(0, 0));
    }
    else if (lhs == rhs)
    {
        return std::pair<uint128_t, uint128_t>(uint128_t(0, 1),
                                               uint128_t(0, 0));
    }
    else if ((lhs == uint128_t(0, 0)) || (lhs < rhs))
    {
        return std::pair<uint128_t, uint128_t>(uint128_t(0, 0), lhs);
    }

    std::pair<uint128_t, uint128_t> qr(uint128_t(0, 0), uint128_t(0, 0));
    for (uint8_t x = lhs.bits(); x > 0; x--)
    {
        qr.first <<= uint128_t(0, 1);
        qr.second <<= uint128_t(0, 1);

        if ((lhs >> (x - 1U)) & 1)
        {
            ++qr.second;
        }

        if (qr.second >= rhs)
        {
            qr.second -= rhs;
            ++qr.first;
        }
    }
    return qr;
}

UINT128_DEVICE uint128_t uint128_t::operator/(const uint128_t &rhs) const
{
    return divmod(*this, rhs).first;
}

UINT128_DEVICE uint128_t &uint128_t::operator/=(const uint128_t &rhs)
{
    *this = *this / rhs;
    return *this;
}

UINT128_DEVICE uint128_t uint128_t::operator%(const uint128_t &rhs) const
{
    return divmod(*this, rhs).second;
}

UINT128_DEVICE uint128_t &uint128_t::operator%=(const uint128_t &rhs)
{
    *this = *this % rhs;
    return *this;
}

UINT128_DEVICE uint128_t &uint128_t::operator++()
{
    return *this += uint128_t(0, 1);
}

UINT128_DEVICE uint128_t uint128_t::operator++(int)
{
    uint128_t temp(*this);
    ++*this;
    return temp;
}

UINT128_DEVICE uint128_t &uint128_t::operator--()
{
    return *this -= uint128_t(0, 1);
}

UINT128_DEVICE uint128_t uint128_t::operator--(int)
{
    uint128_t temp(*this);
    --*this;
    return temp;
}

UINT128_DEVICE uint128_t uint128_t::operator+() const
{
    return *this;
}

UINT128_DEVICE uint128_t uint128_t::operator-() const
{
    return ~*this + uint128_t(0, 1);
}

UINT128_DEVICE const uint64_t &uint128_t::upper() const
{
    return UPPER;
}

UINT128_DEVICE const uint64_t &uint128_t::lower() const
{
    return LOWER;
}

UINT128_DEVICE uint8_t uint128_t::bits() const
{
    uint8_t out = 0;
    if (UPPER)
    {
        out         = 64;
        uint64_t up = UPPER;
        while (up)
        {
            up >>= 1;
            out++;
        }
    }
    else
    {
        uint64_t low = LOWER;
        while (low)
        {
            low >>= 1;
            out++;
        }
    }
    return out;
}

UINT128_DEVICE uint128_t operator<<(const bool &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) << rhs;
}

UINT128_DEVICE uint128_t operator<<(const uint8_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) << rhs;
}

UINT128_DEVICE uint128_t operator<<(const uint16_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) << rhs;
}

UINT128_DEVICE uint128_t operator<<(const uint32_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) << rhs;
}

UINT128_DEVICE uint128_t operator<<(const uint64_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) << rhs;
}

UINT128_DEVICE uint128_t operator<<(const int8_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) << rhs;
}

UINT128_DEVICE uint128_t operator<<(const int16_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) << rhs;
}

UINT128_DEVICE uint128_t operator<<(const int32_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) << rhs;
}

UINT128_DEVICE uint128_t operator<<(const int64_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) << rhs;
}

UINT128_DEVICE uint128_t operator>>(const bool &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) >> rhs;
}

UINT128_DEVICE uint128_t operator>>(const uint8_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) >> rhs;
}

UINT128_DEVICE uint128_t operator>>(const uint16_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) >> rhs;
}

UINT128_DEVICE uint128_t operator>>(const uint32_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) >> rhs;
}

UINT128_DEVICE uint128_t operator>>(const uint64_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) >> rhs;
}

UINT128_DEVICE uint128_t operator>>(const int8_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) >> rhs;
}

UINT128_DEVICE uint128_t operator>>(const int16_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) >> rhs;
}

UINT128_DEVICE uint128_t operator>>(const int32_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) >> rhs;
}

UINT128_DEVICE uint128_t operator>>(const int64_t &lhs, const uint128_t &rhs)
{
    return uint128_t(lhs) >> rhs;
}

} // namespace FastFss::impl
#endif