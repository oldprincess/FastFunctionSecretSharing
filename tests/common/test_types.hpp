#ifndef FAST_FSS_TESTS_COMMON_TEST_TYPES_HPP
#define FAST_FSS_TESTS_COMMON_TEST_TYPES_HPP

#include <cstdint>

#include "wideint/wideint.hpp"

namespace FastFss::tests {

using uint128_t = wideint::uint<2>;
using uint192_t = wideint::uint<3>;
using uint256_t = wideint::uint<4>;

template <typename T>
struct SupportedTypes;

using EvalElementTypes =
    ::testing::Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, uint128_t, uint192_t, uint256_t>;

} // namespace FastFss::tests

#endif
