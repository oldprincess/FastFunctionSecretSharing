#ifndef SRC_IMPL_DEF_H
#define SRC_IMPL_DEF_H

#include <cstddef>
#include <cstdint>
#include <cstdio>

#if defined(__CUDACC__)

#define FAST_FSS_DEVICE        __device__
#define CUDA_MAX_GRID_DIM      (128 * 32)
#define CUDA_DEFAULT_BLOCK_DIM 512

#else

#define FAST_FSS_DEVICE
#define CUDA_MAX_GRID_DIM
#define CUDA_DEFAULT_BLOCK_DIM

#endif

#include "uint128_t.h"

#define _FAST_FSS_DISPATCH_CASE(ELEMENT_TYPE, ...) \
    case sizeof(ELEMENT_TYPE): {                   \
        using scalar_t = ELEMENT_TYPE;             \
        return __VA_ARGS__();                      \
    }

#define _FAST_FSS_DISPATCH_SWITCH(ELEMENT_SIZE, DEFAULT_CASE, ...) \
    [&] {                                                          \
        const auto& THE_TYPE_SIZE = ELEMENT_SIZE;                  \
        switch (THE_TYPE_SIZE)                                     \
        {                                                          \
            __VA_ARGS__                                            \
            default: DEFAULT_CASE                                  \
        }                                                          \
    }

#define FAST_FSS_DISPATCH_INTEGRAL_TYPES(ELEMENT_SIZE, DEFAULT_CASE, ...) \
    _FAST_FSS_DISPATCH_SWITCH(                                            \
        ELEMENT_SIZE, DEFAULT_CASE,                                       \
        _FAST_FSS_DISPATCH_CASE(std::uint8_t, __VA_ARGS__);               \
        _FAST_FSS_DISPATCH_CASE(std::uint16_t, __VA_ARGS__);              \
        _FAST_FSS_DISPATCH_CASE(std::uint32_t, __VA_ARGS__);              \
        _FAST_FSS_DISPATCH_CASE(std::uint64_t, __VA_ARGS__);              \
        _FAST_FSS_DISPATCH_CASE(FastFss::impl::uint128_t, __VA_ARGS__);)()

#endif