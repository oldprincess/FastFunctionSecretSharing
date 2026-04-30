#ifndef SRC_IMPL_DEF_H
#define SRC_IMPL_DEF_H

#include <cstddef>
#include <cstdint>
#include <wideint/wideint.hpp>

#if defined(__CUDACC__)

#define FAST_FSS_DEVICE        __device__
#define FAST_FSS_HD            __host__ __device__
#define CUDA_MAX_GRID_DIM      (128)
#define CUDA_DEFAULT_BLOCK_DIM 256

#else

#define FAST_FSS_DEVICE
#define FAST_FSS_HD
#define CUDA_MAX_GRID_DIM
#define CUDA_DEFAULT_BLOCK_DIM

#endif

#define _FAST_FSS_DISPATCH_CASE(ELEMENT_TYPE, ...) \
    case sizeof(ELEMENT_TYPE): {                   \
        using scalar_t = ELEMENT_TYPE;             \
        return __VA_ARGS__();                      \
    }

#define _FAST_FSS_DISPATCH_SWITCH(ELEMENT_SIZE, DEFAULT_CASE, ...) \
    [&] {                                                          \
        const auto &THE_TYPE_SIZE = ELEMENT_SIZE;                  \
        switch (THE_TYPE_SIZE)                                     \
        {                                                          \
            __VA_ARGS__                                            \
            default: DEFAULT_CASE                                  \
        }                                                          \
    }

#define FAST_FSS_DISPATCH_INTEGRAL_TYPES(ELEMENT_SIZE, DEFAULT_CASE, ...)                                            \
    _FAST_FSS_DISPATCH_SWITCH(                                                                                       \
        ELEMENT_SIZE, DEFAULT_CASE, _FAST_FSS_DISPATCH_CASE(std::uint8_t, __VA_ARGS__);                              \
        _FAST_FSS_DISPATCH_CASE(std::uint16_t, __VA_ARGS__); _FAST_FSS_DISPATCH_CASE(std::uint32_t, __VA_ARGS__);    \
        _FAST_FSS_DISPATCH_CASE(std::uint64_t, __VA_ARGS__); _FAST_FSS_DISPATCH_CASE(wideint::uint<2>, __VA_ARGS__); \
        _FAST_FSS_DISPATCH_CASE(wideint::uint<3>, __VA_ARGS__);                                                      \
        _FAST_FSS_DISPATCH_CASE(wideint::uint<4>, __VA_ARGS__);)                                                     \
    ()

#endif
