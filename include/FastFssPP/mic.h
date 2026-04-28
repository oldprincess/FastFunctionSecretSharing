#ifndef FAST_FSS_PP_MIC_H
#define FAST_FSS_PP_MIC_H

#include <FastFss/cpu/mic.h>
#include <FastFss/cuda/mic.h>
#include <FastFss/errors.h>
#include <FastFss/mic.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace FastFss::mic {

inline void dcfMICKeyZip(std::span<std::uint8_t>       zippedKey,
                         std::span<const std::uint8_t> key,
                         std::size_t                   bitWidthIn,
                         std::size_t                   bitWidthOut,
                         std::size_t                   elementSize,
                         std::size_t                   elementNum)
{
    int ret = FastFss_dcfMICKeyZip(zippedKey.data(), zippedKey.size(), key.data(), key.size(), bitWidthIn, bitWidthOut,
                                   elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfMICKeyZip failed. error code: " + std::to_string(ret));
    }
}

inline void dcfMICKeyUnzip(std::span<std::uint8_t>       key,
                           std::span<const std::uint8_t> zippedKey,
                           std::size_t                   bitWidthIn,
                           std::size_t                   bitWidthOut,
                           std::size_t                   elementSize,
                           std::size_t                   elementNum)
{
    int ret = FastFss_dcfMICKeyUnzip(key.data(), key.size(), zippedKey.data(), zippedKey.size(), bitWidthIn,
                                     bitWidthOut, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfMICKeyUnzip failed. error code: " + std::to_string(ret));
    }
}

inline std::size_t dcfMICGetKeyDataSize(std::size_t bitWidthIn,
                                        std::size_t bitWidthOut,
                                        std::size_t elementSize,
                                        std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         ret = FastFss_dcfMICGetKeyDataSize(&keyDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfMICGetKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

inline std::size_t dcfMICGetZippedKeyDataSize(std::size_t bitWidthIn,
                                              std::size_t bitWidthOut,
                                              std::size_t elementSize,
                                              std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int ret = FastFss_dcfMICGetZippedKeyDataSize(&keyDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfMICGetZippedKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

inline std::size_t dcfMICGetCacheDataSize(std::size_t bitWidthIn,
                                          std::size_t bitWidthOut,
                                          std::size_t elementSize,
                                          std::size_t elementNum)
{
    std::size_t cacheDataSize = 0;
    int         ret = FastFss_dcfMICGetCacheDataSize(&cacheDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfMICGetCacheDataSize failed. error code: " + std::to_string(ret));
    }
    return cacheDataSize;
}

namespace cpu {

template <typename T>
void dcfMICKeyGen(std::span<std::uint8_t>       key,
                  std::span<T>                  z,
                  std::span<const T>            alpha,
                  std::span<const std::uint8_t> seed0,
                  std::span<const std::uint8_t> seed1,
                  std::span<const T>            leftEndpoints,
                  std::span<const T>            rightEndpoints,
                  std::size_t                   bitWidthIn,
                  std::size_t                   bitWidthOut)
{
    int ret = FastFss_cpu_dcfMICKeyGen(
        key.data(), key.size(), z.data(), z.size() * sizeof(T), alpha.data(), alpha.size() * sizeof(T), seed0.data(),
        seed0.size(), seed1.data(), seed1.size(), leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
        rightEndpoints.data(), rightEndpoints.size() * sizeof(T), bitWidthIn, bitWidthOut, sizeof(T), alpha.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_dcfMICKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dcfMICEval(std::span<T>                  sharedOut,
                std::span<const T>            maskedX,
                std::span<const std::uint8_t> key,
                std::span<const T>            sharedZ,
                std::span<const std::uint8_t> seed,
                int                           partyId,
                std::span<const T>            leftEndpoints,
                std::span<const T>            rightEndpoints,
                std::size_t                   bitWidthIn,
                std::size_t                   bitWidthOut,
                std::span<std::uint8_t>       cache)
{
    int ret = FastFss_cpu_dcfMICEval(
        sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T), key.data(),
        key.size(), sharedZ.data(), sharedZ.size() * sizeof(T), seed.data(), seed.size(), partyId, leftEndpoints.data(),
        leftEndpoints.size() * sizeof(T), rightEndpoints.data(), rightEndpoints.size() * sizeof(T), bitWidthIn,
        bitWidthOut, sizeof(T), maskedX.size(), cache.data(), cache.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_dcfMICEval failed. error code: " + std::to_string(ret));
    }
}

} // namespace cpu

namespace cuda {

template <typename T>
void dcfMICKeyGen(std::span<std::uint8_t>       key,
                  std::span<T>                  z,
                  std::span<const T>            alpha,
                  std::span<const std::uint8_t> seed0,
                  std::span<const std::uint8_t> seed1,
                  std::span<const T>            leftEndpoints,
                  std::span<const T>            rightEndpoints,
                  std::size_t                   bitWidthIn,
                  std::size_t                   bitWidthOut,
                  void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_dcfMICKeyGen(key.data(), key.size(), z.data(), z.size() * sizeof(T), alpha.data(),
                                        alpha.size() * sizeof(T), seed0.data(), seed0.size(), seed1.data(),
                                        seed1.size(), leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
                                        rightEndpoints.data(), rightEndpoints.size() * sizeof(T), bitWidthIn,
                                        bitWidthOut, sizeof(T), alpha.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_dcfMICKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dcfMICEval(std::span<T>                  sharedOut,
                std::span<const T>            maskedX,
                std::span<const std::uint8_t> key,
                std::span<const T>            sharedZ,
                std::span<const std::uint8_t> seed,
                int                           partyId,
                std::span<const T>            leftEndpoints,
                std::span<const T>            rightEndpoints,
                std::size_t                   bitWidthIn,
                std::size_t                   bitWidthOut,
                std::span<std::uint8_t>       cache,
                void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_dcfMICEval(
        sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T), key.data(),
        key.size(), sharedZ.data(), sharedZ.size() * sizeof(T), seed.data(), seed.size(), partyId, leftEndpoints.data(),
        leftEndpoints.size() * sizeof(T), rightEndpoints.data(), rightEndpoints.size() * sizeof(T), bitWidthIn,
        bitWidthOut, sizeof(T), maskedX.size(), cache.data(), cache.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_dcfMICEval failed. error code: " + std::to_string(ret));
    }
}

} // namespace cuda

} // namespace FastFss::mic

#endif
