#ifndef FAST_FSS_PP_GROTTO_H
#define FAST_FSS_PP_GROTTO_H

#include <FastFss/cpu/grotto.h>
#include <FastFss/cuda/grotto.h>
#include <FastFss/errors.h>
#include <FastFss/grotto.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace FastFss::grotto {

inline void grottoKeyZip(std::span<std::uint8_t>       zippedKey,
                         std::span<const std::uint8_t> key,
                         std::size_t                   bitWidthIn,
                         std::size_t                   elementSize,
                         std::size_t                   elementNum)
{
    int ret = FastFss_grottoKeyZip(zippedKey.data(), zippedKey.size(), key.data(), key.size(), bitWidthIn, elementSize,
                                   elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_grottoKeyZip failed. error code: " + std::to_string(ret));
    }
}

inline void grottoKeyUnzip(std::span<std::uint8_t>       key,
                           std::span<const std::uint8_t> zippedKey,
                           std::size_t                   bitWidthIn,
                           std::size_t                   elementSize,
                           std::size_t                   elementNum)
{
    int ret = FastFss_grottoKeyUnzip(key.data(), key.size(), zippedKey.data(), zippedKey.size(), bitWidthIn,
                                     elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_grottoKeyUnzip failed. error code: " + std::to_string(ret));
    }
}

inline std::size_t grottoGetKeyDataSize(std::size_t bitWidthIn, std::size_t elementSize, std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         ret         = FastFss_grottoGetKeyDataSize(&keyDataSize, bitWidthIn, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_grottoGetKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

inline std::size_t grottoGetZippedKeyDataSize(std::size_t bitWidthIn, std::size_t elementSize, std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         ret         = FastFss_grottoGetZippedKeyDataSize(&keyDataSize, bitWidthIn, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_grottoGetZippedKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

inline std::size_t grottoGetCacheDataSize(std::size_t bitWidthIn, std::size_t elementSize, std::size_t elementNum)
{
    std::size_t cacheDataSize = 0;
    int         ret           = FastFss_grottoGetCacheDataSize(&cacheDataSize, bitWidthIn, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_grottoGetCacheDataSize failed. error code: " + std::to_string(ret));
    }
    return cacheDataSize;
}

namespace cpu {

template <typename T>
void grottoKeyGen(std::span<std::uint8_t>       key,
                  std::span<const T>            alpha,
                  std::span<const std::uint8_t> seed0,
                  std::span<const std::uint8_t> seed1,
                  std::size_t                   bitWidthIn)
{
    int ret = FastFss_cpu_grottoKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), seed0.data(),
                                       seed0.size(), seed1.data(), seed1.size(), bitWidthIn, sizeof(T), alpha.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_grottoKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void grottoEqEval(std::span<T>                  sharedOut,
                  std::span<const T>            maskedX,
                  std::span<const std::uint8_t> key,
                  std::span<const std::uint8_t> seed,
                  int                           partyId,
                  std::size_t                   bitWidthIn,
                  std::span<std::uint8_t>       cache)
{
    int ret = FastFss_cpu_grottoEqEval(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(),
                                       maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(),
                                       partyId, bitWidthIn, sizeof(T), maskedX.size(), cache.data(), cache.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_grottoEqEval failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void grottoEval(std::span<T>                  sharedOut,
                std::span<const T>            maskedX,
                std::span<const std::uint8_t> key,
                std::span<const std::uint8_t> seed,
                bool                          equalBound,
                int                           partyId,
                std::size_t                   bitWidthIn,
                std::span<std::uint8_t>       cache)
{
    int ret =
        FastFss_cpu_grottoEval(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(),
                               maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(), equalBound,
                               partyId, bitWidthIn, sizeof(T), maskedX.size(), cache.data(), cache.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_grottoEval failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void grottoMICEval(std::span<T>                  sharedOut,
                   std::span<const T>            maskedX,
                   std::span<const std::uint8_t> key,
                   std::span<const std::uint8_t> seed,
                   int                           partyId,
                   std::span<const T>            leftEndpoints,
                   std::span<const T>            rightEndpoints,
                   std::size_t                   bitWidthIn,
                   std::span<std::uint8_t>       cache)
{
    int ret = FastFss_cpu_grottoMICEval(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(),
                                        maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(),
                                        partyId, leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
                                        rightEndpoints.data(), rightEndpoints.size() * sizeof(T), bitWidthIn, sizeof(T),
                                        maskedX.size(), cache.data(), cache.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_grottoMICEval failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void grottoIntervalLutEval(std::span<T>                  sharedOutE,
                           std::span<T>                  sharedOutT,
                           std::span<const T>            maskedX,
                           std::span<const std::uint8_t> key,
                           std::span<const std::uint8_t> seed,
                           int                           partyId,
                           std::span<const T>            leftEndpoints,
                           std::span<const T>            rightEndpoints,
                           std::span<const T>            lookUpTable,
                           std::size_t                   bitWidthIn,
                           std::size_t                   bitWidthOut,
                           std::span<std::uint8_t>       cache)
{
    int ret = FastFss_cpu_grottoIntervalLutEval(
        sharedOutE.data(), sharedOutE.size() * sizeof(T), sharedOutT.data(), sharedOutT.size() * sizeof(T),
        maskedX.data(), maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(), partyId,
        leftEndpoints.data(), leftEndpoints.size() * sizeof(T), rightEndpoints.data(),
        rightEndpoints.size() * sizeof(T), lookUpTable.data(), lookUpTable.size() * sizeof(T), bitWidthIn, bitWidthOut,
        sizeof(T), maskedX.size(), cache.data(), cache.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_grottoIntervalLutEval failed. error code: " + std::to_string(ret));
    }
}

} // namespace cpu

namespace cuda {

template <typename T>
void grottoKeyGen(std::span<std::uint8_t>       key,
                  std::span<const T>            alpha,
                  std::span<const std::uint8_t> seed0,
                  std::span<const std::uint8_t> seed1,
                  std::size_t                   bitWidthIn,
                  void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_grottoKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), seed0.data(),
                                        seed0.size(), seed1.data(), seed1.size(), bitWidthIn, sizeof(T), alpha.size(),
                                        cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_grottoKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void grottoEqEval(std::span<T>                  sharedOut,
                  std::span<const T>            maskedX,
                  std::span<const std::uint8_t> key,
                  std::span<const std::uint8_t> seed,
                  int                           partyId,
                  std::size_t                   bitWidthIn,
                  std::span<std::uint8_t>       cache,
                  void                         *cudaStreamPtr)
{
    int ret =
        FastFss_cuda_grottoEqEval(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(),
                                  maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(), partyId,
                                  bitWidthIn, sizeof(T), maskedX.size(), cache.data(), cache.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_grottoEqEval failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void grottoEval(std::span<T>                  sharedOut,
                std::span<const T>            maskedX,
                std::span<const std::uint8_t> key,
                std::span<const std::uint8_t> seed,
                bool                          equalBound,
                int                           partyId,
                std::size_t                   bitWidthIn,
                std::span<std::uint8_t>       cache,
                void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_grottoEval(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(),
                                      maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(),
                                      equalBound, partyId, bitWidthIn, sizeof(T), maskedX.size(), cache.data(),
                                      cache.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_grottoEval failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void grottoMICEval(std::span<T>                  sharedOut,
                   std::span<const T>            maskedX,
                   std::span<const std::uint8_t> key,
                   std::span<const std::uint8_t> seed,
                   int                           partyId,
                   std::span<const T>            leftEndpoints,
                   std::span<const T>            rightEndpoints,
                   std::size_t                   bitWidthIn,
                   std::span<std::uint8_t>       cache,
                   void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_grottoMICEval(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(),
                                         maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(),
                                         partyId, leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
                                         rightEndpoints.data(), rightEndpoints.size() * sizeof(T), bitWidthIn,
                                         sizeof(T), maskedX.size(), cache.data(), cache.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_grottoMICEval failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void grottoIntervalLutEval(std::span<T>                  sharedOutE,
                           std::span<T>                  sharedOutT,
                           std::span<const T>            maskedX,
                           std::span<const std::uint8_t> key,
                           std::span<const std::uint8_t> seed,
                           int                           partyId,
                           std::span<const T>            leftEndpoints,
                           std::span<const T>            rightEndpoints,
                           std::span<const T>            lookUpTable,
                           std::size_t                   bitWidthIn,
                           std::size_t                   bitWidthOut,
                           std::span<std::uint8_t>       cache,
                           void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_grottoIntervalLutEval(
        sharedOutE.data(), sharedOutE.size() * sizeof(T), sharedOutT.data(), sharedOutT.size() * sizeof(T),
        maskedX.data(),
        maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(), partyId, leftEndpoints.data(),
        leftEndpoints.size() * sizeof(T), rightEndpoints.data(), rightEndpoints.size() * sizeof(T), lookUpTable.data(),
        lookUpTable.size() * sizeof(T), bitWidthIn, bitWidthOut, sizeof(T), maskedX.size(), cache.data(), cache.size(),
        cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_grottoIntervalLutEval failed. error code: " + std::to_string(ret));
    }
}

} // namespace cuda

} // namespace FastFss::grotto

#endif
