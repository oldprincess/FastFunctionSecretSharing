#ifndef FAST_FSS_PP_SPLINE_H
#define FAST_FSS_PP_SPLINE_H

#include <FastFss/cpu/spline.h>
#include <FastFss/cuda/spline.h>
#include <FastFss/errors.h>
#include <FastFss/spline.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace FastFss::spline {

inline void dcfSplineKeyZip(std::span<std::uint8_t>       zippedKey,
                            std::span<const std::uint8_t> key,
                            std::size_t                   degree,
                            std::size_t                   intervalNum,
                            std::size_t                   bitWidthIn,
                            std::size_t                   bitWidthOut,
                            std::size_t                   elementSize,
                            std::size_t                   elementNum)
{
    int ret = FastFss_dcfSplineKeyZip(zippedKey.data(), zippedKey.size(), key.data(), key.size(), degree, intervalNum,
                                      bitWidthIn, bitWidthOut, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfSplineKeyZip failed. error code: " + std::to_string(ret));
    }
}

inline void dcfSplineKeyUnzip(std::span<std::uint8_t>       key,
                              std::span<const std::uint8_t> zippedKey,
                              std::size_t                   degree,
                              std::size_t                   intervalNum,
                              std::size_t                   bitWidthIn,
                              std::size_t                   bitWidthOut,
                              std::size_t                   elementSize,
                              std::size_t                   elementNum)
{
    int ret = FastFss_dcfSplineKeyUnzip(key.data(), key.size(), zippedKey.data(), zippedKey.size(), degree, intervalNum,
                                        bitWidthIn, bitWidthOut, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfSplineKeyUnzip failed. error code: " + std::to_string(ret));
    }
}

inline std::size_t dcfSplineGetKeyDataSize(std::size_t degree,
                                           std::size_t intervalNum,
                                           std::size_t bitWidthIn,
                                           std::size_t bitWidthOut,
                                           std::size_t elementSize,
                                           std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int ret = FastFss_dcfSplineGetKeyDataSize(&keyDataSize, degree, intervalNum, bitWidthIn, bitWidthOut, elementSize,
                                              elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfSplineGetKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

inline std::size_t dcfSplineGetZippedKeyDataSize(std::size_t degree,
                                                 std::size_t intervalNum,
                                                 std::size_t bitWidthIn,
                                                 std::size_t bitWidthOut,
                                                 std::size_t elementSize,
                                                 std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int ret = FastFss_dcfSplineGetZippedKeyDataSize(&keyDataSize, degree, intervalNum, bitWidthIn, bitWidthOut,
                                                    elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfSplineGetZippedKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

inline std::size_t dcfSplineGetCacheDataSize(std::size_t degree,
                                             std::size_t intervalNum,
                                             std::size_t bitWidthIn,
                                             std::size_t bitWidthOut,
                                             std::size_t elementSize,
                                             std::size_t elementNum)
{
    std::size_t cacheDataSize = 0;
    int ret = FastFss_dcfSplineGetCacheDataSize(&cacheDataSize, degree, intervalNum, bitWidthIn, bitWidthOut,
                                                elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfSplineGetCacheDataSize failed. error code: " + std::to_string(ret));
    }
    return cacheDataSize;
}

namespace cpu {

template <typename T>
void dcfSplineKeyGen(std::span<std::uint8_t>       key,
                     std::span<T>                  e,
                     std::span<T>                  beta,
                     std::span<const T>            alpha,
                     std::span<const std::uint8_t> seed0,
                     std::span<const std::uint8_t> seed1,
                     std::span<const T>            coefficients,
                     std::span<const T>            leftEndpoints,
                     std::span<const T>            rightEndpoints,
                     std::size_t                   degree,
                     std::size_t                   bitWidthIn,
                     std::size_t                   bitWidthOut)
{
    int ret = FastFss_cpu_dcfSplineKeyGen(key.data(), key.size(), e.data(), e.size() * sizeof(T), beta.data(),
                                          beta.size() * sizeof(T), alpha.data(), alpha.size() * sizeof(T), seed0.data(),
                                          seed0.size(), seed1.data(), seed1.size(), coefficients.data(),
                                          coefficients.size() * sizeof(T), degree, leftEndpoints.data(),
                                          leftEndpoints.size() * sizeof(T), rightEndpoints.data(),
                                          rightEndpoints.size() * sizeof(T), leftEndpoints.size(), bitWidthIn,
                                          bitWidthOut, sizeof(T), alpha.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_dcfSplineKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dcfSplineEval(std::span<T>                  sharedOut,
                   std::span<const T>            maskedX,
                   std::span<const std::uint8_t> key,
                   std::span<const T>            sharedE,
                   std::span<const T>            sharedBeta,
                   std::span<const std::uint8_t> seed,
                   int                           partyId,
                   std::span<const T>            leftEndpoints,
                   std::span<const T>            rightEndpoints,
                   std::size_t                   degree,
                   std::size_t                   bitWidthIn,
                   std::size_t                   bitWidthOut,
                   std::span<std::uint8_t>       cache)
{
    int ret = FastFss_cpu_dcfSplineEval(
        sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T), key.data(),
        key.size(), sharedE.data(), sharedE.size() * sizeof(T), sharedBeta.data(), sharedBeta.size() * sizeof(T),
        seed.data(), seed.size(), partyId, leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
        rightEndpoints.data(), rightEndpoints.size() * sizeof(T), leftEndpoints.size(), degree, bitWidthIn, bitWidthOut,
        sizeof(T), maskedX.size(), cache.data(), cache.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_dcfSplineEval failed. error code: " + std::to_string(ret));
    }
}

} // namespace cpu

namespace cuda {

template <typename T>
void dcfSplineKeyGen(std::span<std::uint8_t>       key,
                     std::span<T>                  e,
                     std::span<T>                  beta,
                     std::span<const T>            alpha,
                     std::span<const std::uint8_t> seed0,
                     std::span<const std::uint8_t> seed1,
                     std::span<const T>            coefficients,
                     std::span<const T>            leftEndpoints,
                     std::span<const T>            rightEndpoints,
                     std::size_t                   degree,
                     std::size_t                   bitWidthIn,
                     std::size_t                   bitWidthOut,
                     void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_dcfSplineKeyGen(key.data(), key.size(), e.data(), e.size() * sizeof(T), beta.data(),
                                           beta.size() * sizeof(T), alpha.data(), alpha.size() * sizeof(T), seed0.data(),
                                           seed0.size(), seed1.data(), seed1.size(), coefficients.data(),
                                           coefficients.size() * sizeof(T), degree, leftEndpoints.data(),
                                           leftEndpoints.size() * sizeof(T), rightEndpoints.data(),
                                           rightEndpoints.size() * sizeof(T), leftEndpoints.size(), bitWidthIn,
                                           bitWidthOut, sizeof(T), alpha.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_dcfSplineKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dcfSplineEval(std::span<T>                  sharedOut,
                   std::span<const T>            maskedX,
                   std::span<const std::uint8_t> key,
                   std::span<const T>            sharedE,
                   std::span<const T>            sharedBeta,
                   std::span<const std::uint8_t> seed,
                   int                           partyId,
                   std::span<const T>            leftEndpoints,
                   std::span<const T>            rightEndpoints,
                   std::size_t                   degree,
                   std::size_t                   bitWidthIn,
                   std::size_t                   bitWidthOut,
                   std::span<std::uint8_t>       cache,
                   void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_dcfSplineEval(
        sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T), key.data(),
        key.size(), sharedE.data(), sharedE.size() * sizeof(T), sharedBeta.data(), sharedBeta.size() * sizeof(T),
        seed.data(), seed.size(), partyId, leftEndpoints.data(), leftEndpoints.size() * sizeof(T),
        rightEndpoints.data(), rightEndpoints.size() * sizeof(T), leftEndpoints.size(), degree, bitWidthIn, bitWidthOut,
        sizeof(T), maskedX.size(), cache.data(), cache.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_dcfSplineEval failed. error code: " + std::to_string(ret));
    }
}

} // namespace cuda

} // namespace FastFss::spline

#endif
