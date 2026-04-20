#ifndef FAST_FSS_PP_DCF_H
#define FAST_FSS_PP_DCF_H

#include <FastFss/cpu/dcf.h>
#include <FastFss/cuda/dcf.h>
#include <FastFss/dcf.h>
#include <FastFss/errors.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace FastFss::dcf {

inline void dcfKeyZip(std::span<std::uint8_t>       zippedKey,
                      std::span<const std::uint8_t> key,
                      std::size_t                   bitWidthIn,
                      std::size_t                   bitWidthOut,
                      std::size_t                   groupSize,
                      std::size_t                   elementSize,
                      std::size_t                   elementNum)
{
    int ret = FastFss_dcfKeyZip(zippedKey.data(), zippedKey.size(), key.data(), key.size(), bitWidthIn, bitWidthOut,
                                groupSize, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfKeyZip failed. error code: " + std::to_string(ret));
    }
}

inline void dcfKeyUnzip(std::span<std::uint8_t>       key,
                        std::span<const std::uint8_t> zippedKey,
                        std::size_t                   bitWidthIn,
                        std::size_t                   bitWidthOut,
                        std::size_t                   groupSize,
                        std::size_t                   elementSize,
                        std::size_t                   elementNum)
{
    int ret = FastFss_dcfKeyUnzip(key.data(), key.size(), zippedKey.data(), zippedKey.size(), bitWidthIn, bitWidthOut,
                                  groupSize, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfKeyUnzip failed. error code: " + std::to_string(ret));
    }
}

inline std::size_t dcfGetKeyDataSize(std::size_t bitWidthIn,
                                     std::size_t bitWidthOut,
                                     std::size_t groupSize,
                                     std::size_t elementSize,
                                     std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int ret = FastFss_dcfGetKeyDataSize(&keyDataSize, bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfGetKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

inline std::size_t dcfGetZippedKeyDataSize(std::size_t bitWidthIn,
                                           std::size_t bitWidthOut,
                                           std::size_t groupSize,
                                           std::size_t elementSize,
                                           std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         ret =
        FastFss_dcfGetZippedKeyDataSize(&keyDataSize, bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfGetZippedKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

inline std::size_t dcfGetCacheDataSize(std::size_t bitWidthIn,
                                       std::size_t bitWidthOut,
                                       std::size_t groupSize,
                                       std::size_t elementSize,
                                       std::size_t elementNum)
{
    std::size_t cacheDataSize = 0;
    int ret = FastFss_dcfGetCacheDataSize(&cacheDataSize, bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dcfGetCacheDataSize failed. error code: " + std::to_string(ret));
    }
    return cacheDataSize;
}

namespace cpu {

template <typename T>
void dcfKeyGen(std::span<std::uint8_t>       key,
               std::span<const T>            alpha,
               std::span<const T>            beta,
               std::span<const std::uint8_t> seed0,
               std::span<const std::uint8_t> seed1,
               std::size_t                   bitWidthIn,
               std::size_t                   bitWidthOut,
               std::size_t                   groupSize)
{
    int ret = FastFss_cpu_dcfKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), beta.data(),
                                    beta.size() * sizeof(T), seed0.data(), seed0.size(), seed1.data(), seed1.size(),
                                    bitWidthIn, bitWidthOut, groupSize, sizeof(T), alpha.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_dcfKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dcfEval(std::span<T>                  sharedOut,
             std::span<const T>            maskedX,
             std::span<const std::uint8_t> key,
             std::span<const std::uint8_t> seed,
             int                           partyId,
             std::size_t                   bitWidthIn,
             std::size_t                   bitWidthOut,
             std::size_t                   groupSize,
             std::span<std::uint8_t>       cache)
{
    int ret =
        FastFss_cpu_dcfEval(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                            key.data(), key.size(), seed.data(), seed.size(), partyId, bitWidthIn, bitWidthOut,
                            groupSize, sizeof(T), maskedX.size(), cache.data(), cache.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_dcfEval failed. error code: " + std::to_string(ret));
    }
}

} // namespace cpu

namespace cuda {

template <typename T>
void dcfKeyGen(std::span<std::uint8_t>       key,
               std::span<const T>            alpha,
               std::span<const T>            beta,
               std::span<const std::uint8_t> seed0,
               std::span<const std::uint8_t> seed1,
               std::size_t                   bitWidthIn,
               std::size_t                   bitWidthOut,
               std::size_t                   groupSize,
               void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_dcfKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), beta.data(),
                                     beta.size() * sizeof(T), seed0.data(), seed0.size(), seed1.data(), seed1.size(),
                                     bitWidthIn, bitWidthOut, groupSize, sizeof(T), alpha.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_dcfKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dcfEval(std::span<T>                  sharedOut,
             std::span<const T>            maskedX,
             std::span<const std::uint8_t> key,
             std::span<const std::uint8_t> seed,
             int                           partyId,
             std::size_t                   bitWidthIn,
             std::size_t                   bitWidthOut,
             std::size_t                   groupSize,
             std::span<std::uint8_t>       cache,
             void                         *cudaStreamPtr)
{
    int ret =
        FastFss_cuda_dcfEval(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                             key.data(), key.size(), seed.data(), seed.size(), partyId, bitWidthIn, bitWidthOut,
                             groupSize, sizeof(T), maskedX.size(), cache.data(), cache.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_dcfEval failed. error code: " + std::to_string(ret));
    }
}

} // namespace cuda

} // namespace FastFss::dcf

#endif
