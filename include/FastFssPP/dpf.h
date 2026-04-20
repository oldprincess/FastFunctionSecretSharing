#ifndef FAST_FSS_PP_DPF_H
#define FAST_FSS_PP_DPF_H

#include <FastFss/cpu/dpf.h>
#include <FastFss/cuda/dpf.h>
#include <FastFss/dpf.h>
#include <FastFss/errors.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace FastFss::dpf {

inline void dpfKeyZip(std::span<std::uint8_t>       zippedKey,
                      std::span<const std::uint8_t> key,
                      std::size_t                   bitWidthIn,
                      std::size_t                   bitWidthOut,
                      std::size_t                   groupSize,
                      std::size_t                   elementSize,
                      std::size_t                   elementNum)
{
    int ret = FastFss_dpfKeyZip(zippedKey.data(), zippedKey.size(), key.data(), key.size(), bitWidthIn, bitWidthOut,
                                groupSize, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dpfKeyZip failed. error code: " + std::to_string(ret));
    }
}

inline void dpfKeyUnzip(std::span<std::uint8_t>       key,
                        std::span<const std::uint8_t> zippedKey,
                        std::size_t                   bitWidthIn,
                        std::size_t                   bitWidthOut,
                        std::size_t                   groupSize,
                        std::size_t                   elementSize,
                        std::size_t                   elementNum)
{
    int ret = FastFss_dpfKeyUnzip(key.data(), key.size(), zippedKey.data(), zippedKey.size(), bitWidthIn, bitWidthOut,
                                  groupSize, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dpfKeyUnzip failed. error code: " + std::to_string(ret));
    }
}

inline std::size_t dpfGetKeyDataSize(std::size_t bitWidthIn,
                                     std::size_t bitWidthOut,
                                     std::size_t groupSize,
                                     std::size_t elementSize,
                                     std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int ret = FastFss_dpfGetKeyDataSize(&keyDataSize, bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dpfGetKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

inline std::size_t dpfGetZippedKeyDataSize(std::size_t bitWidthIn,
                                           std::size_t bitWidthOut,
                                           std::size_t groupSize,
                                           std::size_t elementSize,
                                           std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         ret =
        FastFss_dpfGetZippedKeyDataSize(&keyDataSize, bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dpfGetZippedKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

inline std::size_t dpfGetCacheDataSize(std::size_t bitWidthIn, std::size_t elementSize, std::size_t elementNum)
{
    std::size_t cacheDataSize = 0;
    int         ret           = FastFss_dpfGetCacheDataSize(&cacheDataSize, bitWidthIn, elementSize, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_dpfGetCacheDataSize failed. error code: " + std::to_string(ret));
    }
    return cacheDataSize;
}

namespace cpu {

template <typename T>
void dpfKeyGen(std::span<std::uint8_t>       key,
               std::span<const T>            alpha,
               std::span<const T>            beta,
               std::span<const std::uint8_t> seed0,
               std::span<const std::uint8_t> seed1,
               std::size_t                   bitWidthIn,
               std::size_t                   bitWidthOut,
               std::size_t                   groupSize)
{
    int ret = FastFss_cpu_dpfKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), beta.data(),
                                    beta.size() * sizeof(T), seed0.data(), seed0.size(), seed1.data(), seed1.size(),
                                    bitWidthIn, bitWidthOut, groupSize, sizeof(T), alpha.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_dpfKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dpfEval(std::span<T>                  sharedOut,
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
        FastFss_cpu_dpfEval(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                            key.data(), key.size(), seed.data(), seed.size(), partyId, bitWidthIn, bitWidthOut,
                            groupSize, sizeof(T), maskedX.size(), cache.data(), cache.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_dpfEval failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dpfEvalAll(std::span<T>                  sharedOut,
                std::span<const T>            maskedX,
                std::span<const std::uint8_t> key,
                std::span<const std::uint8_t> seed,
                int                           partyId,
                std::size_t                   bitWidthIn,
                std::size_t                   bitWidthOut,
                std::size_t                   groupSize,
                std::span<std::uint8_t>       cache)
{
    int ret = FastFss_cpu_dpfEvalAll(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(),
                                     maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(),
                                     partyId, bitWidthIn, bitWidthOut, groupSize, sizeof(T), maskedX.size(),
                                     cache.data(), cache.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_dpfEvalAll failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dpfEvalMulti(std::span<T>                  sharedOut,
                  std::span<const T>            maskedX,
                  std::span<const std::uint8_t> key,
                  std::span<const std::uint8_t> seed,
                  int                           partyId,
                  std::span<const T>            point,
                  std::size_t                   bitWidthIn,
                  std::size_t                   bitWidthOut,
                  std::size_t                   groupSize,
                  std::span<std::uint8_t>       cache)
{
    int ret = FastFss_cpu_dpfEvalMulti(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(),
                                       maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(),
                                       partyId, point.data(), point.size() * sizeof(T), bitWidthIn, bitWidthOut,
                                       groupSize, sizeof(T), maskedX.size(), cache.data(), cache.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_dpfEvalMulti failed. error code: " + std::to_string(ret));
    }
}

} // namespace cpu

namespace cuda {

template <typename T>
void dpfKeyGen(std::span<std::uint8_t>       key,
               std::span<const T>            alpha,
               std::span<const T>            beta,
               std::span<const std::uint8_t> seed0,
               std::span<const std::uint8_t> seed1,
               std::size_t                   bitWidthIn,
               std::size_t                   bitWidthOut,
               std::size_t                   groupSize,
               void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_dpfKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), beta.data(),
                                     beta.size() * sizeof(T), seed0.data(), seed0.size(), seed1.data(), seed1.size(),
                                     bitWidthIn, bitWidthOut, groupSize, sizeof(T), alpha.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_dpfKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dpfEval(std::span<T>                  sharedOut,
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
        FastFss_cuda_dpfEval(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                             key.data(), key.size(), seed.data(), seed.size(), partyId, bitWidthIn, bitWidthOut,
                             groupSize, sizeof(T), maskedX.size(), cache.data(), cache.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_dpfEval failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dpfEvalAll(std::span<T>                  sharedOut,
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
    int ret = FastFss_cuda_dpfEvalAll(sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(),
                                      maskedX.size() * sizeof(T), key.data(), key.size(), seed.data(), seed.size(),
                                      partyId, bitWidthIn, bitWidthOut, groupSize, sizeof(T), maskedX.size(),
                                      cache.data(), cache.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_dpfEvalAll failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void dpfEvalMulti(std::span<T>                  sharedOut,
                  std::span<const T>            maskedX,
                  std::span<const std::uint8_t> key,
                  std::span<const std::uint8_t> seed,
                  int                           partyId,
                  std::span<const T>            point,
                  std::size_t                   bitWidthIn,
                  std::size_t                   bitWidthOut,
                  std::size_t                   groupSize,
                  std::span<std::uint8_t>       cache,
                  void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_dpfEvalMulti(
        sharedOut.data(), sharedOut.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T), key.data(),
        key.size(), seed.data(), seed.size(), partyId, point.data(), point.size() * sizeof(T), bitWidthIn, bitWidthOut,
        groupSize, sizeof(T), maskedX.size(), cache.data(), cache.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_dpfEvalMulti failed. error code: " + std::to_string(ret));
    }
}

} // namespace cuda

} // namespace FastFss::dpf

#endif
