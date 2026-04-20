#ifndef FAST_FSS_PP_OTTT_H
#define FAST_FSS_PP_OTTT_H

#include <FastFss/cpu/ottt.h>
#include <FastFss/cuda/ottt.h>
#include <FastFss/errors.h>
#include <FastFss/ottt.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace FastFss::ottt {

inline std::size_t otttGetKeyDataSize(std::size_t bitWidthIn, std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         ret         = FastFss_otttGetKeyDataSize(&keyDataSize, bitWidthIn, elementNum);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_otttGetKeyDataSize failed. error code: " + std::to_string(ret));
    }
    return keyDataSize;
}

namespace cpu {

template <typename T>
void otttKeyGen(std::span<std::uint8_t> key, std::span<const T> alpha, std::size_t bitWidthIn)
{
    int ret = FastFss_cpu_otttKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), bitWidthIn,
                                     sizeof(T), alpha.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_otttKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void otttLutEval(std::span<T>                  sharedOutE,
                 std::span<T>                  sharedOutT,
                 std::span<const T>            maskedX,
                 std::span<const std::uint8_t> key,
                 int                           partyId,
                 std::span<const T>            lookUpTable,
                 std::size_t                   bitWidthIn)
{
    int ret = FastFss_cpu_otttLutEval(sharedOutE.data(), sharedOutE.size() * sizeof(T), sharedOutT.data(),
                                      sharedOutT.size() * sizeof(T), maskedX.data(), maskedX.size() * sizeof(T),
                                      key.data(), key.size(), partyId, lookUpTable.data(),
                                      lookUpTable.size() * sizeof(T), bitWidthIn, sizeof(T), maskedX.size());
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_otttLutEval failed. error code: " + std::to_string(ret));
    }
}

} // namespace cpu

namespace cuda {

template <typename T>
void otttKeyGen(std::span<std::uint8_t> key, std::span<const T> alpha, std::size_t bitWidthIn, void *cudaStreamPtr)
{
    int ret = FastFss_cuda_otttKeyGen(key.data(), key.size(), alpha.data(), alpha.size() * sizeof(T), bitWidthIn,
                                      sizeof(T), alpha.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_otttKeyGen failed. error code: " + std::to_string(ret));
    }
}

template <typename T>
void otttLutEval(std::span<T>                  sharedOutE,
                 std::span<T>                  sharedOutT,
                 std::span<const T>            maskedX,
                 std::span<const std::uint8_t> key,
                 int                           partyId,
                 std::span<const T>            lookUpTable,
                 std::size_t                   bitWidthIn,
                 void                         *cudaStreamPtr)
{
    int ret = FastFss_cuda_otttLutEval(
        sharedOutE.data(), sharedOutE.size() * sizeof(T), sharedOutT.data(), sharedOutT.size() * sizeof(T),
        maskedX.data(), maskedX.size() * sizeof(T), key.data(), key.size(), partyId, lookUpTable.data(),
        lookUpTable.size() * sizeof(T), bitWidthIn, sizeof(T), maskedX.size(), cudaStreamPtr);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_otttLutEval failed. error code: " + std::to_string(ret));
    }
}

} // namespace cuda

} // namespace FastFss::ottt

#endif
