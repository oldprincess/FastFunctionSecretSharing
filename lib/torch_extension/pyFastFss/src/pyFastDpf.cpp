#include "pyFastDpf.h"

#include <FastFss/cpu/dpf.h>
#include <FastFss/cpu/mic.h>
#include <FastFss/cuda/dpf.h>
#include <FastFss/cuda/mic.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define ERR_LOG(fmt, ...)                                                 \
    std::fprintf(stderr, "[FastFss DPF] " fmt ". %s:%d\n", ##__VA_ARGS__, \
                 __FILE__, __LINE__)

#define ARG_ASSERT(exp)                                    \
    if (!(exp))                                            \
    {                                                      \
        ERR_LOG("assert fail: " #exp);                     \
        throw std::invalid_argument("assert fail: " #exp); \
    }

#define CHECK_ERROR_CODE(ret, func)             \
    if (ret != 0)                               \
    {                                           \
        ERR_LOG(func " ret = %d", ret);         \
        throw std::runtime_error(func " fail"); \
    }

namespace pyFastFss {

std::size_t dpf_get_key_data_size(std::size_t bitWidthIn,
                                  std::size_t bitWidthOut,
                                  std::size_t elementSize,
                                  std::size_t elementNum)
{
    std::size_t dataSize;
    int ret = FastFss_cpu_dpfGetKeyDataSize(&dataSize, bitWidthIn, bitWidthOut,
                                            elementSize, elementNum);
    CHECK_ERROR_CODE(ret, "FastFss_cpu_dpfGetKeyDataSize");
    return dataSize;
}

torch::Tensor& dpf_key_gen(torch::Tensor&       keyOut,
                           const torch::Tensor& alpha,
                           const torch::Tensor& beta,
                           const torch::Tensor& seed0,
                           const torch::Tensor& seed1,
                           std::size_t          bitWidthIn,
                           std::size_t          bitWidthOut,
                           std::size_t          elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    ARG_ASSERT(keyOut.is_contiguous());
    ARG_ASSERT(alpha.is_contiguous());
    ARG_ASSERT(beta.is_contiguous());
    ARG_ASSERT(seed0.is_contiguous());
    ARG_ASSERT(seed1.is_contiguous());

    ARG_ASSERT((std::size_t)alpha.numel() == elementNum);

    if ((std::size_t)beta.numel() != 0)
    {
        ARG_ASSERT((std::size_t)beta.numel() == elementNum);
    }
    ARG_ASSERT((std::size_t)seed0.numel() == 16 * elementNum);
    ARG_ASSERT((std::size_t)seed1.numel() == 16 * elementNum);

    ARG_ASSERT(keyOut.dtype() == torch::kUInt8);
    ARG_ASSERT(seed0.dtype() == torch::kUInt8);
    ARG_ASSERT(seed1.dtype() == torch::kUInt8);

    ARG_ASSERT(alpha.dtype() == beta.dtype());

    std::size_t elementSize = alpha.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);
    ARG_ASSERT(bitWidthOut <= elementSize * 8);

    auto device = alpha.device();
    ARG_ASSERT(keyOut.device() == device);
    ARG_ASSERT(beta.device() == device);
    ARG_ASSERT(seed0.device() == device);
    ARG_ASSERT(seed1.device() == device);

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    std::size_t dpfKeyDataSize = dpf_get_key_data_size(  //
        bitWidthIn, bitWidthOut, elementSize, elementNum //
    );                                                   //

    keyOut.resize_({(std::int64_t)dpfKeyDataSize});

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dpfKeyGen(              //
            keyOut.mutable_data_ptr(),                //
            (std::size_t)keyOut.numel(),              //
            alpha.const_data_ptr(),                   //
            (std::size_t)alpha.numel() * elementSize, //
            beta.const_data_ptr(),                    //
            (std::size_t)beta.numel() * elementSize,  //
            seed0.const_data_ptr(),                   //
            (std::size_t)seed0.numel(),               //
            seed1.const_data_ptr(),                   //
            (std::size_t)seed1.numel(),               //
            bitWidthIn,                               //
            bitWidthOut,                              //
            elementSize,                              //
            elementNum                                //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cpu_dpfKeyGen");
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_dpfKeyGen(             //
            keyOut.mutable_data_ptr(),                //
            (std::size_t)keyOut.numel(),              //
            alpha.const_data_ptr(),                   //
            (std::size_t)alpha.numel() * elementSize, //
            beta.const_data_ptr(),                    //
            (std::size_t)beta.numel() * elementSize,  //
            seed0.const_data_ptr(),                   //
            (std::size_t)seed0.numel(),               //
            seed1.const_data_ptr(),                   //
            (std::size_t)seed1.numel(),               //
            bitWidthIn,                               //
            bitWidthOut,                              //
            elementSize,                              //
            elementNum,                               //
            &stream);
        CHECK_ERROR_CODE(ret, "FastFss_cuda_dpfKeyGen");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return keyOut;
}

torch::Tensor& dpf_eval(torch::Tensor&       sharedOut,
                        const torch::Tensor& maskedX,
                        const torch::Tensor& key,
                        const torch::Tensor& seed,
                        int                  partyId,
                        std::size_t          bitWidthIn,
                        std::size_t          bitWidthOut,
                        std::size_t          elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    ARG_ASSERT(sharedOut.is_contiguous());
    ARG_ASSERT(maskedX.is_contiguous());
    ARG_ASSERT(key.is_contiguous());
    ARG_ASSERT(seed.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);

    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);

    ARG_ASSERT(sharedOut.dtype() == maskedX.dtype());

    std::size_t elementSize = maskedX.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);
    ARG_ASSERT(bitWidthOut <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);

    ARG_ASSERT((std::size_t)key.numel() ==
               dpf_get_key_data_size(bitWidthIn, bitWidthOut, elementSize,
                                     elementNum));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    sharedOut.resize_(maskedX.sizes());

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dpfEval(                  //
            sharedOut.mutable_data_ptr(),               //
            maskedX.const_data_ptr(),                   //
            (std::size_t)maskedX.numel() * elementSize, //
            key.const_data_ptr(),                       //
            (std::size_t)key.numel(),                   //
            seed.const_data_ptr(),                      //
            (std::size_t)seed.numel(),                  //
            partyId,                                    //
            bitWidthIn,                                 //
            bitWidthOut,                                //
            elementSize,                                //
            elementNum, nullptr, 0);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_dpfEval");
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_dpfEval(                 //
            sharedOut.mutable_data_ptr(),               //
            maskedX.const_data_ptr(),                   //
            (std::size_t)maskedX.numel() * elementSize, //
            key.const_data_ptr(),                       //
            (std::size_t)key.numel(),                   //
            seed.const_data_ptr(),                      //
            (std::size_t)seed.numel(),                  //
            partyId,                                    //
            bitWidthIn,                                 //
            bitWidthOut,                                //
            elementSize,                                //
            elementNum, nullptr, 0, &stream);
        CHECK_ERROR_CODE(ret, "FastFss_cuda_dpfEval");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedOut;
}

torch::Tensor& dpf_eval_multi(torch::Tensor&       sharedOut,
                              const torch::Tensor& maskedX,
                              const torch::Tensor& key,
                              const torch::Tensor& seed,
                              int                  partyId,
                              const torch::Tensor& point,
                              std::size_t          bitWidthIn,
                              std::size_t          bitWidthOut,
                              std::size_t          elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    ARG_ASSERT(sharedOut.is_contiguous());
    ARG_ASSERT(maskedX.is_contiguous());
    ARG_ASSERT(key.is_contiguous());
    ARG_ASSERT(seed.is_contiguous());
    ARG_ASSERT(point.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);

    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);

    ARG_ASSERT(sharedOut.dtype() == maskedX.dtype());

    std::size_t elementSize = maskedX.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);
    ARG_ASSERT(bitWidthOut <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);

    ARG_ASSERT((std::size_t)key.numel() ==
               dpf_get_key_data_size(bitWidthIn, bitWidthOut, elementSize,
                                     elementNum));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    std::size_t cacheSize;
    {
        int ret = FastFss_cpu_dpfGetCacheDataSize(
            &cacheSize, bitWidthIn, bitWidthOut, elementSize, elementNum);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_dpfGetCacheDataSize");
    }
    torch::TensorOptions options;
    options             = options.dtype(torch::kUInt8).device(device.type());
    torch::Tensor cache = torch::empty({(std::int64_t)(cacheSize)}, options);

    sharedOut.resize_(maskedX.numel() * point.numel());

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dpfEvalMulti(             //
            sharedOut.mutable_data_ptr(),               //
            sharedOut.numel() * elementSize,            //
            maskedX.const_data_ptr(),                   //
            (std::size_t)maskedX.numel() * elementSize, //
            key.const_data_ptr(),                       //
            (std::size_t)key.numel(),                   //
            seed.const_data_ptr(),                      //
            (std::size_t)seed.numel(),                  //
            partyId,                                    //
            point.const_data_ptr(),                     //
            point.numel() * elementSize,                //
            bitWidthIn,                                 //
            bitWidthOut,                                //
            elementSize,                                //
            elementNum,                                 //
            cache.mutable_data_ptr(), cacheSize);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_dpfEvalMulti");
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_dpfEvalMulti(            //
            sharedOut.mutable_data_ptr(),               //
            sharedOut.numel() * elementSize,            //
            maskedX.const_data_ptr(),                   //
            (std::size_t)maskedX.numel() * elementSize, //
            key.const_data_ptr(),                       //
            (std::size_t)key.numel(),                   //
            seed.const_data_ptr(),                      //
            (std::size_t)seed.numel(),                  //
            partyId,                                    //
            point.const_data_ptr(),                     //
            point.numel() * elementSize,                //
            bitWidthIn,                                 //
            bitWidthOut,                                //
            elementSize,                                //
            elementNum,                                 //
            cache.mutable_data_ptr(), cacheSize,        //
            &stream);
        CHECK_ERROR_CODE(ret, "FastFss_cuda_dpfEvalMulti");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedOut;
}

} // namespace pyFastFss
