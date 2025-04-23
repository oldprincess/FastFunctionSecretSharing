#include <FastFss/cpu/onehot.h>
#include <FastFss/cuda/onehot.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "pyFastFss.h"

#define ERR_LOG(fmt, ...)                                                    \
    std::fprintf(stderr, "[FastFss Grotto] " fmt ". %s:%d\n", ##__VA_ARGS__, \
                 __FILE__, __LINE__)

#define ARG_ASSERT(exp)                                    \
    if (!(exp))                                            \
    {                                                      \
        ERR_LOG("assert fail: " #exp);                     \
        throw std::invalid_argument("assert fail: " #exp); \
    }

namespace pyFastFss {

std::size_t onehot_get_key_data_size(std::size_t bitWidthIn,
                                     std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         result =
        FastFss_cpu_onehotGetKeyDataSize(&keyDataSize, bitWidthIn, elementNum);
    if (result < 0)
    {
        ERR_LOG("FastFss_cpu_onehotGetKeyDataSize ret = %d", result);
        throw std::runtime_error("FastFss_cpu_onehotGetKeyDataSize fail");
    }
    return keyDataSize;
}

void onehot_key_gen(torch::Tensor&       keyInOut,
                    const torch::Tensor& alpha,
                    std::size_t          bitWidthIn,
                    std::size_t          elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    ARG_ASSERT(keyInOut.is_contiguous());
    ARG_ASSERT(alpha.is_contiguous());

    ARG_ASSERT((std::size_t)alpha.numel() == elementNum);

    ARG_ASSERT(keyInOut.dtype() == torch::kUInt8);

    std::size_t elementSize = alpha.element_size();

    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = alpha.device();

    ARG_ASSERT(keyInOut.device() == device);

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    std::size_t onehotKeyDataSize = onehot_get_key_data_size( //
        bitWidthIn, elementNum                                //
    );                                                        //

    ARG_ASSERT((std::size_t)keyInOut.numel() == onehotKeyDataSize);

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_onehotKeyGen(
            keyInOut.mutable_data_ptr(),              //
            (std::size_t)keyInOut.numel(),            //
            alpha.const_data_ptr(),                   //
            (std::size_t)alpha.numel() * elementSize, //
            bitWidthIn,                               //
            elementSize,                              //
            elementNum                                //
        );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_onehotKeyGen ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_onehotKeyGen fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_onehotKeyGen(
            keyInOut.mutable_data_ptr(),              //
            (std::size_t)keyInOut.numel(),            //
            alpha.const_data_ptr(),                   //
            (std::size_t)alpha.numel() * elementSize, //
            bitWidthIn,                               //
            elementSize,                              //
            elementNum,                               //
            &stream                                   //
        );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_onehotKeyGen ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_onehotKeyGen fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

void onehot_lut_eval(torch::Tensor&       sharedOutE,
                     torch::Tensor&       sharedOutT,
                     const torch::Tensor& maskedX,
                     const torch::Tensor& key,
                     int                  partyId,
                     const torch::Tensor& lookUpTable,
                     std::size_t          bitWidthIn,
                     std::size_t          bitWidthOut,
                     std::size_t          elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================
    ARG_ASSERT(sharedOutE.is_contiguous() && //
               sharedOutT.is_contiguous() && //
               maskedX.is_contiguous() &&    //
               key.is_contiguous() &&        //
               lookUpTable.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8);

    auto dtype = maskedX.dtype();
    ARG_ASSERT(sharedOutE.dtype() == dtype && sharedOutT.dtype() == dtype &&
               lookUpTable.dtype() == dtype);

    std::size_t elementSize = maskedX.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOutT.device() == device && sharedOutE.device() == device &&
               key.device() == device && lookUpTable.device() == device);

    ARG_ASSERT((std::size_t)key.numel() ==
               onehot_get_key_data_size(bitWidthIn, elementNum));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    if ((std::size_t)sharedOutE.numel() != elementNum)
    {
        sharedOutE.resize_({(std::int64_t)(elementNum)});
    }
    if ((std::size_t)sharedOutT.numel() != elementNum)
    {
        sharedOutT.resize_({(std::int64_t)(elementNum)});
    }

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_onehotLutEval(                //
            sharedOutE.mutable_data_ptr(),                  //
            sharedOutT.mutable_data_ptr(),                  //
            maskedX.const_data_ptr(),                       //
            (std::size_t)maskedX.numel() * elementSize,     //
            key.const_data_ptr(),                           //
            (std::size_t)key.numel(),                       //
            partyId,                                        //
            lookUpTable.const_data_ptr(),                   //
            (std::size_t)lookUpTable.numel() * elementSize, //
            bitWidthIn,                                     //
            elementSize,                                    //
            elementNum);
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_onehotLutEval fail ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_onehotLutEval fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_onehotLutEval(               //
            sharedOutE.mutable_data_ptr(),                  //
            sharedOutT.mutable_data_ptr(),                  //
            maskedX.const_data_ptr(),                       //
            (std::size_t)maskedX.numel() * elementSize,     //
            key.const_data_ptr(),                           //
            (std::size_t)key.numel(),                       //
            partyId,                                        //
            lookUpTable.const_data_ptr(),                   //
            (std::size_t)lookUpTable.numel() * elementSize, //
            bitWidthIn,                                     //
            elementSize,                                    //
            elementNum, &stream);
        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_onehotLutEval ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_onehotLutEval fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

} // namespace pyFastFss
