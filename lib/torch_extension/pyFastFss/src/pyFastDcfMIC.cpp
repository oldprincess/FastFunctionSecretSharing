#include "pyFastDcfMIC.h"

#include <FastFss/cpu/dcf.h>
#include <FastFss/cpu/mic.h>
#include <FastFss/cuda/dcf.h>
#include <FastFss/cuda/mic.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define ERR_LOG(fmt, ...)                                                     \
    std::fprintf(stderr, "[FastFss DCF MIC] " fmt ". %s:%d\n", ##__VA_ARGS__, \
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

std::size_t dcf_mic_get_key_data_size(std::size_t bitWidthIn,
                                      std::size_t bitWidthOut,
                                      std::size_t elementSize,
                                      std::size_t elementNum)
{
    std::size_t result = 0;
    int ret = FastFss_cpu_dcfMICGetKeyDataSize(&result, bitWidthIn, bitWidthOut,
                                               elementSize, elementNum);
    CHECK_ERROR_CODE(ret, "FastFss_cpu_dcfMICGetKeyDataSize");
    return result;
}

py::tuple dcf_mic_key_gen(torch::Tensor&       keyOut,
                          torch::Tensor&       zOut,
                          const torch::Tensor& alpha,
                          const torch::Tensor& seed0,
                          const torch::Tensor& seed1,
                          const torch::Tensor& leftBoundary,
                          const torch::Tensor& rightBoundary,
                          std::size_t          bitWidthIn,
                          std::size_t          bitWidthOut,
                          std::size_t          elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    ARG_ASSERT(keyOut.is_contiguous());
    ARG_ASSERT(zOut.is_contiguous());
    ARG_ASSERT(alpha.is_contiguous());
    ARG_ASSERT(seed0.is_contiguous());
    ARG_ASSERT(seed1.is_contiguous());
    ARG_ASSERT(leftBoundary.is_contiguous());
    ARG_ASSERT(rightBoundary.is_contiguous());

    ARG_ASSERT((std::size_t)alpha.numel() == elementNum);
    ARG_ASSERT((std::size_t)seed0.numel() == 16 * elementNum);
    ARG_ASSERT((std::size_t)seed1.numel() == 16 * elementNum);

    ARG_ASSERT(keyOut.dtype() == torch::kUInt8);
    ARG_ASSERT(seed0.dtype() == torch::kUInt8);
    ARG_ASSERT(seed1.dtype() == torch::kUInt8);

    auto dtype = alpha.dtype();
    ARG_ASSERT(zOut.dtype() == dtype);
    ARG_ASSERT(leftBoundary.dtype() == dtype);
    ARG_ASSERT(rightBoundary.dtype() == dtype);

    std::size_t elementSize = alpha.element_size();

    ARG_ASSERT(bitWidthIn <= elementSize * 8);
    ARG_ASSERT(bitWidthOut <= elementSize * 8);

    auto device = alpha.device();

    ARG_ASSERT(keyOut.device() == device);
    ARG_ASSERT(zOut.device() == device);
    ARG_ASSERT(seed0.device() == device);
    ARG_ASSERT(seed1.device() == device);
    ARG_ASSERT(leftBoundary.device() == device);
    ARG_ASSERT(rightBoundary.device() == device);

    ARG_ASSERT(leftBoundary.numel() == rightBoundary.numel());

    std::size_t intervalNum = (std::size_t)leftBoundary.numel();

    zOut.resize_({(std::int64_t)(intervalNum * elementNum)});

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    std::size_t dcfMICKeyDataSize = dcf_mic_get_key_data_size( //
        bitWidthIn, bitWidthOut, elementSize, elementNum       //
    );                                                         //

    keyOut.resize_({(std::int64_t)dcfMICKeyDataSize});

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dcfMICKeyGen(
            keyOut.mutable_data_ptr(),                        //
            (std::size_t)keyOut.numel(),                      //
            zOut.mutable_data_ptr(),                          //
            (std::size_t)zOut.numel() * elementSize,          //
            alpha.const_data_ptr(),                           //
            (std::size_t)alpha.numel() * elementSize,         //
            seed0.const_data_ptr(),                           //
            (std::size_t)seed0.numel(),                       //
            seed1.const_data_ptr(),                           //
            (std::size_t)seed1.numel(),                       //
            leftBoundary.const_data_ptr(),                    //
            (std::size_t)leftBoundary.numel() * elementSize,  //
            rightBoundary.const_data_ptr(),                   //
            (std::size_t)rightBoundary.numel() * elementSize, //
            bitWidthIn,                                       //
            bitWidthOut,                                      //
            elementSize,                                      //
            elementNum                                        //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cpu_dcfMICKeyGen");
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_dcfMICKeyGen(
            keyOut.mutable_data_ptr(),                        //
            (std::size_t)keyOut.numel(),                      //
            zOut.mutable_data_ptr(),                          //
            (std::size_t)zOut.numel() * elementSize,          //
            alpha.const_data_ptr(),                           //
            (std::size_t)alpha.numel() * elementSize,         //
            seed0.const_data_ptr(),                           //
            (std::size_t)seed0.numel(),                       //
            seed1.const_data_ptr(),                           //
            (std::size_t)seed1.numel(),                       //
            leftBoundary.const_data_ptr(),                    //
            (std::size_t)leftBoundary.numel() * elementSize,  //
            rightBoundary.const_data_ptr(),                   //
            (std::size_t)rightBoundary.numel() * elementSize, //
            bitWidthIn,                                       //
            bitWidthOut,                                      //
            elementSize,                                      //
            elementNum,                                       //
            &stream                                           //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cuda_dcfMICKeyGen");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return py::make_tuple(keyOut, zOut);
}

torch::Tensor& dcf_mic_eval(torch::Tensor&       sharedOut,
                            const torch::Tensor& maskedX,
                            const torch::Tensor& key,
                            const torch::Tensor& sharedZ,
                            const torch::Tensor& seed,
                            int                  partyId,
                            const torch::Tensor& leftBoundary,
                            const torch::Tensor& rightBoundary,
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
    ARG_ASSERT(sharedZ.is_contiguous());
    ARG_ASSERT(seed.is_contiguous());
    ARG_ASSERT(leftBoundary.is_contiguous());
    ARG_ASSERT(rightBoundary.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);

    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);

    auto dtype = maskedX.dtype();
    ARG_ASSERT(sharedOut.dtype() == dtype);
    ARG_ASSERT(sharedZ.dtype() == dtype);
    ARG_ASSERT(leftBoundary.dtype() == dtype);
    ARG_ASSERT(rightBoundary.dtype() == dtype);

    std::size_t elementSize = maskedX.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);
    ARG_ASSERT(bitWidthOut <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(sharedZ.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT(leftBoundary.device() == device);
    ARG_ASSERT(rightBoundary.device() == device);

    ARG_ASSERT(leftBoundary.numel() == rightBoundary.numel());

    std::size_t intervalNum = (std::size_t)leftBoundary.numel();
    ARG_ASSERT((std::size_t)sharedZ.numel() == intervalNum * elementNum);

    std::size_t keyDataSize = dcf_mic_get_key_data_size(
        bitWidthIn, bitWidthOut, elementSize, elementNum);
    ARG_ASSERT((std::size_t)key.numel() == keyDataSize);

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    sharedOut.resize_({(std::int64_t)(intervalNum * elementNum)});

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dcfMICEval(                     //
            sharedOut.mutable_data_ptr(),                     //
            (std::size_t)sharedOut.numel() * elementSize,     //
            maskedX.const_data_ptr(),                         //
            (std::size_t)maskedX.numel() * elementSize,       //
            key.const_data_ptr(),                             //
            (std::size_t)key.numel(),                         //
            sharedZ.const_data_ptr(),                         //
            (std::size_t)sharedZ.numel() * elementSize,       //
            seed.const_data_ptr(),                            //
            (std::size_t)seed.numel(),                        //
            partyId,                                          //
            leftBoundary.const_data_ptr(),                    //
            (std::size_t)leftBoundary.numel() * elementSize,  //
            rightBoundary.const_data_ptr(),                   //
            (std::size_t)rightBoundary.numel() * elementSize, //
            bitWidthIn,                                       //
            bitWidthOut,                                      //
            elementSize,                                      //
            elementNum, nullptr, 0);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_dcfMICEval");
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_dcfMICEval(                    //
            sharedOut.mutable_data_ptr(),                     //
            (std::size_t)sharedOut.numel() * elementSize,     //
            maskedX.const_data_ptr(),                         //
            (std::size_t)maskedX.numel() * elementSize,       //
            key.const_data_ptr(),                             //
            (std::size_t)key.numel(),                         //
            sharedZ.const_data_ptr(),                         //
            (std::size_t)sharedZ.numel() * elementSize,       //
            seed.const_data_ptr(),                            //
            (std::size_t)seed.numel(),                        //
            partyId,                                          //
            leftBoundary.const_data_ptr(),                    //
            (std::size_t)leftBoundary.numel() * elementSize,  //
            rightBoundary.const_data_ptr(),                   //
            (std::size_t)rightBoundary.numel() * elementSize, //
            bitWidthIn,                                       //
            bitWidthOut,                                      //
            elementSize,                                      //
            elementNum, nullptr, 0, &stream);
        CHECK_ERROR_CODE(ret, "FastFss_cuda_dcfMICEval");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedOut;
}

} // namespace pyFastFss
