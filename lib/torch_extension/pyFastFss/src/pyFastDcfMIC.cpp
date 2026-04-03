#include "pyFastDcfMIC.h"

#include <FastFss/cpu/dcf.h>
#include <FastFss/cpu/mic.h>
#include <FastFss/cuda/dcf.h>
#include <FastFss/cuda/mic.h>
#include <FastFss/mic.h>
#ifndef NO_CUDA
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define ERR_LOG(fmt, ...) std::fprintf(stderr, "[FastFss DCF MIC] " fmt ". %s:%d\n", ##__VA_ARGS__, __FILE__, __LINE__)

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
    int         ret    = FastFss_dcfMICGetKeyDataSize(&result, bitWidthIn, bitWidthOut, elementSize, elementNum);
    CHECK_ERROR_CODE(ret, "FastFss_dcfMICGetKeyDataSize");
    return result;
}

std::size_t dcf_mic_get_cache_data_size(std::size_t bitWidthIn,
                                        std::size_t bitWidthOut,
                                        std::size_t elementSize,
                                        std::size_t elementNum)
{
    std::size_t result = 0;
    int         ret    = FastFss_dcfMICGetCacheDataSize(&result, bitWidthIn, bitWidthOut, elementSize, elementNum);
    CHECK_ERROR_CODE(ret, "FastFss_dcfMICGetCacheDataSize");
    return result;
}

py::tuple dcf_mic_key_gen(torch::Tensor       &keyOut,
                          torch::Tensor       &zOut,
                          const torch::Tensor &alpha,
                          const torch::Tensor &seed0,
                          const torch::Tensor &seed1,
                          const torch::Tensor &leftBoundary,
                          const torch::Tensor &rightBoundary,
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

    const auto valueBitWidth = max_bit_width({bitWidthIn, bitWidthOut});
    const auto alphaLayout   = inspect_value_tensor(alpha, valueBitWidth);
    const auto leftLayout    = inspect_value_tensor(leftBoundary, valueBitWidth);
    const auto rightLayout   = inspect_value_tensor(rightBoundary, valueBitWidth);

    ARG_ASSERT(alphaLayout.logicalElementNum == elementNum);
    ARG_ASSERT((std::size_t)seed0.numel() == 16 * elementNum);
    ARG_ASSERT((std::size_t)seed1.numel() == 16 * elementNum);

    ARG_ASSERT(keyOut.dtype() == torch::kUInt8);
    ARG_ASSERT(seed0.dtype() == torch::kUInt8);
    ARG_ASSERT(seed1.dtype() == torch::kUInt8);

    auto dtype = alpha.dtype();
    ARG_ASSERT(zOut.dtype() == dtype);
    ARG_ASSERT(leftBoundary.dtype() == dtype);
    ARG_ASSERT(rightBoundary.dtype() == dtype);

    std::size_t elementSize = alphaLayout.elementSize;

    ARG_ASSERT(bitWidthIn <= elementSize * 8);
    ARG_ASSERT(bitWidthOut <= elementSize * 8);

    auto device = alpha.device();

    ARG_ASSERT(keyOut.device() == device);
    ARG_ASSERT(zOut.device() == device);
    ARG_ASSERT(seed0.device() == device);
    ARG_ASSERT(seed1.device() == device);
    ARG_ASSERT(leftBoundary.device() == device);
    ARG_ASSERT(rightBoundary.device() == device);

    assert_same_logical_shape(leftLayout, rightLayout,
                              "leftBoundary and rightBoundary must have the same logical shape");

    std::size_t intervalNum = leftLayout.logicalElementNum;

    zOut.resize_(make_value_shape(append_logical_dim(alphaLayout.logicalShape, static_cast<std::int64_t>(intervalNum)),
                                  valueBitWidth));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    std::size_t dcfMICKeyDataSize = dcf_mic_get_key_data_size( //
        bitWidthIn, bitWidthOut, elementSize, elementNum       //
    );                                                         //

    keyOut.resize_({(std::int64_t)dcfMICKeyDataSize});

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dcfMICKeyGen(keyOut.mutable_data_ptr(),                                         //
                                           (std::size_t)keyOut.numel(),                                       //
                                           zOut.mutable_data_ptr(),                                           //
                                           (std::size_t)zOut.numel() * zOut.element_size(),                   //
                                           alpha.const_data_ptr(),                                            //
                                           (std::size_t)alpha.numel() * alpha.element_size(),                 //
                                           seed0.const_data_ptr(),                                            //
                                           (std::size_t)seed0.numel(),                                        //
                                           seed1.const_data_ptr(),                                            //
                                           (std::size_t)seed1.numel(),                                        //
                                           leftBoundary.const_data_ptr(),                                     //
                                           (std::size_t)leftBoundary.numel() * leftBoundary.element_size(),   //
                                           rightBoundary.const_data_ptr(),                                    //
                                           (std::size_t)rightBoundary.numel() * rightBoundary.element_size(), //
                                           bitWidthIn,                                                        //
                                           bitWidthOut,                                                       //
                                           elementSize,                                                       //
                                           elementNum                                                         //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cpu_dcfMICKeyGen");
    }
#ifndef NO_CUDA
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_dcfMICKeyGen(keyOut.mutable_data_ptr(),                                         //
                                            (std::size_t)keyOut.numel(),                                       //
                                            zOut.mutable_data_ptr(),                                           //
                                            (std::size_t)zOut.numel() * zOut.element_size(),                   //
                                            alpha.const_data_ptr(),                                            //
                                            (std::size_t)alpha.numel() * alpha.element_size(),                 //
                                            seed0.const_data_ptr(),                                            //
                                            (std::size_t)seed0.numel(),                                        //
                                            seed1.const_data_ptr(),                                            //
                                            (std::size_t)seed1.numel(),                                        //
                                            leftBoundary.const_data_ptr(),                                     //
                                            (std::size_t)leftBoundary.numel() * leftBoundary.element_size(),   //
                                            rightBoundary.const_data_ptr(),                                    //
                                            (std::size_t)rightBoundary.numel() * rightBoundary.element_size(), //
                                            bitWidthIn,                                                        //
                                            bitWidthOut,                                                       //
                                            elementSize,                                                       //
                                            elementNum,                                                        //
                                            &stream                                                            //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cuda_dcfMICKeyGen");
    }
#endif
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return py::make_tuple(keyOut, zOut);
}

torch::Tensor &dcf_mic_eval(torch::Tensor       &sharedOut,
                            const torch::Tensor &maskedX,
                            const torch::Tensor &key,
                            const torch::Tensor &sharedZ,
                            const torch::Tensor &seed,
                            int                  partyId,
                            const torch::Tensor &leftBoundary,
                            const torch::Tensor &rightBoundary,
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

    const auto valueBitWidth = max_bit_width({bitWidthIn, bitWidthOut});
    const auto maskedLayout  = inspect_value_tensor(maskedX, valueBitWidth);
    const auto leftLayout    = inspect_value_tensor(leftBoundary, valueBitWidth);
    const auto rightLayout   = inspect_value_tensor(rightBoundary, valueBitWidth);
    const auto sharedZLayout = inspect_value_tensor(sharedZ, valueBitWidth);

    ARG_ASSERT(maskedLayout.logicalElementNum == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);

    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);

    auto dtype = maskedX.dtype();
    ARG_ASSERT(sharedOut.dtype() == dtype);
    ARG_ASSERT(sharedZ.dtype() == dtype);
    ARG_ASSERT(leftBoundary.dtype() == dtype);
    ARG_ASSERT(rightBoundary.dtype() == dtype);

    std::size_t elementSize = maskedLayout.elementSize;
    ARG_ASSERT(bitWidthIn <= elementSize * 8);
    ARG_ASSERT(bitWidthOut <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(sharedZ.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT(leftBoundary.device() == device);
    ARG_ASSERT(rightBoundary.device() == device);

    assert_same_logical_shape(leftLayout, rightLayout,
                              "leftBoundary and rightBoundary must have the same logical shape");

    std::size_t intervalNum = leftLayout.logicalElementNum;
    ARG_ASSERT(sharedZLayout.logicalShape ==
               append_logical_dim(maskedLayout.logicalShape, static_cast<std::int64_t>(intervalNum)));

    std::size_t keyDataSize = dcf_mic_get_key_data_size(bitWidthIn, bitWidthOut, elementSize, elementNum);
    ARG_ASSERT((std::size_t)key.numel() == keyDataSize);

    std::size_t          cacheSize = dcf_mic_get_cache_data_size(bitWidthIn, bitWidthOut, elementSize, elementNum);
    torch::TensorOptions options;
    options             = options.dtype(torch::kUInt8).device(device.type());
    torch::Tensor cache = torch::empty({(std::int64_t)(cacheSize)}, options);

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    sharedOut.resize_(make_value_shape(
        append_logical_dim(maskedLayout.logicalShape, static_cast<std::int64_t>(intervalNum)), valueBitWidth));

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dcfMICEval(                                      //
            sharedOut.mutable_data_ptr(),                                      //
            (std::size_t)sharedOut.numel() * sharedOut.element_size(),         //
            maskedX.const_data_ptr(),                                          //
            (std::size_t)maskedX.numel() * maskedX.element_size(),             //
            key.const_data_ptr(),                                              //
            (std::size_t)key.numel(),                                          //
            sharedZ.const_data_ptr(),                                          //
            (std::size_t)sharedZ.numel() * sharedZ.element_size(),             //
            seed.const_data_ptr(),                                             //
            (std::size_t)seed.numel(),                                         //
            partyId,                                                           //
            leftBoundary.const_data_ptr(),                                     //
            (std::size_t)leftBoundary.numel() * leftBoundary.element_size(),   //
            rightBoundary.const_data_ptr(),                                    //
            (std::size_t)rightBoundary.numel() * rightBoundary.element_size(), //
            bitWidthIn,                                                        //
            bitWidthOut,                                                       //
            elementSize,                                                       //
            elementNum,                                                        //
            cache.mutable_data_ptr(),                                          //
            cache.numel());
        CHECK_ERROR_CODE(ret, "FastFss_cpu_dcfMICEval");
    }
#ifndef NO_CUDA
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_dcfMICEval(                                     //
            sharedOut.mutable_data_ptr(),                                      //
            (std::size_t)sharedOut.numel() * sharedOut.element_size(),         //
            maskedX.const_data_ptr(),                                          //
            (std::size_t)maskedX.numel() * maskedX.element_size(),             //
            key.const_data_ptr(),                                              //
            (std::size_t)key.numel(),                                          //
            sharedZ.const_data_ptr(),                                          //
            (std::size_t)sharedZ.numel() * sharedZ.element_size(),             //
            seed.const_data_ptr(),                                             //
            (std::size_t)seed.numel(),                                         //
            partyId,                                                           //
            leftBoundary.const_data_ptr(),                                     //
            (std::size_t)leftBoundary.numel() * leftBoundary.element_size(),   //
            rightBoundary.const_data_ptr(),                                    //
            (std::size_t)rightBoundary.numel() * rightBoundary.element_size(), //
            bitWidthIn,                                                        //
            bitWidthOut,                                                       //
            elementSize,                                                       //
            elementNum,                                                        //
            cache.mutable_data_ptr(),                                          //
            cache.numel(),                                                     //
            &stream);
        CHECK_ERROR_CODE(ret, "FastFss_cuda_dcfMICEval");
    }
#endif
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedOut;
}

} // namespace pyFastFss
