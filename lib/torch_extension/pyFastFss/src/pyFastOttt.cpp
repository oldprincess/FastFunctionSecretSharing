#include "pyFastOttt.h"

#include <FastFss/cpu/ottt.h>
#include <FastFss/cuda/ottt.h>
#include <FastFss/ottt.h>
#ifndef NO_CUDA
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define ERR_LOG(fmt, ...) std::fprintf(stderr, "[FastFss OTTT] " fmt ". %s:%d\n", ##__VA_ARGS__, __FILE__, __LINE__)

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

std::size_t ottt_get_key_data_size(std::size_t bitWidthIn, std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         result      = FastFss_otttGetKeyDataSize(&keyDataSize, bitWidthIn, elementNum);
    CHECK_ERROR_CODE(result, "FastFss_otttGetKeyDataSize");
    return keyDataSize;
}

torch::Tensor &ottt_key_gen(torch::Tensor       &keyInOut,
                            const torch::Tensor &alpha,
                            std::size_t          bitWidthIn,
                            std::size_t          elementNum)
{
    ARG_ASSERT(keyInOut.is_contiguous());
    ARG_ASSERT(alpha.is_contiguous());
    const auto alphaLayout = inspect_value_tensor(alpha, bitWidthIn);
    ARG_ASSERT(alphaLayout.logicalElementNum == elementNum);
    ARG_ASSERT(keyInOut.dtype() == torch::kUInt8);

    std::size_t elementSize = alphaLayout.elementSize;
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = alpha.device();
    ARG_ASSERT(keyInOut.device() == device);
    ARG_ASSERT((std::size_t)keyInOut.numel() == ottt_get_key_data_size(bitWidthIn, elementNum));

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_otttKeyGen(keyInOut.mutable_data_ptr(),                       //
                                         (std::size_t)keyInOut.numel(),                     //
                                         alpha.const_data_ptr(),                            //
                                         (std::size_t)alpha.numel() * alpha.element_size(), //
                                         bitWidthIn,                                        //
                                         elementSize,                                       //
                                         elementNum                                         //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cpu_otttKeyGen");
    }
#ifndef NO_CUDA
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_otttKeyGen(keyInOut.mutable_data_ptr(),                       //
                                          (std::size_t)keyInOut.numel(),                     //
                                          alpha.const_data_ptr(),                            //
                                          (std::size_t)alpha.numel() * alpha.element_size(), //
                                          bitWidthIn,                                        //
                                          elementSize,                                       //
                                          elementNum,                                        //
                                          &stream                                            //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cuda_otttKeyGen");
    }
#endif
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return keyInOut;
}

py::tuple ottt_lut_eval(torch::Tensor       &sharedOutE,
                        torch::Tensor       &sharedOutT,
                        const torch::Tensor &maskedX,
                        const torch::Tensor &key,
                        int                  partyId,
                        const torch::Tensor &lookUpTable,
                        std::size_t          bitWidthIn,
                        std::size_t          bitWidthOut,
                        std::size_t          elementNum)
{
    ARG_ASSERT(sharedOutE.is_contiguous());
    ARG_ASSERT(sharedOutT.is_contiguous());
    ARG_ASSERT(maskedX.is_contiguous());
    ARG_ASSERT(key.is_contiguous());
    ARG_ASSERT(lookUpTable.is_contiguous());

    const auto valueBitWidth = max_bit_width({bitWidthIn, bitWidthOut});
    const auto maskedLayout  = inspect_value_tensor(maskedX, valueBitWidth);
    const auto lutLayout     = inspect_value_tensor(lookUpTable, valueBitWidth);

    ARG_ASSERT(maskedLayout.logicalElementNum == elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8);

    auto dtype = maskedX.dtype();
    ARG_ASSERT(sharedOutE.dtype() == dtype);
    ARG_ASSERT(sharedOutT.dtype() == dtype);
    ARG_ASSERT(lookUpTable.dtype() == dtype);

    std::size_t elementSize = maskedLayout.elementSize;
    ARG_ASSERT(bitWidthIn <= elementSize * 8);
    ARG_ASSERT(bitWidthOut <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOutE.device() == device);
    ARG_ASSERT(sharedOutT.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(lookUpTable.device() == device);

    ARG_ASSERT((std::size_t)key.numel() == ottt_get_key_data_size(bitWidthIn, elementNum));

    sharedOutE.resize_(make_value_shape(maskedLayout.logicalShape, valueBitWidth));
    sharedOutT.resize_(make_value_shape(maskedLayout.logicalShape, valueBitWidth));

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_otttLutEval(sharedOutE.mutable_data_ptr(),                                 //
                                          (std::size_t)sharedOutE.numel() * sharedOutE.element_size(),   //
                                          sharedOutT.mutable_data_ptr(),                                 //
                                          (std::size_t)sharedOutT.numel() * sharedOutT.element_size(),   //
                                          maskedX.const_data_ptr(),                                      //
                                          (std::size_t)maskedX.numel() * maskedX.element_size(),         //
                                          key.const_data_ptr(),                                          //
                                          (std::size_t)key.numel(),                                      //
                                          partyId,                                                       //
                                          lookUpTable.const_data_ptr(),                                  //
                                          (std::size_t)lookUpTable.numel() * lookUpTable.element_size(), //
                                          bitWidthIn,                                                    //
                                          elementSize,                                                   //
                                          elementNum                                                     //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cpu_otttLutEval");
    }
#ifndef NO_CUDA
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_otttLutEval(sharedOutE.mutable_data_ptr(),                                 //
                                           (std::size_t)sharedOutE.numel() * sharedOutE.element_size(),   //
                                           sharedOutT.mutable_data_ptr(),                                 //
                                           (std::size_t)sharedOutT.numel() * sharedOutT.element_size(),   //
                                           maskedX.const_data_ptr(),                                      //
                                           (std::size_t)maskedX.numel() * maskedX.element_size(),         //
                                           key.const_data_ptr(),                                          //
                                           (std::size_t)key.numel(),                                      //
                                           partyId,                                                       //
                                           lookUpTable.const_data_ptr(),                                  //
                                           (std::size_t)lookUpTable.numel() * lookUpTable.element_size(), //
                                           bitWidthIn,                                                    //
                                           elementSize,                                                   //
                                           elementNum,                                                    //
                                           &stream                                                        //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cuda_otttLutEval");
    }
#endif
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return py::make_tuple(sharedOutE, sharedOutT);
}

} // namespace pyFastFss
