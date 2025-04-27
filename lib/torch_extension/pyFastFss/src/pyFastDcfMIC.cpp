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
        ERR_LOG(func "ret = %d", ret);          \
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

void dcf_mic_key_gen(torch::Tensor keyOut,
                     torch::Tensor zOut,
                     torch::Tensor alpha,
                     torch::Tensor seed0,
                     torch::Tensor seed1,
                     torch::Tensor leftBoundary,
                     torch::Tensor rightBoundary,
                     std::size_t   bitWidthIn,
                     std::size_t   bitWidthOut,
                     std::size_t   elementNum)
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
}

void dcf_mic_eval(torch::Tensor sharedOut,
                  torch::Tensor maskedX,
                  torch::Tensor key,
                  torch::Tensor sharedZ,
                  torch::Tensor seed,
                  int           partyId,
                  torch::Tensor leftBoundary,
                  torch::Tensor rightBoundary,
                  std::size_t   bitWidthIn,
                  std::size_t   bitWidthOut,
                  std::size_t   elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    if (!sharedOut.is_contiguous() || !maskedX.is_contiguous() ||
        !key.is_contiguous() || !sharedZ.is_contiguous() ||
        !seed.is_contiguous() || !leftBoundary.is_contiguous() ||
        !rightBoundary.is_contiguous())
    {
        std::fprintf(stderr,                                         //
                     "[FastFss] tensor must be contiguous. %s:%d\n", //
                     __FILE__, __LINE__);                            //
        throw std::invalid_argument("tensor must be contiguous");
    }

    if ((std::size_t)maskedX.numel() != elementNum)
    {
        std::fprintf(stderr,                                              //
                     "[FastFss]  maskedX.numel() != elementNum. %s:%d\n", //
                     __FILE__, __LINE__                                   //
        );                                                                //
        throw std::invalid_argument("maskedX.numel() != elementNum");
    }

    if ((std::size_t)seed.numel() != 16 * elementNum)
    {
        std::fprintf(stderr,                                               //
                     "[FastFss] seed.numel() != 16 * elementNum. %s:%d\n", //
                     __FILE__, __LINE__                                    //
        );                                                                 //
        throw std::invalid_argument("seed.numel() != 16 * elementNum");
    }

    if (key.dtype() != torch::kUInt8 || seed.dtype() != torch::kUInt8)
    {
        std::fprintf(
            stderr,                                                          //
            "[FastFss] key.dtype seed.dtype must be torch::kUInt8. %s:%d\n", //
            __FILE__, __LINE__                                               //
        );                                                                   //
        throw std::invalid_argument(                                         //
            "key.dtype seed.dtype must be torch::kUInt8"                     //
        );                                                                   //
    }

    if (maskedX.dtype() != sharedOut.dtype() ||
        maskedX.dtype() != sharedZ.dtype() ||
        maskedX.dtype() != leftBoundary.dtype() ||
        maskedX.dtype() != rightBoundary.dtype())
    {
        std::fprintf(stderr,                                             //
                     "[FastFss] maskedX sharedOut sharedZ leftBoundary " //
                     "rightBoundary dtype "                              //
                     "must be same. %s:%d\n",                            //
                     __FILE__, __LINE__                                  //
        );                                                               //
        throw std::invalid_argument("maskedX sharedOut sharedZ leftBoundary "
                                    "rightBoundary dtype must be same");
    }

    std::size_t elementSize = maskedX.element_size();
    if (bitWidthIn > elementSize * 8 || bitWidthOut > elementSize * 8)
    {
        std::fprintf(stderr,                                      //
                     "[FastFss] bitWidthIn <= elementSize *8 && " //
                     "bitWidthOut <= elementSize *8. %s:%d\n",    //
                     __FILE__, __LINE__                           //
        );                                                        //
        throw std::invalid_argument(                              //
            "bitWidthIn > elementSize * 8 || "                    //
            "bitWidthOut > elementSize *8"                        //
        );                                                        //
    }

    auto device = maskedX.device();
    if (sharedOut.device() != device || key.device() != device ||
        sharedZ.device() != device || seed.device() != device ||
        leftBoundary.device() != device || rightBoundary.device() != device)
    {
        std::fprintf(stderr,                                   //
                     "[FastFss] device must be same. %s:%d\n", //
                     __FILE__, __LINE__                        //
        );
        throw std::invalid_argument("device must be same");
    }

    if (leftBoundary.numel() != rightBoundary.numel())
    {
        std::fprintf(
            stderr,                                                    //
            "[FastFss] intervalNum != rightBoundary.numel(). %s:%d\n", //
            __FILE__, __LINE__                                         //
        );
        throw std::invalid_argument("intervalNum != rightBoundary.numel()");
    }

    std::size_t intervalNum = (std::size_t)leftBoundary.numel();

    if ((std::size_t)sharedZ.numel() != intervalNum * elementNum)
    {
        std::fprintf(
            stderr,                                                           //
            "[FastFss] sharedZ.numel() != intervalNum * elementNum. %s:%d\n", //
            __FILE__, __LINE__                                                //
        );
        throw std::invalid_argument(                      //
            "sharedZ.numel() != intervalNum * elementNum" //
        );                                                //
    }

    if ((std::size_t)key.numel() !=
        dcf_mic_get_key_data_size(bitWidthIn, bitWidthOut, elementSize,
                                  elementNum))
    {
        std::fprintf(
            stderr,                                                        //
            "[FastFss] key.numel() != dcf_mic_get_key_data_size. %s:%d\n", //
            __FILE__, __LINE__                                             //
        );
        throw std::invalid_argument(                   //
            "key.numel() != dcf_mic_get_key_data_size" //
        );
    }

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
}

} // namespace pyFastFss
