#include <FastFss/cpu/grotto.h>
#include <FastFss/cuda/grotto.h>
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

std::size_t grotto_get_key_data_size(std::size_t bitWidthIn,
                                     std::size_t elementSize,
                                     std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int result = FastFss_cpu_grottoGetKeyDataSize(&keyDataSize, bitWidthIn,
                                                  elementSize, elementNum);
    if (result < 0)
    {
        ERR_LOG("FastFss_cpu_grottoGetKeyDataSize ret = %d", result);
        throw std::runtime_error("FastFss_cpu_grottoGetKeyDataSize fail");
    }
    return keyDataSize;
}

std::size_t grotto_get_zipped_key_data_size(std::size_t bitWidthIn,
                                            std::size_t elementSize,
                                            std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         result      = FastFss_cpu_grottoGetZippedKeyDataSize(
        &keyDataSize, bitWidthIn, elementSize, elementNum);
    if (result < 0)
    {
        ERR_LOG("FastFss_cpu_grottoGetZippedKeyDataSize ret = %d", result);
        throw std::runtime_error("FastFss_cpu_grottoGetZippedKeyDataSize fail");
    }
    return keyDataSize;
}

void grotto_key_zip(torch::Tensor zippedKeyOut,
                    torch::Tensor key,
                    std::size_t   bitWidthIn,
                    std::size_t   elementNum)
{
    auto device = key.device();

    ARG_ASSERT(zippedKeyOut.device() == device);
    ARG_ASSERT(zippedKeyOut.dtype() == torch::kUInt8);
    ARG_ASSERT(zippedKeyOut.dtype() == torch::kUInt8);
    ARG_ASSERT(key.dtype() == torch::kUInt8);

    std::size_t zippedKeyDataSize =
        grotto_get_zipped_key_data_size(bitWidthIn, 1, elementNum);
    if ((std::size_t)zippedKeyOut.numel() != zippedKeyDataSize)
    {
        zippedKeyOut.resize_({(std::int64_t)zippedKeyDataSize});
    }

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoKeyZip(    //
            zippedKeyOut.mutable_data_ptr(),   //
            (std::size_t)zippedKeyOut.numel(), //
            key.const_data_ptr(),              //
            (std::size_t)key.numel(),          //
            bitWidthIn,                        //
            1,                                 //
            elementNum                         //
        );
        if (ret < 0)
        {
            ERR_LOG("FastFss_cpu_grottoKeyZip ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_grottoKeyZip fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_grottoKeyZip(   //
            zippedKeyOut.mutable_data_ptr(),   //
            (std::size_t)zippedKeyOut.numel(), //
            key.const_data_ptr(),              //
            (std::size_t)key.numel(),          //
            bitWidthIn,                        //
            1,                                 //
            elementNum,                        //
            &stream                            //
        );
        if (ret < 0)
        {
            ERR_LOG("FastFss_cuda_grottoKeyZip ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_grottoKeyZip fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

void grotto_key_unzip(torch::Tensor keyOut,
                      torch::Tensor zippedKey,
                      std::size_t   bitWidthIn,
                      std::size_t   elementNum)
{
    auto device = zippedKey.device();

    ARG_ASSERT(keyOut.device() == device);
    ARG_ASSERT(keyOut.dtype() == torch::kUInt8);
    ARG_ASSERT(zippedKey.dtype() == torch::kUInt8);

    std::size_t keyDataSize =
        grotto_get_key_data_size(bitWidthIn, 1, elementNum);
    if ((std::size_t)keyOut.numel() != keyDataSize)
    {
        keyOut.resize_({(std::int64_t)keyDataSize});
    }

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoKeyUnzip( //
            keyOut.mutable_data_ptr(),        //
            (std::size_t)keyOut.numel(),      //
            zippedKey.const_data_ptr(),       //
            (std::size_t)zippedKey.numel(),   //
            bitWidthIn,                       //
            1,                                //
            elementNum                        //
        );
        if (ret < 0)
        {
            ERR_LOG("FastFss_cpu_grottoKeyUnzip ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_grottoKeyUnzip fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_grottoKeyUnzip( //
            keyOut.mutable_data_ptr(),         //
            (std::size_t)keyOut.numel(),       //
            zippedKey.const_data_ptr(),        //
            (std::size_t)zippedKey.numel(),    //
            bitWidthIn,                        //
            1,                                 //
            elementNum,                        //
            &stream                            //
        );
        if (ret < 0)
        {
            ERR_LOG("FastFss_cuda_grottoKeyUnzip ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_grottoKeyUnzip fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

void grotto_key_gen(torch::Tensor keyOut,
                    torch::Tensor alpha,
                    torch::Tensor seed0,
                    torch::Tensor seed1,
                    std::size_t   bitWidthIn,
                    std::size_t   elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    ARG_ASSERT(keyOut.is_contiguous());
    ARG_ASSERT(alpha.is_contiguous());
    ARG_ASSERT(seed0.is_contiguous());
    ARG_ASSERT(seed1.is_contiguous());

    ARG_ASSERT((std::size_t)alpha.numel() == elementNum);
    ARG_ASSERT((std::size_t)seed0.numel() == 16 * elementNum);
    ARG_ASSERT((std::size_t)seed1.numel() == 16 * elementNum);

    ARG_ASSERT(keyOut.dtype() == torch::kUInt8);
    ARG_ASSERT(seed0.dtype() == torch::kUInt8);
    ARG_ASSERT(seed1.dtype() == torch::kUInt8);

    std::size_t elementSize = alpha.element_size();

    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = alpha.device();

    ARG_ASSERT(keyOut.device() == device && seed0.device() == device &&
               seed1.device() == device);

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    std::size_t grottoKeyDataSize = grotto_get_key_data_size( //
        bitWidthIn, elementSize, elementNum                   //
    );                                                        //

    if ((std::size_t)keyOut.numel() != grottoKeyDataSize)
    {
        keyOut.resize_({(std::int64_t)grottoKeyDataSize});
    }

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoKeyGen(
            keyOut.mutable_data_ptr(),                //
            (std::size_t)keyOut.numel(),              //
            alpha.const_data_ptr(),                   //
            (std::size_t)alpha.numel() * elementSize, //
            seed0.const_data_ptr(),                   //
            (std::size_t)seed0.numel(),               //
            seed1.const_data_ptr(),                   //
            (std::size_t)seed1.numel(),               //
            bitWidthIn,                               //
            elementSize,                              //
            elementNum                                //
        );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_grottoKeyGen ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_grottoKeyGen fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_grottoKeyGen(
            keyOut.mutable_data_ptr(),                //
            (std::size_t)keyOut.numel(),              //
            alpha.const_data_ptr(),                   //
            (std::size_t)alpha.numel() * elementSize, //
            seed0.const_data_ptr(),                   //
            (std::size_t)seed0.numel(),               //
            seed1.const_data_ptr(),                   //
            (std::size_t)seed1.numel(),               //
            bitWidthIn,                               //
            elementSize,                              //
            elementNum,                               //
            &stream                                   //
        );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_grottoKeyGen ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_grottoKeyGen fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}
void grotto_eval_eq(torch::Tensor sharedOut,
                    torch::Tensor maskedX,
                    torch::Tensor key,
                    torch::Tensor seed,
                    int           partyId,
                    std::size_t   bitWidthIn,
                    std::size_t   elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    ARG_ASSERT(sharedOut.is_contiguous() && //
               maskedX.is_contiguous() &&   //
               key.is_contiguous() &&       //
               seed.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum &&
               (std::size_t)seed.numel() == 16 * elementNum);

    ARG_ASSERT(key.dtype() == torch::kUInt8 && seed.dtype() == torch::kUInt8);
    ARG_ASSERT(sharedOut.dtype() == maskedX.dtype());

    std::size_t elementSize = maskedX.element_size();

    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();

    ARG_ASSERT(sharedOut.device() == device && key.device() == device &&
               seed.device() == device);

    ARG_ASSERT((std::size_t)key.numel() ==
               grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    if (sharedOut.numel() != maskedX.numel())
    {
        sharedOut.resize_({maskedX.numel()});
    }

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoEvalEq(
            sharedOut.mutable_data_ptr(),               //
            maskedX.const_data_ptr(),                   //
            (std::size_t)maskedX.numel() * elementSize, //
            key.const_data_ptr(),                       //
            (std::size_t)key.numel(),                   //
            seed.const_data_ptr(),                      //
            (std::size_t)seed.numel(),                  //
            partyId,                                    //
            bitWidthIn,                                 //
            elementSize,                                //
            elementNum,                                 //
            nullptr,                                    //
            0                                           //
        );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_grottoEval fail ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_grottoEvalEq fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_grottoEvalEq(
            sharedOut.mutable_data_ptr(),               //
            maskedX.const_data_ptr(),                   //
            (std::size_t)maskedX.numel() * elementSize, //
            key.const_data_ptr(),                       //
            (std::size_t)key.numel(),                   //
            seed.const_data_ptr(),                      //
            (std::size_t)seed.numel(),                  //
            partyId,                                    //
            bitWidthIn,                                 //
            elementSize,                                //
            elementNum,                                 //
            nullptr,                                    //
            0,                                          //
            &stream                                     //
        );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_grottoEval fail ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_grottoEvalEq fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

void grotto_eval(torch::Tensor sharedOut,
                 torch::Tensor maskedX,
                 torch::Tensor key,
                 torch::Tensor seed,
                 bool          equalBound,
                 int           partyId,
                 std::size_t   bitWidthIn,
                 std::size_t   elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    ARG_ASSERT(sharedOut.is_contiguous() && //
               maskedX.is_contiguous() &&   //
               key.is_contiguous() &&       //
               seed.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum &&
               (std::size_t)seed.numel() == 16 * elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8 && seed.dtype() == torch::kUInt8);
    ARG_ASSERT(sharedOut.dtype() == maskedX.dtype());

    std::size_t elementSize = maskedX.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOut.device() == device && key.device() == device &&
               seed.device() == device);
    ARG_ASSERT((std::size_t)key.numel() ==
               grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    if (sharedOut.numel() != maskedX.numel())
    {
        sharedOut.resize_({maskedX.numel()});
    }

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoEval(
            sharedOut.mutable_data_ptr(),               //
            maskedX.const_data_ptr(),                   //
            (std::size_t)maskedX.numel() * elementSize, //
            key.const_data_ptr(),                       //
            (std::size_t)key.numel(),                   //
            seed.const_data_ptr(),                      //
            (std::size_t)seed.numel(),                  //
            equalBound,                                 //
            partyId,                                    //
            bitWidthIn,                                 //
            elementSize,                                //
            elementNum,                                 //
            nullptr,                                    //
            0);
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_grottoEval fail ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_grottoEval fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        int          ret    = FastFss_cuda_grottoEval(
            sharedOut.mutable_data_ptr(),               //
            maskedX.const_data_ptr(),                   //
            (std::size_t)maskedX.numel() * elementSize, //
            key.const_data_ptr(),                       //
            (std::size_t)key.numel(),                   //
            seed.const_data_ptr(),                      //
            (std::size_t)seed.numel(),                  //
            equalBound,                                 //
            partyId,                                    //
            bitWidthIn,                                 //
            elementSize,                                //
            elementNum,                                 //
            nullptr,                                    //
            0,                                          //
            &stream                                     //
        );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_grottoEval fail ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_grottoEval fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

void grotto_mic_eval(torch::Tensor sharedBooleanOut,
                     torch::Tensor maskedX,
                     torch::Tensor key,
                     torch::Tensor seed,
                     int           partyId,
                     torch::Tensor leftBoundary,
                     torch::Tensor rightBoundary,
                     std::size_t   bitWidthIn,
                     std::size_t   elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================
    ARG_ASSERT(sharedBooleanOut.is_contiguous() && //
               maskedX.is_contiguous() &&          //
               key.is_contiguous() &&              //
               seed.is_contiguous() &&             //
               leftBoundary.is_contiguous() &&     //
               rightBoundary.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum &&
               (std::size_t)seed.numel() == 16 * elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8 && seed.dtype() == torch::kUInt8);
    ARG_ASSERT(sharedBooleanOut.dtype() == maskedX.dtype() &&
               sharedBooleanOut.dtype() == leftBoundary.dtype() &&
               sharedBooleanOut.dtype() == rightBoundary.dtype());

    std::size_t elementSize = maskedX.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedBooleanOut.device() == device && key.device() == device &&
               seed.device() == device && leftBoundary.device() == device &&
               rightBoundary.device() == device);
    ARG_ASSERT(leftBoundary.numel() == rightBoundary.numel());

    std::size_t intervalNum = (std::size_t)leftBoundary.numel();
    ARG_ASSERT((std::size_t)key.numel() ==
               grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    if ((std::size_t)sharedBooleanOut.numel() != intervalNum * elementNum)
    {
        sharedBooleanOut.resize_({(std::int64_t)(intervalNum * elementNum)});
    }

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoMICEval(                     //
            sharedBooleanOut.mutable_data_ptr(),                 //
            (std::size_t)sharedBooleanOut.numel() * elementSize, //
            maskedX.const_data_ptr(),                            //
            (std::size_t)maskedX.numel() * elementSize,          //
            key.const_data_ptr(),                                //
            (std::size_t)key.numel(),                            //
            seed.const_data_ptr(),                               //
            (std::size_t)seed.numel(),                           //
            partyId,                                             //
            leftBoundary.const_data_ptr(),                       //
            (std::size_t)leftBoundary.numel() * elementSize,     //
            rightBoundary.const_data_ptr(),                      //
            (std::size_t)rightBoundary.numel() * elementSize,    //
            bitWidthIn,                                          //
            elementSize,                                         //
            elementNum,                                          //
            nullptr,                                             //
            0);
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_grottoMICEval ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_grottoMICEval fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_grottoMICEval(                    //
            sharedBooleanOut.mutable_data_ptr(),                 //
            (std::size_t)sharedBooleanOut.numel() * elementSize, //
            maskedX.const_data_ptr(),                            //
            (std::size_t)maskedX.numel() * elementSize,          //
            key.const_data_ptr(),                                //
            (std::size_t)key.numel(),                            //
            seed.const_data_ptr(),                               //
            (std::size_t)seed.numel(),                           //
            partyId,                                             //
            leftBoundary.const_data_ptr(),                       //
            (std::size_t)leftBoundary.numel() * elementSize,     //
            rightBoundary.const_data_ptr(),                      //
            (std::size_t)rightBoundary.numel() * elementSize,    //
            bitWidthIn,                                          //
            elementSize,                                         //
            elementNum,                                          //
            nullptr,                                             //
            0,                                                   //
            &stream                                              //
        );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_grottoMICEval ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_grottoMICEval fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

void grotto_interval_lut_eval(torch::Tensor sharedOutE,
                              torch::Tensor sharedOutT,
                              torch::Tensor maskedX,
                              torch::Tensor key,
                              torch::Tensor seed,
                              int           partyId,
                              torch::Tensor leftBoundary,
                              torch::Tensor rightBoundary,
                              torch::Tensor lookUpTable,
                              std::size_t   bitWidthIn,
                              std::size_t   bitWidthOut,
                              std::size_t   elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================
    ARG_ASSERT(sharedOutE.is_contiguous() &&    //
               sharedOutT.is_contiguous() &&    //
               maskedX.is_contiguous() &&       //
               key.is_contiguous() &&           //
               seed.is_contiguous() &&          //
               leftBoundary.is_contiguous() &&  //
               rightBoundary.is_contiguous() && //
               lookUpTable.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum &&
               (std::size_t)seed.numel() == 16 * elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8 && seed.dtype() == torch::kUInt8);

    auto dtype = maskedX.dtype();
    ARG_ASSERT(sharedOutE.dtype() == dtype && sharedOutT.dtype() == dtype &&
               leftBoundary.dtype() == dtype &&
               rightBoundary.dtype() == dtype && lookUpTable.dtype() == dtype);

    std::size_t elementSize = maskedX.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOutT.device() == device && sharedOutE.device() == device &&
               key.device() == device && seed.device() == device &&
               leftBoundary.device() == device &&
               rightBoundary.device() == device &&
               lookUpTable.device() == device);

    ARG_ASSERT(leftBoundary.numel() == rightBoundary.numel() &&
               leftBoundary.numel() == lookUpTable.numel());

    ARG_ASSERT((std::size_t)key.numel() ==
               grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

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

    std::size_t cacheSize;
    {
        int ret = FastFss_cpu_grottoGetCacheDataSize(&cacheSize, bitWidthIn,
                                                     elementSize, elementNum);
        if (ret)
        {
            ERR_LOG("FastFss_cpu_grottoGetCacheDataSize ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_grottoGetCacheDataSize fail");
        }
    }
    torch::TensorOptions options;
    options             = options.dtype(torch::kUInt8).device(device.type());
    torch::Tensor cache = torch::empty({(std::int64_t)(cacheSize)}, options);

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoIntervalLutEval(          //
            sharedOutE.mutable_data_ptr(),                    //
            sharedOutT.mutable_data_ptr(),                    //
            maskedX.const_data_ptr(),                         //
            (std::size_t)maskedX.numel() * elementSize,       //
            key.const_data_ptr(),                             //
            (std::size_t)key.numel(),                         //
            seed.const_data_ptr(),                            //
            (std::size_t)seed.numel(),                        //
            partyId,                                          //
            leftBoundary.const_data_ptr(),                    //
            (std::size_t)leftBoundary.numel() * elementSize,  //
            rightBoundary.const_data_ptr(),                   //
            (std::size_t)rightBoundary.numel() * elementSize, //
            lookUpTable.const_data_ptr(),                     //
            (std::size_t)lookUpTable.numel() * elementSize,   //
            bitWidthIn,                                       //
            bitWidthOut,                                      //
            elementSize,                                      //
            elementNum, cache.mutable_data_ptr(), cacheSize);
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_grottoIntervalLutEval fail ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_grottoIntervalLutEval fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        ERR_LOG("stream ptr = %p", stream);

        int ret = FastFss_cuda_grottoIntervalLutEval(         //
            sharedOutE.mutable_data_ptr(),                    //
            sharedOutT.mutable_data_ptr(),                    //
            maskedX.const_data_ptr(),                         //
            (std::size_t)maskedX.numel() * elementSize,       //
            key.const_data_ptr(),                             //
            (std::size_t)key.numel(),                         //
            seed.const_data_ptr(),                            //
            (std::size_t)seed.numel(),                        //
            partyId,                                          //
            leftBoundary.const_data_ptr(),                    //
            (std::size_t)leftBoundary.numel() * elementSize,  //
            rightBoundary.const_data_ptr(),                   //
            (std::size_t)rightBoundary.numel() * elementSize, //
            lookUpTable.const_data_ptr(),                     //
            (std::size_t)lookUpTable.numel() * elementSize,   //
            bitWidthIn,                                       //
            bitWidthOut,                                      //
            elementSize,                                      //
            elementNum, cache.mutable_data_ptr(), cacheSize, &stream);
        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_grottoIntervalLutEval ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_grottoIntervalLutEval fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

} // namespace pyFastFss
