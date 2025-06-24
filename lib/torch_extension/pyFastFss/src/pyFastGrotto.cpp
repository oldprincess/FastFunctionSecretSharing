#include "pyFastGrotto.h"

#include <FastFss/cpu/grotto.h>
#include <FastFss/cuda/grotto.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define ERR_LOG(fmt, ...)                                                    \
    std::fprintf(stderr, "[FastFss Grotto] " fmt ". %s:%d\n", ##__VA_ARGS__, \
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

std::size_t grotto_get_key_data_size(std::size_t bitWidthIn,
                                     std::size_t elementSize,
                                     std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int result = FastFss_cpu_grottoGetKeyDataSize(&keyDataSize, bitWidthIn,
                                                  elementSize, elementNum);
    CHECK_ERROR_CODE(result, "FastFss_cpu_grottoGetKeyDataSize");
    return keyDataSize;
}

torch::Tensor& grotto_key_gen(torch::Tensor&       keyOut,
                              const torch::Tensor& alpha,
                              const torch::Tensor& seed0,
                              const torch::Tensor& seed1,
                              std::size_t          bitWidthIn,
                              std::size_t          elementNum)
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

    ARG_ASSERT(keyOut.device() == device);
    ARG_ASSERT(seed0.device() == device);
    ARG_ASSERT(seed1.device() == device);

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    std::size_t grottoKeyDataSize = grotto_get_key_data_size( //
        bitWidthIn, elementSize, elementNum                   //
    );                                                        //
    keyOut.resize_({(std::int64_t)grottoKeyDataSize});

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
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoKeyGen");
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
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoKeyGen");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return keyOut;
}

torch::Tensor& grotto_eq_eval(torch::Tensor&       sharedOut,
                              const torch::Tensor& maskedX,
                              const torch::Tensor& key,
                              const torch::Tensor& seed,
                              int                  partyId,
                              std::size_t          bitWidthIn,
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

    auto device = maskedX.device();

    ARG_ASSERT(sharedOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);

    ARG_ASSERT((std::size_t)key.numel() ==
               grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    sharedOut.resize_({maskedX.numel()});

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoEqEval(
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
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoEqEval");
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_grottoEqEval(
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
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoEqEval");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedOut;
}

torch::Tensor& grotto_eq_multi_eval(torch::Tensor&       sharedOut,
                                    const torch::Tensor& maskedX,
                                    const torch::Tensor& key,
                                    const torch::Tensor& seed,
                                    int                  partyId,
                                    const torch::Tensor& point,
                                    std::size_t          bitWidthIn,
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

    ARG_ASSERT(sharedOut.dtype() == maskedX.dtype());
    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);
    ARG_ASSERT(point.dtype() == maskedX.dtype());

    std::size_t elementSize = maskedX.element_size();

    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();

    ARG_ASSERT(sharedOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT(point.device() == device);

    ARG_ASSERT((std::size_t)key.numel() ==
               grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    std::size_t pointsNum = (std::size_t)point.numel();
    sharedOut.resize_({(std::int64_t)(pointsNum * elementNum)});

    std::size_t cacheSize;
    {
        int ret = FastFss_cpu_grottoGetCacheDataSize(&cacheSize, bitWidthIn,
                                                     elementSize, elementNum);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoGetCacheDataSize");
    }
    torch::TensorOptions options;
    options             = options.dtype(torch::kUInt8).device(device.type());
    torch::Tensor cache = torch::empty({(std::int64_t)(cacheSize)}, options);

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoEqMultiEval(          //
            sharedOut.mutable_data_ptr(),                 //
            (std::size_t)sharedOut.numel() * elementSize, //
            maskedX.const_data_ptr(),                     //
            (std::size_t)maskedX.numel() * elementSize,   //
            key.const_data_ptr(),                         //
            (std::size_t)key.numel(),                     //
            seed.const_data_ptr(),                        //
            (std::size_t)seed.numel(),                    //
            partyId,                                      //
            point.const_data_ptr(),                       //
            (std::size_t)point.numel() * elementSize,     //
            bitWidthIn,                                   //
            elementSize,                                  //
            elementNum,                                   //
            cache.mutable_data_ptr(), cacheSize);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoEqMultiEval");
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_grottoEqMultiEval(         //
            sharedOut.mutable_data_ptr(),                 //
            (std::size_t)sharedOut.numel() * elementSize, //
            maskedX.const_data_ptr(),                     //
            (std::size_t)maskedX.numel() * elementSize,   //
            key.const_data_ptr(),                         //
            (std::size_t)key.numel(),                     //
            seed.const_data_ptr(),                        //
            (std::size_t)seed.numel(),                    //
            partyId,                                      //
            point.const_data_ptr(),                       //
            (std::size_t)point.numel() * elementSize,     //
            bitWidthIn,                                   //
            elementSize,                                  //
            elementNum,                                   //
            cache.mutable_data_ptr(), cacheSize, &stream);
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoEqMultiEval");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedOut;
}

torch::Tensor& grotto_eval(torch::Tensor&       sharedOut,
                           const torch::Tensor& maskedX,
                           const torch::Tensor& key,
                           const torch::Tensor& seed,
                           bool                 equalBound,
                           int                  partyId,
                           std::size_t          bitWidthIn,
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

    auto device = maskedX.device();
    ARG_ASSERT(sharedOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT((std::size_t)key.numel() ==
               grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    sharedOut.resize_({maskedX.numel()});

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
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoEqEval");
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_grottoEval(
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
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoEval");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedOut;
}

torch::Tensor& grotto_mic_eval(torch::Tensor&       sharedBooleanOut,
                               const torch::Tensor& maskedX,
                               const torch::Tensor& key,
                               const torch::Tensor& seed,
                               int                  partyId,
                               const torch::Tensor& leftBoundary,
                               const torch::Tensor& rightBoundary,
                               std::size_t          bitWidthIn,
                               std::size_t          elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================
    ARG_ASSERT(sharedBooleanOut.is_contiguous());
    ARG_ASSERT(maskedX.is_contiguous());
    ARG_ASSERT(key.is_contiguous());
    ARG_ASSERT(seed.is_contiguous());
    ARG_ASSERT(leftBoundary.is_contiguous());
    ARG_ASSERT(rightBoundary.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);
    ARG_ASSERT(sharedBooleanOut.dtype() == maskedX.dtype());
    ARG_ASSERT(sharedBooleanOut.dtype() == leftBoundary.dtype());
    ARG_ASSERT(sharedBooleanOut.dtype() == rightBoundary.dtype());

    std::size_t elementSize = maskedX.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedBooleanOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT(leftBoundary.device() == device);
    ARG_ASSERT(rightBoundary.device() == device);
    ARG_ASSERT(leftBoundary.numel() == rightBoundary.numel());

    std::size_t intervalNum = (std::size_t)leftBoundary.numel();
    ARG_ASSERT((std::size_t)key.numel() ==
               grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    sharedBooleanOut.resize_({(std::int64_t)(intervalNum * elementNum)});

    std::size_t cacheSize;
    {
        int ret = FastFss_cpu_grottoGetCacheDataSize(&cacheSize, bitWidthIn,
                                                     elementSize, elementNum);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoGetCacheDataSize");
    }
    torch::TensorOptions options;
    options             = options.dtype(torch::kUInt8).device(device.type());
    torch::Tensor cache = torch::empty({(std::int64_t)(cacheSize)}, options);

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
            cache.mutable_data_ptr(), cacheSize);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoMICEval");
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
            cache.mutable_data_ptr(), cacheSize,                 //
            &stream                                              //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoMICEval");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedBooleanOut;
}

py::tuple grotto_lut_eval(torch::Tensor&       sharedOutE,
                          torch::Tensor&       sharedOutT,
                          const torch::Tensor& maskedX,
                          const torch::Tensor& key,
                          const torch::Tensor& seed,
                          int                  partyId,
                          const torch::Tensor& lookUpTable,
                          std::size_t          bitWidthIn,
                          std::size_t          bitWidthOut,
                          std::size_t          elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================
    ARG_ASSERT(sharedOutE.is_contiguous());
    ARG_ASSERT(sharedOutT.is_contiguous());
    ARG_ASSERT(maskedX.is_contiguous());
    ARG_ASSERT(key.is_contiguous());
    ARG_ASSERT(seed.is_contiguous());
    ARG_ASSERT(lookUpTable.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);

    auto dtype = maskedX.dtype();
    ARG_ASSERT(sharedOutE.dtype() == dtype);
    ARG_ASSERT(sharedOutT.dtype() == dtype);
    ARG_ASSERT(lookUpTable.dtype() == dtype);

    std::size_t elementSize = maskedX.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOutT.device() == device);
    ARG_ASSERT(sharedOutE.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT(lookUpTable.device() == device);

    ARG_ASSERT((std::size_t)key.numel() ==
               grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    auto lutNum = lookUpTable.numel() / (1LL << bitWidthIn);
    ARG_ASSERT(lookUpTable.numel() == lutNum * (1LL << bitWidthIn));

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    sharedOutE.resize_({(std::int64_t)(elementNum)});
    sharedOutT.resize_({(std::int64_t)(elementNum)*lutNum});

    std::size_t cacheSize;
    {
        int ret = FastFss_cpu_grottoGetCacheDataSize(&cacheSize, bitWidthIn,
                                                     elementSize, elementNum);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoGetCacheDataSize");
    }
    torch::TensorOptions options;
    options             = options.dtype(torch::kUInt8).device(device.type());
    torch::Tensor cache = torch::empty({(std::int64_t)(cacheSize)}, options);

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoLutEval(                //
            sharedOutE.mutable_data_ptr(),                  //
            sharedOutT.mutable_data_ptr(),                  //
            maskedX.const_data_ptr(),                       //
            (std::size_t)maskedX.numel() * elementSize,     //
            key.const_data_ptr(),                           //
            (std::size_t)key.numel(),                       //
            seed.const_data_ptr(),                          //
            (std::size_t)seed.numel(),                      //
            partyId,                                        //
            lookUpTable.const_data_ptr(),                   //
            (std::size_t)lookUpTable.numel() * elementSize, //
            bitWidthIn,                                     //
            bitWidthOut,                                    //
            elementSize,                                    //
            elementNum, cache.mutable_data_ptr(), cacheSize);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoLutEval");
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_grottoLutEval(               //
            sharedOutE.mutable_data_ptr(),                  //
            sharedOutT.mutable_data_ptr(),                  //
            maskedX.const_data_ptr(),                       //
            (std::size_t)maskedX.numel() * elementSize,     //
            key.const_data_ptr(),                           //
            (std::size_t)key.numel(),                       //
            seed.const_data_ptr(),                          //
            (std::size_t)seed.numel(),                      //
            partyId,                                        //
            lookUpTable.const_data_ptr(),                   //
            (std::size_t)lookUpTable.numel() * elementSize, //
            bitWidthIn,                                     //
            bitWidthOut,                                    //
            elementSize,                                    //
            elementNum, cache.mutable_data_ptr(), cacheSize, &stream);
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoLutEval");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return py::make_tuple(sharedOutE, sharedOutT);
}

py::tuple grotto_interval_lut_eval(torch::Tensor&       sharedOutE,
                                   torch::Tensor&       sharedOutT,
                                   const torch::Tensor& maskedX,
                                   const torch::Tensor& key,
                                   const torch::Tensor& seed,
                                   int                  partyId,
                                   const torch::Tensor& leftBoundary,
                                   const torch::Tensor& rightBoundary,
                                   const torch::Tensor& lookUpTable,
                                   std::size_t          bitWidthIn,
                                   std::size_t          bitWidthOut,
                                   std::size_t          elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================
    ARG_ASSERT(sharedOutE.is_contiguous());
    ARG_ASSERT(sharedOutT.is_contiguous());
    ARG_ASSERT(maskedX.is_contiguous());
    ARG_ASSERT(key.is_contiguous());
    ARG_ASSERT(seed.is_contiguous());
    ARG_ASSERT(leftBoundary.is_contiguous());
    ARG_ASSERT(rightBoundary.is_contiguous());
    ARG_ASSERT(lookUpTable.is_contiguous());

    ARG_ASSERT((std::size_t)maskedX.numel() == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);

    auto dtype = maskedX.dtype();
    ARG_ASSERT(sharedOutE.dtype() == dtype);
    ARG_ASSERT(sharedOutT.dtype() == dtype);
    ARG_ASSERT(leftBoundary.dtype() == dtype);
    ARG_ASSERT(rightBoundary.dtype() == dtype);
    ARG_ASSERT(lookUpTable.dtype() == dtype);

    std::size_t elementSize = maskedX.element_size();
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOutT.device() == device);
    ARG_ASSERT(sharedOutE.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT(leftBoundary.device() == device);
    ARG_ASSERT(rightBoundary.device() == device);
    ARG_ASSERT(lookUpTable.device() == device);

    ARG_ASSERT(leftBoundary.numel() == rightBoundary.numel());

    ARG_ASSERT((std::size_t)key.numel() ==
               grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    auto lutNum = lookUpTable.numel() / leftBoundary.numel();
    ARG_ASSERT(lookUpTable.numel() == lutNum * leftBoundary.numel());

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    sharedOutE.resize_({(std::int64_t)(elementNum)});
    sharedOutT.resize_({(std::int64_t)(elementNum)*lutNum});

    std::size_t cacheSize;
    {
        int ret = FastFss_cpu_grottoGetCacheDataSize(&cacheSize, bitWidthIn,
                                                     elementSize, elementNum);
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoGetCacheDataSize");
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
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoIntervalLutEval");
    }
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

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
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoIntervalLutEval");
    }
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return py::make_tuple(sharedOutE, sharedOutT);
}

} // namespace pyFastFss
