#include "pyFastGrotto.h"

#include <FastFss/cpu/grotto.h>
#include <FastFss/cuda/grotto.h>
#include <FastFss/grotto.h>
#ifndef NO_CUDA
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define ERR_LOG(fmt, ...) std::fprintf(stderr, "[FastFss Grotto] " fmt ". %s:%d\n", ##__VA_ARGS__, __FILE__, __LINE__)

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

namespace {

std::size_t grotto_get_cache_data_size(std::size_t bitWidthIn, std::size_t elementSize, std::size_t elementNum)
{
    std::size_t cacheSize = 0;
    int         ret       = FastFss_grottoGetCacheDataSize(&cacheSize, bitWidthIn, elementSize, elementNum);
    CHECK_ERROR_CODE(ret, "FastFss_grottoGetCacheDataSize");
    return cacheSize;
}

torch::Tensor make_u8_cache(const torch::Device &device, std::size_t size)
{
    torch::TensorOptions options;
    options = options.dtype(torch::kUInt8).device(device.type());
    return torch::empty({(std::int64_t)size}, options);
}

} // namespace

std::size_t grotto_get_key_data_size(std::size_t bitWidthIn, std::size_t elementSize, std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         result      = FastFss_grottoGetKeyDataSize(&keyDataSize, bitWidthIn, elementSize, elementNum);
    CHECK_ERROR_CODE(result, "FastFss_grottoGetKeyDataSize");
    return keyDataSize;
}

torch::Tensor &grotto_key_gen(torch::Tensor       &keyOut,
                              const torch::Tensor &alpha,
                              const torch::Tensor &seed0,
                              const torch::Tensor &seed1,
                              std::size_t          bitWidthIn,
                              std::size_t          elementNum)
{
    ARG_ASSERT(keyOut.is_contiguous());
    ARG_ASSERT(alpha.is_contiguous());
    ARG_ASSERT(seed0.is_contiguous());
    ARG_ASSERT(seed1.is_contiguous());
    const auto alphaLayout = inspect_value_tensor(alpha, bitWidthIn);
    ARG_ASSERT(alphaLayout.logicalElementNum == elementNum);
    ARG_ASSERT((std::size_t)seed0.numel() == 16 * elementNum);
    ARG_ASSERT((std::size_t)seed1.numel() == 16 * elementNum);
    ARG_ASSERT(keyOut.dtype() == torch::kUInt8);
    ARG_ASSERT(seed0.dtype() == torch::kUInt8);
    ARG_ASSERT(seed1.dtype() == torch::kUInt8);

    std::size_t elementSize = alphaLayout.elementSize;
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = alpha.device();
    ARG_ASSERT(keyOut.device() == device);
    ARG_ASSERT(seed0.device() == device);
    ARG_ASSERT(seed1.device() == device);

    keyOut.resize_({(std::int64_t)grotto_get_key_data_size(bitWidthIn, elementSize, elementNum)});

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoKeyGen(keyOut.mutable_data_ptr(),                         //
                                           (std::size_t)keyOut.numel(),                       //
                                           alpha.const_data_ptr(),                            //
                                           (std::size_t)alpha.numel() * alpha.element_size(), //
                                           seed0.const_data_ptr(),                            //
                                           (std::size_t)seed0.numel(),                        //
                                           seed1.const_data_ptr(),                            //
                                           (std::size_t)seed1.numel(),                        //
                                           bitWidthIn,                                        //
                                           elementSize,                                       //
                                           elementNum                                         //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoKeyGen");
    }
#ifndef NO_CUDA
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        int          ret    = FastFss_cuda_grottoKeyGen(keyOut.mutable_data_ptr(),                         //
                                                        (std::size_t)keyOut.numel(),                       //
                                                        alpha.const_data_ptr(),                            //
                                                        (std::size_t)alpha.numel() * alpha.element_size(), //
                                                        seed0.const_data_ptr(),                            //
                                                        (std::size_t)seed0.numel(),                        //
                                                        seed1.const_data_ptr(),                            //
                                                        (std::size_t)seed1.numel(),                        //
                                                        bitWidthIn,                                        //
                                                        elementSize,                                       //
                                                        elementNum,                                        //
                                                        &stream                                            //
                    );
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoKeyGen");
    }
#endif
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return keyOut;
}

torch::Tensor &grotto_eq_eval(torch::Tensor       &sharedOut,
                              const torch::Tensor &maskedX,
                              const torch::Tensor &key,
                              const torch::Tensor &seed,
                              int                  partyId,
                              std::size_t          bitWidthIn,
                              std::size_t          elementNum)
{
    ARG_ASSERT(sharedOut.is_contiguous());
    ARG_ASSERT(maskedX.is_contiguous());
    ARG_ASSERT(key.is_contiguous());
    ARG_ASSERT(seed.is_contiguous());
    const auto maskedLayout = inspect_value_tensor(maskedX, bitWidthIn);
    ARG_ASSERT(maskedLayout.logicalElementNum == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);
    ARG_ASSERT(sharedOut.dtype() == maskedX.dtype());

    std::size_t elementSize = maskedLayout.elementSize;
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT((std::size_t)key.numel() == grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    sharedOut.resize_(make_value_shape(maskedLayout.logicalShape, bitWidthIn));
    torch::Tensor cache = make_u8_cache(device, grotto_get_cache_data_size(bitWidthIn, elementSize, elementNum));

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoEqEval(sharedOut.mutable_data_ptr(),                              //
                                           (std::size_t)sharedOut.numel() * sharedOut.element_size(), //
                                           maskedX.const_data_ptr(),                                  //
                                           (std::size_t)maskedX.numel() * maskedX.element_size(),     //
                                           key.const_data_ptr(),                                      //
                                           (std::size_t)key.numel(),                                  //
                                           seed.const_data_ptr(),                                     //
                                           (std::size_t)seed.numel(),                                 //
                                           partyId,                                                   //
                                           bitWidthIn,                                                //
                                           elementSize,                                               //
                                           elementNum,                                                //
                                           cache.mutable_data_ptr(),                                  //
                                           (std::size_t)cache.numel()                                 //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoEqEval");
    }
#ifndef NO_CUDA
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        int          ret    = FastFss_cuda_grottoEqEval(sharedOut.mutable_data_ptr(),                              //
                                                        (std::size_t)sharedOut.numel() * sharedOut.element_size(), //
                                                        maskedX.const_data_ptr(),                                  //
                                                        (std::size_t)maskedX.numel() * maskedX.element_size(),     //
                                                        key.const_data_ptr(),                                      //
                                                        (std::size_t)key.numel(),                                  //
                                                        seed.const_data_ptr(),                                     //
                                                        (std::size_t)seed.numel(),                                 //
                                                        partyId,                                                   //
                                                        bitWidthIn,                                                //
                                                        elementSize,                                               //
                                                        elementNum,                                                //
                                                        cache.mutable_data_ptr(),                                  //
                                                        (std::size_t)cache.numel(),                                //
                                                        &stream                                                    //
                    );
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoEqEval");
    }
#endif
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedOut;
}

torch::Tensor &grotto_eval(torch::Tensor       &sharedOut,
                           const torch::Tensor &maskedX,
                           const torch::Tensor &key,
                           const torch::Tensor &seed,
                           bool                 equalBound,
                           int                  partyId,
                           std::size_t          bitWidthIn,
                           std::size_t          elementNum)
{
    ARG_ASSERT(sharedOut.is_contiguous());
    ARG_ASSERT(maskedX.is_contiguous());
    ARG_ASSERT(key.is_contiguous());
    ARG_ASSERT(seed.is_contiguous());
    const auto maskedLayout = inspect_value_tensor(maskedX, bitWidthIn);
    ARG_ASSERT(maskedLayout.logicalElementNum == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);
    ARG_ASSERT(sharedOut.dtype() == maskedX.dtype());

    std::size_t elementSize = maskedLayout.elementSize;
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT((std::size_t)key.numel() == grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    sharedOut.resize_(make_value_shape(maskedLayout.logicalShape, bitWidthIn));
    torch::Tensor cache = make_u8_cache(device, grotto_get_cache_data_size(bitWidthIn, elementSize, elementNum));

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoEval(sharedOut.mutable_data_ptr(),                              //
                                         (std::size_t)sharedOut.numel() * sharedOut.element_size(), //
                                         maskedX.const_data_ptr(),                                  //
                                         (std::size_t)maskedX.numel() * maskedX.element_size(),     //
                                         key.const_data_ptr(),                                      //
                                         (std::size_t)key.numel(),                                  //
                                         seed.const_data_ptr(),                                     //
                                         (std::size_t)seed.numel(),                                 //
                                         equalBound,                                                //
                                         partyId,                                                   //
                                         bitWidthIn,                                                //
                                         elementSize,                                               //
                                         elementNum,                                                //
                                         cache.mutable_data_ptr(),                                  //
                                         (std::size_t)cache.numel()                                 //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoEval");
    }
#ifndef NO_CUDA
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        int          ret    = FastFss_cuda_grottoEval(sharedOut.mutable_data_ptr(),                              //
                                                      (std::size_t)sharedOut.numel() * sharedOut.element_size(), //
                                                      maskedX.const_data_ptr(),                                  //
                                                      (std::size_t)maskedX.numel() * maskedX.element_size(),     //
                                                      key.const_data_ptr(),                                      //
                                                      (std::size_t)key.numel(),                                  //
                                                      seed.const_data_ptr(),                                     //
                                                      (std::size_t)seed.numel(),                                 //
                                                      equalBound,                                                //
                                                      partyId,                                                   //
                                                      bitWidthIn,                                                //
                                                      elementSize,                                               //
                                                      elementNum,                                                //
                                                      cache.mutable_data_ptr(),                                  //
                                                      (std::size_t)cache.numel(),                                //
                                                      &stream                                                    //
                    );
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoEval");
    }
#endif
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedOut;
}

torch::Tensor &grotto_mic_eval(torch::Tensor       &sharedBooleanOut,
                               const torch::Tensor &maskedX,
                               const torch::Tensor &key,
                               const torch::Tensor &seed,
                               int                  partyId,
                               const torch::Tensor &leftBoundary,
                               const torch::Tensor &rightBoundary,
                               std::size_t          bitWidthIn,
                               std::size_t          elementNum)
{
    ARG_ASSERT(sharedBooleanOut.is_contiguous());
    ARG_ASSERT(maskedX.is_contiguous());
    ARG_ASSERT(key.is_contiguous());
    ARG_ASSERT(seed.is_contiguous());
    ARG_ASSERT(leftBoundary.is_contiguous());
    ARG_ASSERT(rightBoundary.is_contiguous());
    const auto maskedLayout = inspect_value_tensor(maskedX, bitWidthIn);
    const auto leftLayout   = inspect_value_tensor(leftBoundary, bitWidthIn);
    const auto rightLayout  = inspect_value_tensor(rightBoundary, bitWidthIn);

    ARG_ASSERT(maskedLayout.logicalElementNum == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);
    ARG_ASSERT(sharedBooleanOut.dtype() == maskedX.dtype());
    ARG_ASSERT(sharedBooleanOut.dtype() == leftBoundary.dtype());
    ARG_ASSERT(sharedBooleanOut.dtype() == rightBoundary.dtype());

    std::size_t elementSize = maskedLayout.elementSize;
    ARG_ASSERT(bitWidthIn <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedBooleanOut.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT(leftBoundary.device() == device);
    ARG_ASSERT(rightBoundary.device() == device);
    assert_same_logical_shape(leftLayout, rightLayout,
                              "leftBoundary and rightBoundary must have the same logical shape");
    ARG_ASSERT((std::size_t)key.numel() == grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    sharedBooleanOut.resize_(make_value_shape(
        append_logical_dim(maskedLayout.logicalShape, static_cast<std::int64_t>(leftLayout.logicalElementNum)),
        bitWidthIn));
    torch::Tensor cache = make_u8_cache(device, grotto_get_cache_data_size(bitWidthIn, elementSize, elementNum));

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoMICEval(sharedBooleanOut.mutable_data_ptr(),                                     //
                                            (std::size_t)sharedBooleanOut.numel() * sharedBooleanOut.element_size(), //
                                            maskedX.const_data_ptr(),                                                //
                                            (std::size_t)maskedX.numel() * maskedX.element_size(),                   //
                                            key.const_data_ptr(),                                                    //
                                            (std::size_t)key.numel(),                                                //
                                            seed.const_data_ptr(),                                                   //
                                            (std::size_t)seed.numel(),                                               //
                                            partyId,                                                                 //
                                            leftBoundary.const_data_ptr(),                                           //
                                            (std::size_t)leftBoundary.numel() * leftBoundary.element_size(),         //
                                            rightBoundary.const_data_ptr(),                                          //
                                            (std::size_t)rightBoundary.numel() * rightBoundary.element_size(),       //
                                            bitWidthIn,                                                              //
                                            elementSize,                                                             //
                                            elementNum,                                                              //
                                            cache.mutable_data_ptr(),                                                //
                                            (std::size_t)cache.numel()                                               //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoMICEval");
    }
#ifndef NO_CUDA
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        int          ret    = FastFss_cuda_grottoMICEval(sharedBooleanOut.mutable_data_ptr(), //
                                                         (std::size_t)sharedBooleanOut.numel() * sharedBooleanOut.element_size(), //
                                                         maskedX.const_data_ptr(),                              //
                                                         (std::size_t)maskedX.numel() * maskedX.element_size(), //
                                                         key.const_data_ptr(),                                  //
                                                         (std::size_t)key.numel(),                              //
                                                         seed.const_data_ptr(),                                 //
                                                         (std::size_t)seed.numel(),                             //
                                                         partyId,                                               //
                                                         leftBoundary.const_data_ptr(),                         //
                                                         (std::size_t)leftBoundary.numel() * leftBoundary.element_size(), //
                                                         rightBoundary.const_data_ptr(), //
                                                         (std::size_t)rightBoundary.numel() * rightBoundary.element_size(), //
                                                         bitWidthIn,                 //
                                                         elementSize,                //
                                                         elementNum,                 //
                                                         cache.mutable_data_ptr(),   //
                                                         (std::size_t)cache.numel(), //
                                                         &stream                     //
                    );
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoMICEval");
    }
#endif
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return sharedBooleanOut;
}

py::tuple grotto_interval_lut_eval(torch::Tensor       &sharedOutE,
                                   torch::Tensor       &sharedOutT,
                                   const torch::Tensor &maskedX,
                                   const torch::Tensor &key,
                                   const torch::Tensor &seed,
                                   int                  partyId,
                                   const torch::Tensor &leftBoundary,
                                   const torch::Tensor &rightBoundary,
                                   const torch::Tensor &lookUpTable,
                                   std::size_t          bitWidthIn,
                                   std::size_t          bitWidthOut,
                                   std::size_t          elementNum)
{
    ARG_ASSERT(sharedOutE.is_contiguous());
    ARG_ASSERT(sharedOutT.is_contiguous());
    ARG_ASSERT(maskedX.is_contiguous());
    ARG_ASSERT(key.is_contiguous());
    ARG_ASSERT(seed.is_contiguous());
    ARG_ASSERT(leftBoundary.is_contiguous());
    ARG_ASSERT(rightBoundary.is_contiguous());
    ARG_ASSERT(lookUpTable.is_contiguous());
    const auto valueBitWidth = max_bit_width({bitWidthIn, bitWidthOut});
    const auto maskedLayout  = inspect_value_tensor(maskedX, valueBitWidth);
    const auto leftLayout    = inspect_value_tensor(leftBoundary, valueBitWidth);
    const auto rightLayout   = inspect_value_tensor(rightBoundary, valueBitWidth);
    const auto lutLayout     = inspect_value_tensor(lookUpTable, valueBitWidth);

    ARG_ASSERT(maskedLayout.logicalElementNum == elementNum);
    ARG_ASSERT((std::size_t)seed.numel() == 16 * elementNum);
    ARG_ASSERT(key.dtype() == torch::kUInt8);
    ARG_ASSERT(seed.dtype() == torch::kUInt8);

    auto dtype = maskedX.dtype();
    ARG_ASSERT(sharedOutE.dtype() == dtype);
    ARG_ASSERT(sharedOutT.dtype() == dtype);
    ARG_ASSERT(leftBoundary.dtype() == dtype);
    ARG_ASSERT(rightBoundary.dtype() == dtype);
    ARG_ASSERT(lookUpTable.dtype() == dtype);

    std::size_t elementSize = maskedLayout.elementSize;
    ARG_ASSERT(bitWidthIn <= elementSize * 8);
    ARG_ASSERT(bitWidthOut <= elementSize * 8);

    auto device = maskedX.device();
    ARG_ASSERT(sharedOutE.device() == device);
    ARG_ASSERT(sharedOutT.device() == device);
    ARG_ASSERT(key.device() == device);
    ARG_ASSERT(seed.device() == device);
    ARG_ASSERT(leftBoundary.device() == device);
    ARG_ASSERT(rightBoundary.device() == device);
    ARG_ASSERT(lookUpTable.device() == device);
    assert_same_logical_shape(leftLayout, rightLayout,
                              "leftBoundary and rightBoundary must have the same logical shape");
    ARG_ASSERT((std::size_t)key.numel() == grotto_get_key_data_size(bitWidthIn, elementSize, elementNum));

    ARG_ASSERT(lutLayout.logicalElementNum % leftLayout.logicalElementNum == 0);
    auto lutNum = lutLayout.logicalElementNum / leftLayout.logicalElementNum;

    sharedOutE.resize_(make_value_shape(maskedLayout.logicalShape, valueBitWidth));
    sharedOutT.resize_(make_value_shape(
        append_logical_dim(maskedLayout.logicalShape, static_cast<std::int64_t>(lutNum)), valueBitWidth));
    torch::Tensor cache = make_u8_cache(device, grotto_get_cache_data_size(bitWidthIn, elementSize, elementNum));

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_grottoIntervalLutEval(sharedOutE.mutable_data_ptr(),                                   //
                                                    (std::size_t)sharedOutE.numel() * sharedOutE.element_size(),     //
                                                    sharedOutT.mutable_data_ptr(),                                   //
                                                    (std::size_t)sharedOutT.numel() * sharedOutT.element_size(),     //
                                                    maskedX.const_data_ptr(),                                        //
                                                    (std::size_t)maskedX.numel() * maskedX.element_size(),           //
                                                    key.const_data_ptr(),                                            //
                                                    (std::size_t)key.numel(),                                        //
                                                    seed.const_data_ptr(),                                           //
                                                    (std::size_t)seed.numel(),                                       //
                                                    partyId,                                                         //
                                                    leftBoundary.const_data_ptr(),                                   //
                                                    (std::size_t)leftBoundary.numel() * leftBoundary.element_size(), //
                                                    rightBoundary.const_data_ptr(),                                  //
                                                    (std::size_t)rightBoundary.numel() * rightBoundary.element_size(),
                                                    lookUpTable.const_data_ptr(),                                  //
                                                    (std::size_t)lookUpTable.numel() * lookUpTable.element_size(), //
                                                    bitWidthIn,                                                    //
                                                    bitWidthOut,                                                   //
                                                    elementSize,                                                   //
                                                    elementNum,                                                    //
                                                    cache.mutable_data_ptr(),                                      //
                                                    (std::size_t)cache.numel()                                     //
        );
        CHECK_ERROR_CODE(ret, "FastFss_cpu_grottoIntervalLutEval");
    }
#ifndef NO_CUDA
    else if (device.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        int          ret    = FastFss_cuda_grottoIntervalLutEval(sharedOutE.mutable_data_ptr(), //
                                                                 (std::size_t)sharedOutE.numel() * sharedOutE.element_size(), //
                                                                 sharedOutT.mutable_data_ptr(), //
                                                                 (std::size_t)sharedOutT.numel() * sharedOutT.element_size(), //
                                                                 maskedX.const_data_ptr(), //
                                                                 (std::size_t)maskedX.numel() * maskedX.element_size(), //
                                                                 key.const_data_ptr(),          //
                                                                 (std::size_t)key.numel(),      //
                                                                 seed.const_data_ptr(),         //
                                                                 (std::size_t)seed.numel(),     //
                                                                 partyId,                       //
                                                                 leftBoundary.const_data_ptr(), //
                                                                 (std::size_t)leftBoundary.numel() * leftBoundary.element_size(), //
                                                                 rightBoundary.const_data_ptr(), //
                                                                 (std::size_t)rightBoundary.numel() * rightBoundary.element_size(),
                                                                 lookUpTable.const_data_ptr(), //
                                                                 (std::size_t)lookUpTable.numel() * lookUpTable.element_size(), //
                                                                 bitWidthIn,                 //
                                                                 bitWidthOut,                //
                                                                 elementSize,                //
                                                                 elementNum,                 //
                                                                 cache.mutable_data_ptr(),   //
                                                                 (std::size_t)cache.numel(), //
                                                                 &stream                     //
                    );
        CHECK_ERROR_CODE(ret, "FastFss_cuda_grottoIntervalLutEval");
    }
#endif
    else
    {
        throw std::invalid_argument("device must be CPU or CUDA");
    }
    return py::make_tuple(sharedOutE, sharedOutT);
}

} // namespace pyFastFss
