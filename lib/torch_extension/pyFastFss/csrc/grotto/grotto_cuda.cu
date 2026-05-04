#include <FastFss/cuda/grotto.h>
#include <FastFss/grotto.h>
#include <c10/cuda/CUDAStream.h>

#include "grotto_cuda.h"

torch::Tensor &cuda_grotto_key_gen(torch::Tensor       &key,
                                   const torch::Tensor &alpha,
                                   const torch::Tensor &seed0,
                                   const torch::Tensor &seed1,
                                   std::size_t          bitWidthIn,
                                   std::size_t          elementSize,
                                   std::size_t          elementNum)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int ret = FastFss_cuda_grottoKeyGen(      //
        key.mutable_data_ptr(),               //
        key.element_size() * key.numel(),     //
        alpha.const_data_ptr(),               //
        alpha.element_size() * alpha.numel(), //
        seed0.const_data_ptr(),               //
        seed0.element_size() * seed0.numel(), //
        seed1.const_data_ptr(),               //
        seed1.element_size() * seed1.numel(), //
        bitWidthIn,                           //
        elementSize,                          //
        elementNum,                           //
        &stream                               //
    );                                        //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_grottoKeyGen failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return key;
}

torch::Tensor &cuda_grotto_eq_eval(torch::Tensor       &out,
                                   const torch::Tensor &maskedX,
                                   const torch::Tensor &key,
                                   const torch::Tensor &seed,
                                   int                  partyId,
                                   std::size_t          bitWidthIn,
                                   std::size_t          elementSize,
                                   std::size_t          elementNum)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int ret = FastFss_cuda_grottoEqEval(          //
        out.mutable_data_ptr(),                   //
        out.element_size() * out.numel(),         //
        maskedX.const_data_ptr(),                 //
        maskedX.element_size() * maskedX.numel(), //
        key.const_data_ptr(),                     //
        key.element_size() * key.numel(),         //
        seed.const_data_ptr(),                    //
        seed.element_size() * seed.numel(),       //
        partyId,                                  //
        bitWidthIn,                               //
        elementSize,                              //
        elementNum,                               //
        nullptr,                                  //
        0,                                        //
        &stream                                   //
    );                                            //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_grottoEqEval failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return out;
}

torch::Tensor &cuda_grotto_eval(torch::Tensor       &out,
                                const torch::Tensor &maskedX,
                                const torch::Tensor &key,
                                const torch::Tensor &seed,
                                bool                 equalBound,
                                int                  partyId,
                                std::size_t          bitWidthIn,
                                std::size_t          elementSize,
                                std::size_t          elementNum)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int ret = FastFss_cuda_grottoEval(            //
        out.mutable_data_ptr(),                   //
        out.element_size() * out.numel(),         //
        maskedX.const_data_ptr(),                 //
        maskedX.element_size() * maskedX.numel(), //
        key.const_data_ptr(),                     //
        key.element_size() * key.numel(),         //
        seed.const_data_ptr(),                    //
        seed.element_size() * seed.numel(),       //
        equalBound,                               //
        partyId,                                  //
        bitWidthIn,                               //
        elementSize,                              //
        elementNum,                               //
        nullptr,                                  //
        0,                                        //
        &stream                                   //
    );                                            //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_grottoEval failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return out;
}

torch::Tensor &cuda_grotto_mic_eval(torch::Tensor       &out,
                                    const torch::Tensor &maskedX,
                                    const torch::Tensor &key,
                                    const torch::Tensor &seed,
                                    int                  partyId,
                                    const torch::Tensor &leftEndpoints,
                                    const torch::Tensor &rightEndpoints,
                                    std::size_t          bitWidthIn,
                                    std::size_t          elementSize,
                                    std::size_t          elementNum)
{
    std::size_t cacheDataSize = 0;
    int         ret           = FastFss_grottoGetCacheDataSize(&cacheDataSize, bitWidthIn, elementSize, elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_grottoGetCacheDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    auto cacheData = torch::empty(cacheDataSize, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    ret = FastFss_cuda_grottoMICEval(                           //
        out.mutable_data_ptr(),                                 //
        out.element_size() * out.numel(),                       //
        maskedX.const_data_ptr(),                               //
        maskedX.element_size() * maskedX.numel(),               //
        key.const_data_ptr(),                                   //
        key.element_size() * key.numel(),                       //
        seed.const_data_ptr(),                                  //
        seed.element_size() * seed.numel(),                     //
        partyId,                                                //
        leftEndpoints.const_data_ptr(),                         //
        leftEndpoints.element_size() * leftEndpoints.numel(),   //
        rightEndpoints.const_data_ptr(),                        //
        rightEndpoints.element_size() * rightEndpoints.numel(), //
        bitWidthIn,                                             //
        elementSize,                                            //
        elementNum,                                             //
        cacheData.mutable_data_ptr(),                           //
        cacheData.element_size() * cacheData.numel(),           //
        &stream                                                 //
    );                                                          //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_grottoMICEval failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return out;
}

std::tuple<torch::Tensor &, torch::Tensor &> cuda_grotto_interval_lut_eval(torch::Tensor       &outE,
                                                                           torch::Tensor       &outT,
                                                                           const torch::Tensor &maskedX,
                                                                           const torch::Tensor &key,
                                                                           const torch::Tensor &seed,
                                                                           int                  partyId,
                                                                           const torch::Tensor &leftEndpoints,
                                                                           const torch::Tensor &rightEndpoints,
                                                                           const torch::Tensor &lookUpTable,
                                                                           std::size_t          bitWidthIn,
                                                                           std::size_t          bitWidthOut,
                                                                           std::size_t          elementSize,
                                                                           std::size_t          elementNum)
{
    std::size_t cacheDataSize = 0;
    int         ret           = FastFss_grottoGetCacheDataSize(&cacheDataSize, bitWidthIn, elementSize, elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_grottoGetCacheDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    auto cacheData = torch::empty(cacheDataSize, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    ret = FastFss_cuda_grottoIntervalLutEval(                   //
        outE.mutable_data_ptr(),                                //
        outE.element_size() * outE.numel(),                     //
        outT.mutable_data_ptr(),                                //
        outT.element_size() * outT.numel(),                     //
        maskedX.const_data_ptr(),                               //
        maskedX.element_size() * maskedX.numel(),               //
        key.const_data_ptr(),                                   //
        key.element_size() * key.numel(),                       //
        seed.const_data_ptr(),                                  //
        seed.element_size() * seed.numel(),                     //
        partyId,                                                //
        leftEndpoints.const_data_ptr(),                         //
        leftEndpoints.element_size() * leftEndpoints.numel(),   //
        rightEndpoints.const_data_ptr(),                        //
        rightEndpoints.element_size() * rightEndpoints.numel(), //
        lookUpTable.const_data_ptr(),                           //
        lookUpTable.element_size() * lookUpTable.numel(),       //
        bitWidthIn,                                             //
        bitWidthOut,                                            //
        elementSize,                                            //
        elementNum,                                             //
        cacheData.mutable_data_ptr(),                           //
        cacheData.element_size() * cacheData.numel(),           //
        &stream                                                 //
    );                                                          //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_grottoIntervalLutEval failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return std::tuple<torch::Tensor &, torch::Tensor &>(outE, outT);
}
