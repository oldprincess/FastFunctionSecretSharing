#include <FastFss/cuda/dpf.h>
#include <FastFss/dpf.h>
#include <c10/cuda/CUDAStream.h>

#include "dpf_cuda.h"

torch::Tensor &cuda_dpf_key_gen(torch::Tensor       &key,
                                const torch::Tensor &alpha,
                                const torch::Tensor &beta,
                                const torch::Tensor &seed0,
                                const torch::Tensor &seed1,
                                std::size_t          bitWidthIn,
                                std::size_t          bitWidthOut,
                                std::size_t          groupSize,
                                std::size_t          elementSize,
                                std::size_t          elementNum)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int ret = FastFss_cuda_dpfKeyGen(         //
        key.mutable_data_ptr(),               //
        key.element_size() * key.numel(),     //
        alpha.const_data_ptr(),               //
        alpha.element_size() * alpha.numel(), //
        beta.const_data_ptr(),                //
        beta.element_size() * beta.numel(),   //
        seed0.const_data_ptr(),               //
        seed0.element_size() * seed0.numel(), //
        seed1.const_data_ptr(),               //
        seed1.element_size() * seed1.numel(), //
        bitWidthIn,                           //
        bitWidthOut,                          //
        groupSize,                            //
        elementSize,                          //
        elementNum,                           //
        &stream                               //
    );                                        //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_dpfKeyGen failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return key;
}

torch::Tensor &cuda_dpf_eval(torch::Tensor       &out,
                             const torch::Tensor &maskedX,
                             const torch::Tensor &key,
                             const torch::Tensor &seed,
                             int                  partyId,
                             std::size_t          bitWidthIn,
                             std::size_t          bitWidthOut,
                             std::size_t          groupSize,
                             std::size_t          elementSize,
                             std::size_t          elementNum)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int ret = FastFss_cuda_dpfEval(               //
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
        bitWidthOut,                              //
        groupSize,                                //
        elementSize,                              //
        elementNum,                               //
        nullptr,                                  //
        0,                                        //
        &stream                                   //
    );                                            //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_dpfEval failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return out;
}

torch::Tensor &cuda_dpf_eval_all(torch::Tensor       &out,
                                 const torch::Tensor &maskedX,
                                 const torch::Tensor &key,
                                 const torch::Tensor &seed,
                                 int                  partyId,
                                 std::size_t          bitWidthIn,
                                 std::size_t          bitWidthOut,
                                 std::size_t          groupSize,
                                 std::size_t          elementSize,
                                 std::size_t          elementNum)
{
    std::size_t cacheDataSize = 0;
    int         ret           = FastFss_dpfGetCacheDataSize(&cacheDataSize, bitWidthIn, elementSize, elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_dpfGetCacheDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    auto cacheData = torch::empty(cacheDataSize, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    ret = FastFss_cuda_dpfEvalAll(                    //
        out.mutable_data_ptr(),                       //
        out.element_size() * out.numel(),             //
        maskedX.const_data_ptr(),                     //
        maskedX.element_size() * maskedX.numel(),     //
        key.const_data_ptr(),                         //
        key.element_size() * key.numel(),             //
        seed.const_data_ptr(),                        //
        seed.element_size() * seed.numel(),           //
        partyId,                                      //
        bitWidthIn,                                   //
        bitWidthOut,                                  //
        groupSize,                                    //
        elementSize,                                  //
        elementNum,                                   //
        cacheData.mutable_data_ptr(),                 //
        cacheData.element_size() * cacheData.numel(), //
        &stream                                       //
    );                                                //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_dpfEvalAll failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return out;
}

torch::Tensor &cuda_dpf_eval_multi(torch::Tensor       &out,
                                   const torch::Tensor &maskedX,
                                   const torch::Tensor &key,
                                   const torch::Tensor &seed,
                                   int                  partyId,
                                   const torch::Tensor &point,
                                   std::size_t          bitWidthIn,
                                   std::size_t          bitWidthOut,
                                   std::size_t          groupSize,
                                   std::size_t          elementSize,
                                   std::size_t          elementNum)
{
    std::size_t cacheDataSize = 0;
    int         ret           = FastFss_dpfGetCacheDataSize(&cacheDataSize, bitWidthIn, elementSize, elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_dpfGetCacheDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    auto cacheData = torch::empty(cacheDataSize, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    ret = FastFss_cuda_dpfEvalMulti(                  //
        out.mutable_data_ptr(),                       //
        out.element_size() * out.numel(),             //
        maskedX.const_data_ptr(),                     //
        maskedX.element_size() * maskedX.numel(),     //
        key.const_data_ptr(),                         //
        key.element_size() * key.numel(),             //
        seed.const_data_ptr(),                        //
        seed.element_size() * seed.numel(),           //
        partyId,                                      //
        point.const_data_ptr(),                       //
        point.element_size() * point.numel(),         //
        bitWidthIn,                                   //
        bitWidthOut,                                  //
        groupSize,                                    //
        elementSize,                                  //
        elementNum,                                   //
        cacheData.mutable_data_ptr(),                 //
        cacheData.element_size() * cacheData.numel(), //
        &stream                                       //
    );                                                //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_dpfEvalMulti failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return out;
}
