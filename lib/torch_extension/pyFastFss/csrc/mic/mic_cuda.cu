#include <FastFss/cuda/mic.h>
#include <FastFss/mic.h>
#include <c10/cuda/CUDAStream.h>

#include "mic_cuda.h"

std::tuple<torch::Tensor&, torch::Tensor&> cuda_dcf_mic_key_gen(torch::Tensor&       key,
                                                                torch::Tensor&       z,
                                                                const torch::Tensor& alpha,
                                                                const torch::Tensor& seed0,
                                                                const torch::Tensor& seed1,
                                                                const torch::Tensor& leftEndpoints,
                                                                const torch::Tensor& rightEndpoints,
                                                                std::size_t          bitWidthIn,
                                                                std::size_t          bitWidthOut,
                                                                std::size_t          elementSize,
                                                                std::size_t          elementNum)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int ret = FastFss_cuda_dcfMICKeyGen(        //
        key.mutable_data_ptr(),                 //
        key.element_size() * key.numel(),       //
        z.mutable_data_ptr(),                   //
        z.element_size() * z.numel(),           //
        alpha.const_data_ptr(),                 //
        alpha.element_size() * alpha.numel(),   //
        seed0.const_data_ptr(),                 //
        seed0.element_size() * seed0.numel(),   //
        seed1.const_data_ptr(),                 //
        seed1.element_size() * seed1.numel(),   //
        leftEndpoints.const_data_ptr(),         //
        leftEndpoints.element_size() * leftEndpoints.numel(),   //
        rightEndpoints.const_data_ptr(),        //
        rightEndpoints.element_size() * rightEndpoints.numel(), //
        bitWidthIn,                             //
        bitWidthOut,                            //
        elementSize,                            //
        elementNum,                             //
        &stream                                 //
    );                                          //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_dcfMICKeyGen failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return std::tuple<torch::Tensor&, torch::Tensor&>(key, z);
}

torch::Tensor& cuda_dcf_mic_eval(torch::Tensor&       out,
                                 const torch::Tensor& maskedX,
                                 const torch::Tensor& key,
                                 const torch::Tensor& sharedZ,
                                 const torch::Tensor& seed,
                                 int                  partyId,
                                 const torch::Tensor& leftEndpoints,
                                 const torch::Tensor& rightEndpoints,
                                 std::size_t          bitWidthIn,
                                 std::size_t          bitWidthOut,
                                 std::size_t          elementSize,
                                 std::size_t          elementNum)
{
    std::size_t cacheDataSize = 0;
    int ret = FastFss_dcfMICGetCacheDataSize(&cacheDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_dcfMICGetCacheDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    auto cacheData = torch::empty(cacheDataSize, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    ret = FastFss_cuda_dcfMICEval(                //
        out.mutable_data_ptr(),                   //
        out.element_size() * out.numel(),         //
        maskedX.const_data_ptr(),                 //
        maskedX.element_size() * maskedX.numel(), //
        key.const_data_ptr(),                     //
        key.element_size() * key.numel(),         //
        sharedZ.const_data_ptr(),                 //
        sharedZ.element_size() * sharedZ.numel(), //
        seed.const_data_ptr(),                    //
        seed.element_size() * seed.numel(),       //
        partyId,                                  //
        leftEndpoints.const_data_ptr(),           //
        leftEndpoints.element_size() * leftEndpoints.numel(),   //
        rightEndpoints.const_data_ptr(),          //
        rightEndpoints.element_size() * rightEndpoints.numel(), //
        bitWidthIn,                               //
        bitWidthOut,                              //
        elementSize,                              //
        elementNum,                               //
        cacheData.mutable_data_ptr(),             //
        cacheData.element_size() * cacheData.numel(), //
        &stream                                   //
    );                                            //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_dcfMICEval failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return out;
}
