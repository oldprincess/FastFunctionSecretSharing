#include <FastFss/cuda/ottt.h>
#include <FastFss/ottt.h>
#include <c10/cuda/CUDAStream.h>

#include "ottt_cuda.h"

torch::Tensor &cuda_ottt_key_gen(torch::Tensor       &key,
                                 const torch::Tensor &alpha,
                                 std::size_t          bitWidthIn,
                                 std::size_t          elementSize,
                                 std::size_t          elementNum)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int ret = FastFss_cuda_otttKeyGen(        //
        key.mutable_data_ptr(),               //
        key.element_size() * key.numel(),     //
        alpha.const_data_ptr(),               //
        alpha.element_size() * alpha.numel(), //
        bitWidthIn,                           //
        elementSize,                          //
        elementNum,                           //
        &stream                               //
    );                                        //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_otttKeyGen failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return key;
}

std::tuple<torch::Tensor &, torch::Tensor &> cuda_ottt_lut_eval(torch::Tensor       &outE,
                                                                torch::Tensor       &outT,
                                                                const torch::Tensor &maskedX,
                                                                const torch::Tensor &key,
                                                                int                  partyId,
                                                                const torch::Tensor &lookUpTable,
                                                                std::size_t          bitWidthIn,
                                                                std::size_t          elementSize,
                                                                std::size_t          elementNum)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int ret = FastFss_cuda_otttLutEval(                   //
        outE.mutable_data_ptr(),                          //
        outE.element_size() * outE.numel(),               //
        outT.mutable_data_ptr(),                          //
        outT.element_size() * outT.numel(),               //
        maskedX.const_data_ptr(),                         //
        maskedX.element_size() * maskedX.numel(),         //
        key.const_data_ptr(),                             //
        key.element_size() * key.numel(),                 //
        partyId,                                          //
        lookUpTable.const_data_ptr(),                     //
        lookUpTable.element_size() * lookUpTable.numel(), //
        bitWidthIn,                                       //
        elementSize,                                      //
        elementNum,                                       //
        &stream                                           //
    );                                                    //
    if (ret != 0)
    {
        std::string msg = "FastFss_cuda_otttLutEval failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return std::tuple<torch::Tensor &, torch::Tensor &>(outE, outT);
}
