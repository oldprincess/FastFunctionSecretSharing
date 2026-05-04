#include "dcf_cpu.h"

#include <FastFss/cpu/dcf.h>
#include <FastFss/dcf.h>

torch::Tensor& cpu_dcf_key_gen(torch::Tensor&       key,
                               const torch::Tensor& alpha,
                               const torch::Tensor& beta,
                               const torch::Tensor& seed0,
                               const torch::Tensor& seed1,
                               std::size_t          bitWidthIn,
                               std::size_t          bitWidthOut,
                               std::size_t          groupSize,
                               std::size_t          elementSize,
                               std::size_t          elementNum)
{
    int ret = FastFss_cpu_dcfKeyGen(          //
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
        elementNum                            //
    );                                        //
    if (ret != 0)
    {
        std::string msg = "FastFss_cpu_dcfKeyGen failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return key;
}

torch::Tensor& cpu_dcf_eval(torch::Tensor&       out,
                            const torch::Tensor& maskedX,
                            const torch::Tensor& key,
                            const torch::Tensor& seed,
                            int                  partyId,
                            std::size_t          bitWidthIn,
                            std::size_t          bitWidthOut,
                            std::size_t          groupSize,
                            std::size_t          elementSize,
                            std::size_t          elementNum)
{
    int ret = FastFss_cpu_dcfEval(                //
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
        0                                         //
    );                                            //
    if (ret != 0)
    {
        std::string msg = "FastFss_cpu_dcfEval failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return out;
}
