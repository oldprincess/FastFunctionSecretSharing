#include "spline_cpu.h"

#include <FastFss/cpu/spline.h>
#include <FastFss/spline.h>

std::tuple<torch::Tensor&, torch::Tensor&, torch::Tensor&> cpu_spline_key_gen(torch::Tensor&       key,
                                                                               torch::Tensor&       e,
                                                                               torch::Tensor&       beta,
                                                                               const torch::Tensor& alpha,
                                                                               const torch::Tensor& seed0,
                                                                               const torch::Tensor& seed1,
                                                                               const torch::Tensor& coefficients,
                                                                               std::size_t          degree,
                                                                               const torch::Tensor& leftEndpoints,
                                                                               const torch::Tensor& rightEndpoints,
                                                                               std::size_t          intervalNum,
                                                                               std::size_t          bitWidthIn,
                                                                               std::size_t          bitWidthOut,
                                                                               std::size_t          elementSize,
                                                                               std::size_t          elementNum)
{
    int ret = FastFss_cpu_dcfSplineKeyGen(      //
        key.mutable_data_ptr(),                 //
        key.element_size() * key.numel(),       //
        e.mutable_data_ptr(),                   //
        e.element_size() * e.numel(),           //
        beta.mutable_data_ptr(),                //
        beta.element_size() * beta.numel(),     //
        alpha.const_data_ptr(),                 //
        alpha.element_size() * alpha.numel(),   //
        seed0.const_data_ptr(),                 //
        seed0.element_size() * seed0.numel(),   //
        seed1.const_data_ptr(),                 //
        seed1.element_size() * seed1.numel(),   //
        coefficients.const_data_ptr(),          //
        coefficients.element_size() * coefficients.numel(), //
        degree,                                 //
        leftEndpoints.const_data_ptr(),         //
        leftEndpoints.element_size() * leftEndpoints.numel(),   //
        rightEndpoints.const_data_ptr(),        //
        rightEndpoints.element_size() * rightEndpoints.numel(), //
        intervalNum,                            //
        bitWidthIn,                             //
        bitWidthOut,                            //
        elementSize,                            //
        elementNum                              //
    );                                          //
    if (ret != 0)
    {
        std::string msg = "FastFss_cpu_dcfSplineKeyGen failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return std::tuple<torch::Tensor&, torch::Tensor&, torch::Tensor&>(key, e, beta);
}

torch::Tensor& cpu_spline_eval(torch::Tensor&       out,
                               const torch::Tensor& maskedX,
                               const torch::Tensor& key,
                               const torch::Tensor& sharedE,
                               const torch::Tensor& sharedBeta,
                               const torch::Tensor& seed,
                               int                  partyId,
                               const torch::Tensor& leftEndpoints,
                               const torch::Tensor& rightEndpoints,
                               std::size_t          intervalNum,
                               std::size_t          degree,
                               std::size_t          bitWidthIn,
                               std::size_t          bitWidthOut,
                               std::size_t          elementSize,
                               std::size_t          elementNum)
{
    std::size_t cacheDataSize = 0;
    int ret = FastFss_dcfSplineGetCacheDataSize(
        &cacheDataSize, degree, intervalNum, bitWidthIn, bitWidthOut, elementSize, elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_dcfSplineGetCacheDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    auto cacheData = torch::empty(cacheDataSize, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));

    ret = FastFss_cpu_dcfSplineEval(              //
        out.mutable_data_ptr(),                   //
        out.element_size() * out.numel(),         //
        maskedX.const_data_ptr(),                 //
        maskedX.element_size() * maskedX.numel(), //
        key.const_data_ptr(),                     //
        key.element_size() * key.numel(),         //
        sharedE.const_data_ptr(),                 //
        sharedE.element_size() * sharedE.numel(), //
        sharedBeta.const_data_ptr(),              //
        sharedBeta.element_size() * sharedBeta.numel(), //
        seed.const_data_ptr(),                    //
        seed.element_size() * seed.numel(),       //
        partyId,                                  //
        leftEndpoints.const_data_ptr(),           //
        leftEndpoints.element_size() * leftEndpoints.numel(),   //
        rightEndpoints.const_data_ptr(),          //
        rightEndpoints.element_size() * rightEndpoints.numel(), //
        intervalNum,                              //
        degree,                                   //
        bitWidthIn,                               //
        bitWidthOut,                              //
        elementSize,                              //
        elementNum,                               //
        cacheData.mutable_data_ptr(),             //
        cacheData.element_size() * cacheData.numel() //
    );                                            //
    if (ret != 0)
    {
        std::string msg = "FastFss_cpu_dcfSplineEval failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return out;
}
