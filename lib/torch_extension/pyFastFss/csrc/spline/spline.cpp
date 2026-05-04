#include "spline.h"

#include <FastFss/spline.h>

#include "spline_cpu.h"
#ifdef FAST_FSS_ENABLE_CUDA
#include "spline_cuda.h"
#endif

std::size_t spline_get_key_data_size(std::size_t degree,
                                     std::size_t intervalNum,
                                     std::size_t bitWidthIn,
                                     std::size_t bitWidthOut,
                                     std::size_t elementSize,
                                     std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int ret = FastFss_dcfSplineGetKeyDataSize(&keyDataSize, degree, intervalNum, bitWidthIn, bitWidthOut, elementSize,
                                              elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_dcfSplineGetKeyDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return keyDataSize;
}

std::tuple<torch::Tensor&, torch::Tensor&, torch::Tensor&> spline_key_gen(torch::Tensor&       key,
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
                                                                          std::size_t          bitWidthOut)
{
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(e.is_contiguous());
    TORCH_CHECK(beta.is_contiguous());
    TORCH_CHECK(alpha.is_contiguous());
    TORCH_CHECK(seed0.is_contiguous());
    TORCH_CHECK(seed1.is_contiguous());
    TORCH_CHECK(coefficients.is_contiguous());
    TORCH_CHECK(leftEndpoints.is_contiguous());
    TORCH_CHECK(rightEndpoints.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
    TORCH_CHECK(seed0.dtype() == torch::kUInt8);
    TORCH_CHECK(seed1.dtype() == torch::kUInt8);
    auto dtype = alpha.dtype();
    TORCH_CHECK(e.dtype() == dtype);
    TORCH_CHECK(beta.dtype() == dtype);
    TORCH_CHECK(coefficients.dtype() == dtype);
    TORCH_CHECK(leftEndpoints.dtype() == dtype);
    TORCH_CHECK(rightEndpoints.dtype() == dtype);

    std::size_t elementSize = 0;
    std::size_t elementNum  = 0;
    if (bitWidthIn > 64 || bitWidthOut > 64)
    {
        TORCH_CHECK(dtype == torch::kInt64);
        auto limbNum = alpha.sizes().back();
        TORCH_CHECK(coefficients.sizes().back() == limbNum);
        TORCH_CHECK(leftEndpoints.sizes().back() == limbNum);
        TORCH_CHECK(rightEndpoints.sizes().back() == limbNum);
        elementSize = sizeof(std::int64_t) * limbNum;
        elementNum  = alpha.numel() / limbNum;
    }
    else
    {
        elementSize = dtype.itemsize();
        elementNum  = alpha.numel();
    }

    auto device = key.device();
    TORCH_CHECK(e.device() == device);
    TORCH_CHECK(beta.device() == device);
    TORCH_CHECK(alpha.device() == device);
    TORCH_CHECK(seed0.device() == device);
    TORCH_CHECK(seed1.device() == device);
    TORCH_CHECK(coefficients.device() == device);
    TORCH_CHECK(leftEndpoints.device() == device);
    TORCH_CHECK(rightEndpoints.device() == device);

    std::size_t coeffNum  = degree + 1;
    std::size_t groupSize = intervalNum * coeffNum;
    std::size_t keyDataSize =
        spline_get_key_data_size(degree, intervalNum, bitWidthIn, bitWidthOut, elementSize, elementNum);
    key.resize_({(std::int64_t)keyDataSize});

    auto                      group_sizes_raw = alpha.sizes();
    std::vector<std::int64_t> group_sizes(group_sizes_raw.begin(), group_sizes_raw.end());
    group_sizes.push_back((std::int64_t)groupSize);
    e.resize_(group_sizes);
    beta.resize_(group_sizes);

    if (device.is_cpu())
    {
        return cpu_spline_key_gen(key, e, beta, alpha, seed0, seed1, coefficients, degree, leftEndpoints,
                                  rightEndpoints, intervalNum, bitWidthIn, bitWidthOut, elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_spline_key_gen(key, e, beta, alpha, seed0, seed1, coefficients, degree, leftEndpoints,
                                   rightEndpoints, intervalNum, bitWidthIn, bitWidthOut, elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}

torch::Tensor& spline_eval(torch::Tensor&       out,
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
                           std::size_t          bitWidthOut)
{
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(maskedX.is_contiguous());
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(sharedE.is_contiguous());
    TORCH_CHECK(sharedBeta.is_contiguous());
    TORCH_CHECK(seed.is_contiguous());
    TORCH_CHECK(leftEndpoints.is_contiguous());
    TORCH_CHECK(rightEndpoints.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
    TORCH_CHECK(seed.dtype() == torch::kUInt8);
    auto dtype = maskedX.dtype();
    TORCH_CHECK(out.dtype() == dtype);
    TORCH_CHECK(sharedE.dtype() == dtype);
    TORCH_CHECK(sharedBeta.dtype() == dtype);
    TORCH_CHECK(leftEndpoints.dtype() == dtype);
    TORCH_CHECK(rightEndpoints.dtype() == dtype);

    std::size_t elementSize = 0;
    std::size_t elementNum  = 0;
    if (bitWidthIn > 64 || bitWidthOut > 64)
    {
        TORCH_CHECK(dtype == torch::kInt64);
        auto limbNum = maskedX.sizes().back();
        TORCH_CHECK(sharedE.sizes().back() == limbNum);
        TORCH_CHECK(sharedBeta.sizes().back() == limbNum);
        TORCH_CHECK(leftEndpoints.sizes().back() == limbNum);
        TORCH_CHECK(rightEndpoints.sizes().back() == limbNum);
        elementSize = sizeof(std::int64_t) * limbNum;
        elementNum  = maskedX.numel() / limbNum;
    }
    else
    {
        elementSize = dtype.itemsize();
        elementNum  = maskedX.numel();
    }
    out.resize_(maskedX.sizes());

    auto device = out.device();
    TORCH_CHECK(maskedX.device() == device);
    TORCH_CHECK(key.device() == device);
    TORCH_CHECK(sharedE.device() == device);
    TORCH_CHECK(sharedBeta.device() == device);
    TORCH_CHECK(seed.device() == device);
    TORCH_CHECK(leftEndpoints.device() == device);
    TORCH_CHECK(rightEndpoints.device() == device);

    if (device.is_cpu())
    {
        return cpu_spline_eval(out, maskedX, key, sharedE, sharedBeta, seed, partyId, leftEndpoints, rightEndpoints,
                               intervalNum, degree, bitWidthIn, bitWidthOut, elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_spline_eval(out, maskedX, key, sharedE, sharedBeta, seed, partyId, leftEndpoints, rightEndpoints,
                                intervalNum, degree, bitWidthIn, bitWidthOut, elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}
