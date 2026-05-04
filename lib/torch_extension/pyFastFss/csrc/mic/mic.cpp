#include "mic.h"

#include <FastFss/mic.h>

#include "mic_cpu.h"
#ifdef FAST_FSS_ENABLE_CUDA
#include "mic_cuda.h"
#endif

std::size_t dcf_mic_get_key_data_size(std::size_t bitWidthIn,
                                      std::size_t bitWidthOut,
                                      std::size_t elementSize,
                                      std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         ret = FastFss_dcfMICGetKeyDataSize(&keyDataSize, bitWidthIn, bitWidthOut, elementSize, elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_dcfMICGetKeyDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return keyDataSize;
}

std::tuple<torch::Tensor&, torch::Tensor&> dcf_mic_key_gen(torch::Tensor&       key,
                                                           torch::Tensor&       z,
                                                           const torch::Tensor& alpha,
                                                           const torch::Tensor& seed0,
                                                           const torch::Tensor& seed1,
                                                           const torch::Tensor& leftEndpoints,
                                                           const torch::Tensor& rightEndpoints,
                                                           std::size_t          bitWidthIn,
                                                           std::size_t          bitWidthOut)
{
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(z.is_contiguous());
    TORCH_CHECK(alpha.is_contiguous());
    TORCH_CHECK(seed0.is_contiguous());
    TORCH_CHECK(seed1.is_contiguous());
    TORCH_CHECK(leftEndpoints.is_contiguous());
    TORCH_CHECK(rightEndpoints.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
    TORCH_CHECK(seed0.dtype() == torch::kUInt8);
    TORCH_CHECK(seed1.dtype() == torch::kUInt8);
    auto dtype = alpha.dtype();
    TORCH_CHECK(z.dtype() == dtype);
    TORCH_CHECK(leftEndpoints.dtype() == dtype);
    TORCH_CHECK(rightEndpoints.dtype() == dtype);

    std::size_t elementSize = 0;
    std::size_t elementNum  = 0;
    if (bitWidthIn > 64 || bitWidthOut > 64)
    {
        TORCH_CHECK(dtype == torch::kInt64);
        auto limbNum = alpha.sizes().back();
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
    TORCH_CHECK(z.device() == device);
    TORCH_CHECK(alpha.device() == device);
    TORCH_CHECK(seed0.device() == device);
    TORCH_CHECK(seed1.device() == device);
    TORCH_CHECK(leftEndpoints.device() == device);
    TORCH_CHECK(rightEndpoints.device() == device);

    std::size_t intervalNum = leftEndpoints.numel();
    if (bitWidthIn > 64 || bitWidthOut > 64)
    {
        intervalNum /= leftEndpoints.sizes().back();
    }

    std::size_t keyDataSize = dcf_mic_get_key_data_size(bitWidthIn, bitWidthOut, elementSize, elementNum);
    key.resize_({(std::int64_t)keyDataSize});
    auto                      z_sizes_raw = alpha.sizes();
    std::vector<std::int64_t> z_sizes(z_sizes_raw.begin(), z_sizes_raw.end());
    z_sizes.push_back((std::int64_t)intervalNum);
    z.resize_(z_sizes);

    if (device.is_cpu())
    {
        return cpu_dcf_mic_key_gen(key, z, alpha, seed0, seed1, leftEndpoints, rightEndpoints, bitWidthIn, bitWidthOut,
                                   elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_dcf_mic_key_gen(key, z, alpha, seed0, seed1, leftEndpoints, rightEndpoints, bitWidthIn, bitWidthOut,
                                    elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}

torch::Tensor& dcf_mic_eval(torch::Tensor&       out,
                            const torch::Tensor& maskedX,
                            const torch::Tensor& key,
                            const torch::Tensor& sharedZ,
                            const torch::Tensor& seed,
                            int                  partyId,
                            const torch::Tensor& leftEndpoints,
                            const torch::Tensor& rightEndpoints,
                            std::size_t          bitWidthIn,
                            std::size_t          bitWidthOut)
{
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(maskedX.is_contiguous());
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(sharedZ.is_contiguous());
    TORCH_CHECK(seed.is_contiguous());
    TORCH_CHECK(leftEndpoints.is_contiguous());
    TORCH_CHECK(rightEndpoints.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
    TORCH_CHECK(seed.dtype() == torch::kUInt8);
    auto dtype = maskedX.dtype();
    TORCH_CHECK(out.dtype() == dtype);
    TORCH_CHECK(sharedZ.dtype() == dtype);
    TORCH_CHECK(leftEndpoints.dtype() == dtype);
    TORCH_CHECK(rightEndpoints.dtype() == dtype);

    std::size_t elementSize = 0;
    std::size_t elementNum  = 0;
    if (bitWidthIn > 64 || bitWidthOut > 64)
    {
        TORCH_CHECK(dtype == torch::kInt64);
        auto limbNum = maskedX.sizes().back();
        TORCH_CHECK(sharedZ.sizes().back() == limbNum);
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

    auto device = out.device();
    TORCH_CHECK(maskedX.device() == device);
    TORCH_CHECK(key.device() == device);
    TORCH_CHECK(sharedZ.device() == device);
    TORCH_CHECK(seed.device() == device);
    TORCH_CHECK(leftEndpoints.device() == device);
    TORCH_CHECK(rightEndpoints.device() == device);

    std::size_t intervalNum = leftEndpoints.numel();
    if (bitWidthIn > 64 || bitWidthOut > 64)
    {
        intervalNum /= leftEndpoints.sizes().back();
    }

    auto                      out_sizes_raw = maskedX.sizes();
    std::vector<std::int64_t> out_sizes(out_sizes_raw.begin(), out_sizes_raw.end());
    out_sizes.push_back((std::int64_t)intervalNum);
    out.resize_(out_sizes);

    if (device.is_cpu())
    {
        return cpu_dcf_mic_eval(out, maskedX, key, sharedZ, seed, partyId, leftEndpoints, rightEndpoints, bitWidthIn,
                                bitWidthOut, elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_dcf_mic_eval(out, maskedX, key, sharedZ, seed, partyId, leftEndpoints, rightEndpoints, bitWidthIn,
                                 bitWidthOut, elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}
