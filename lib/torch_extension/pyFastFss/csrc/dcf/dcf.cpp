#include "dcf.h"

#include <FastFss/dcf.h>

#include "dcf_cpu.h"
#ifdef FAST_FSS_ENABLE_CUDA
#include "dcf_cuda.h"
#endif

std::size_t dcf_get_key_data_size(std::size_t bitWidthIn,
                                  std::size_t bitWidthOut,
                                  std::size_t groupSize,
                                  std::size_t elementSize,
                                  std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int ret = FastFss_dcfGetKeyDataSize(&keyDataSize, bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_dcfGetKeyDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return keyDataSize;
}

torch::Tensor& dcf_key_gen(torch::Tensor&       key,
                           const torch::Tensor& alpha,
                           const torch::Tensor& beta,
                           const torch::Tensor& seed0,
                           const torch::Tensor& seed1,
                           std::size_t          bitWidthIn,
                           std::size_t          bitWidthOut,
                           std::size_t          groupSize)
{
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(alpha.is_contiguous());
    TORCH_CHECK(beta.is_contiguous());
    TORCH_CHECK(seed0.is_contiguous());
    TORCH_CHECK(seed1.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
    TORCH_CHECK(seed0.dtype() == torch::kUInt8);
    TORCH_CHECK(seed1.dtype() == torch::kUInt8);
    auto dtype = alpha.dtype();
    TORCH_CHECK(beta.dtype() == dtype);

    std::size_t elementSize = 0;
    std::size_t elementNum  = 0;
    if (bitWidthIn > 64 || bitWidthOut > 64)
    {
        TORCH_CHECK(dtype == torch::kInt64);
        auto limbNum = alpha.sizes().back();
        TORCH_CHECK(beta.sizes().back() == limbNum);
        elementSize = sizeof(std::int64_t) * limbNum;
        elementNum  = alpha.numel() / limbNum;
    }
    else
    {
        elementSize = dtype.itemsize();
        elementNum  = alpha.numel();
    }

    auto device = key.device();
    TORCH_CHECK(alpha.device() == device);
    TORCH_CHECK(beta.device() == device);
    TORCH_CHECK(seed0.device() == device);
    TORCH_CHECK(seed1.device() == device);

    std::size_t keyDataSize = dcf_get_key_data_size(bitWidthIn, bitWidthOut, groupSize, elementSize, elementNum);
    key.resize_({(std::int64_t)keyDataSize});

    if (device.is_cpu())
    {
        return cpu_dcf_key_gen(key, alpha, beta, seed0, seed1, bitWidthIn, bitWidthOut, groupSize, elementSize,
                               elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_dcf_key_gen(key, alpha, beta, seed0, seed1, bitWidthIn, bitWidthOut, groupSize, elementSize,
                                elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}

torch::Tensor& dcf_eval(torch::Tensor&       out,
                        const torch::Tensor& maskedX,
                        const torch::Tensor& key,
                        const torch::Tensor& seed,
                        int                  partyId,
                        std::size_t          bitWidthIn,
                        std::size_t          bitWidthOut,
                        std::size_t          groupSize)
{
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(maskedX.is_contiguous());
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(seed.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
    TORCH_CHECK(seed.dtype() == torch::kUInt8);
    auto dtype = maskedX.dtype();
    TORCH_CHECK(out.dtype() == dtype);

    std::size_t elementSize = 0;
    std::size_t elementNum  = 0;
    if (bitWidthIn > 64 || bitWidthOut > 64)
    {
        TORCH_CHECK(dtype == torch::kInt64);
        auto limbNum = maskedX.sizes().back();
        elementSize  = sizeof(std::int64_t) * limbNum;
        elementNum   = maskedX.numel() / limbNum;
    }
    else
    {
        elementSize = dtype.itemsize();
        elementNum  = maskedX.numel();
    }

    auto                      out_sizes_raw = maskedX.sizes();
    std::vector<std::int64_t> out_sizes(out_sizes_raw.begin(), out_sizes_raw.end());
    out_sizes.push_back((std::int64_t)groupSize);
    out.resize_(out_sizes);

    auto device = out.device();
    TORCH_CHECK(maskedX.device() == device);
    TORCH_CHECK(key.device() == device);
    TORCH_CHECK(seed.device() == device);

    if (device.is_cpu())
    {
        return cpu_dcf_eval(out, maskedX, key, seed, partyId, bitWidthIn, bitWidthOut, groupSize, elementSize,
                            elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_dcf_eval(out, maskedX, key, seed, partyId, bitWidthIn, bitWidthOut, groupSize, elementSize,
                             elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}
