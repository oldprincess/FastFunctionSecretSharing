#include "grotto.h"

#include <FastFss/grotto.h>

#include "grotto_cpu.h"
#ifdef FAST_FSS_ENABLE_CUDA
#include "grotto_cuda.h"
#endif

std::size_t grotto_get_key_data_size(std::size_t bitWidthIn, std::size_t elementSize, std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         ret         = FastFss_grottoGetKeyDataSize(&keyDataSize, bitWidthIn, elementSize, elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_grottoGetKeyDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return keyDataSize;
}

torch::Tensor& grotto_key_gen(torch::Tensor&       key,
                              const torch::Tensor& alpha,
                              const torch::Tensor& seed0,
                              const torch::Tensor& seed1,
                              std::size_t          bitWidthIn)
{
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(alpha.is_contiguous());
    TORCH_CHECK(seed0.is_contiguous());
    TORCH_CHECK(seed1.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
    TORCH_CHECK(seed0.dtype() == torch::kUInt8);
    TORCH_CHECK(seed1.dtype() == torch::kUInt8);
    auto dtype = alpha.dtype();

    std::size_t elementSize = 0;
    std::size_t elementNum  = 0;
    if (bitWidthIn > 64)
    {
        TORCH_CHECK(dtype == torch::kInt64);
        auto limbNum = alpha.sizes().back();
        elementSize  = sizeof(std::int64_t) * limbNum;
        elementNum   = alpha.numel() / limbNum;
    }
    else
    {
        elementSize = dtype.itemsize();
        elementNum  = alpha.numel();
    }

    auto device = key.device();
    TORCH_CHECK(alpha.device() == device);
    TORCH_CHECK(seed0.device() == device);
    TORCH_CHECK(seed1.device() == device);

    std::size_t keyDataSize = grotto_get_key_data_size(bitWidthIn, elementSize, elementNum);
    key.resize_({(std::int64_t)keyDataSize});

    if (device.is_cpu())
    {
        return cpu_grotto_key_gen(key, alpha, seed0, seed1, bitWidthIn, elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_grotto_key_gen(key, alpha, seed0, seed1, bitWidthIn, elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}

torch::Tensor& grotto_eq_eval(torch::Tensor&       out,
                              const torch::Tensor& maskedX,
                              const torch::Tensor& key,
                              const torch::Tensor& seed,
                              int                  partyId,
                              std::size_t          bitWidthIn)
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
    if (bitWidthIn > 64)
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

    auto device = out.device();
    TORCH_CHECK(maskedX.device() == device);
    TORCH_CHECK(key.device() == device);
    TORCH_CHECK(seed.device() == device);

    if (device.is_cpu())
    {
        return cpu_grotto_eq_eval(out, maskedX, key, seed, partyId, bitWidthIn, elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_grotto_eq_eval(out, maskedX, key, seed, partyId, bitWidthIn, elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}

torch::Tensor& grotto_eval(torch::Tensor&       out,
                           const torch::Tensor& maskedX,
                           const torch::Tensor& key,
                           const torch::Tensor& seed,
                           bool                 equalBound,
                           int                  partyId,
                           std::size_t          bitWidthIn)
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
    if (bitWidthIn > 64)
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
    out.resize_(maskedX.sizes());

    auto device = out.device();
    TORCH_CHECK(maskedX.device() == device);
    TORCH_CHECK(key.device() == device);
    TORCH_CHECK(seed.device() == device);

    if (device.is_cpu())
    {
        return cpu_grotto_eval(out, maskedX, key, seed, equalBound, partyId, bitWidthIn, elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_grotto_eval(out, maskedX, key, seed, equalBound, partyId, bitWidthIn, elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}

torch::Tensor& grotto_mic_eval(torch::Tensor&       out,
                               const torch::Tensor& maskedX,
                               const torch::Tensor& key,
                               const torch::Tensor& seed,
                               int                  partyId,
                               const torch::Tensor& leftEndpoints,
                               const torch::Tensor& rightEndpoints,
                               std::size_t          bitWidthIn)
{
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(maskedX.is_contiguous());
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(seed.is_contiguous());
    TORCH_CHECK(leftEndpoints.is_contiguous());
    TORCH_CHECK(rightEndpoints.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
    TORCH_CHECK(seed.dtype() == torch::kUInt8);
    auto dtype = maskedX.dtype();
    TORCH_CHECK(out.dtype() == dtype);
    TORCH_CHECK(leftEndpoints.dtype() == dtype);
    TORCH_CHECK(rightEndpoints.dtype() == dtype);

    std::size_t elementSize = 0;
    std::size_t elementNum  = 0;
    if (bitWidthIn > 64)
    {
        TORCH_CHECK(dtype == torch::kInt64);
        auto limbNum = maskedX.sizes().back();
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
    TORCH_CHECK(seed.device() == device);
    TORCH_CHECK(leftEndpoints.device() == device);
    TORCH_CHECK(rightEndpoints.device() == device);

    std::size_t intervalNum = leftEndpoints.numel();
    if (bitWidthIn > 64)
    {
        intervalNum /= leftEndpoints.sizes().back();
    }

    auto                      out_sizes_raw = maskedX.sizes();
    std::vector<std::int64_t> out_sizes(out_sizes_raw.begin(), out_sizes_raw.end());
    out_sizes.push_back((std::int64_t)intervalNum);
    out.resize_(out_sizes);

    if (device.is_cpu())
    {
        return cpu_grotto_mic_eval(out, maskedX, key, seed, partyId, leftEndpoints, rightEndpoints, bitWidthIn,
                                   elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_grotto_mic_eval(out, maskedX, key, seed, partyId, leftEndpoints, rightEndpoints, bitWidthIn,
                                    elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}

std::tuple<torch::Tensor&, torch::Tensor&> grotto_interval_lut_eval(torch::Tensor&       outE,
                                                                    torch::Tensor&       outT,
                                                                    const torch::Tensor& maskedX,
                                                                    const torch::Tensor& key,
                                                                    const torch::Tensor& seed,
                                                                    int                  partyId,
                                                                    const torch::Tensor& leftEndpoints,
                                                                    const torch::Tensor& rightEndpoints,
                                                                    const torch::Tensor& lookUpTable,
                                                                    std::size_t          bitWidthIn,
                                                                    std::size_t          bitWidthOut)
{
    TORCH_CHECK(outE.is_contiguous());
    TORCH_CHECK(outT.is_contiguous());
    TORCH_CHECK(maskedX.is_contiguous());
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(seed.is_contiguous());
    TORCH_CHECK(leftEndpoints.is_contiguous());
    TORCH_CHECK(rightEndpoints.is_contiguous());
    TORCH_CHECK(lookUpTable.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
    TORCH_CHECK(seed.dtype() == torch::kUInt8);
    auto dtype = maskedX.dtype();
    TORCH_CHECK(outE.dtype() == dtype);
    TORCH_CHECK(leftEndpoints.dtype() == dtype);
    TORCH_CHECK(rightEndpoints.dtype() == dtype);
    TORCH_CHECK(outT.dtype() == dtype);
    TORCH_CHECK(lookUpTable.dtype() == dtype);

    std::size_t elementSize = 0;
    std::size_t elementNum  = 0;
    if (bitWidthIn > 64 || bitWidthOut > 64)
    {
        TORCH_CHECK(dtype == torch::kInt64);
        auto limbNum = maskedX.sizes().back();
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
    outE.resize_(maskedX.sizes());

    std::size_t lutNum = lookUpTable.numel() / leftEndpoints.numel();
    auto                      outT_sizes_raw = maskedX.sizes();
    std::vector<std::int64_t> outT_sizes(outT_sizes_raw.begin(), outT_sizes_raw.end());
    outT_sizes.push_back((std::int64_t)lutNum);
    outT.resize_(outT_sizes);

    auto device = outE.device();
    TORCH_CHECK(outT.device() == device);
    TORCH_CHECK(maskedX.device() == device);
    TORCH_CHECK(key.device() == device);
    TORCH_CHECK(seed.device() == device);
    TORCH_CHECK(leftEndpoints.device() == device);
    TORCH_CHECK(rightEndpoints.device() == device);
    TORCH_CHECK(lookUpTable.device() == device);

    if (device.is_cpu())
    {
        return cpu_grotto_interval_lut_eval(outE, outT, maskedX, key, seed, partyId, leftEndpoints, rightEndpoints,
                                            lookUpTable, bitWidthIn, bitWidthOut, elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_grotto_interval_lut_eval(outE, outT, maskedX, key, seed, partyId, leftEndpoints, rightEndpoints,
                                             lookUpTable, bitWidthIn, bitWidthOut, elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}
