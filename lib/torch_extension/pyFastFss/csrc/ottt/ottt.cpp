#include "ottt.h"

#include <FastFss/ottt.h>

#include "ottt_cpu.h"
#ifdef FAST_FSS_ENABLE_CUDA
#include "ottt_cuda.h"
#endif

std::size_t ottt_get_key_data_size(std::size_t bitWidthIn, std::size_t elementNum)
{
    std::size_t keyDataSize = 0;
    int         ret         = FastFss_otttGetKeyDataSize(&keyDataSize, bitWidthIn, elementNum);
    if (ret != 0)
    {
        std::string msg = "FastFss_otttGetKeyDataSize failed : " + std::to_string(ret);
        throw std::runtime_error(msg);
    }
    return keyDataSize;
}

torch::Tensor& ottt_key_gen(torch::Tensor& key, const torch::Tensor& alpha, std::size_t bitWidthIn)
{
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(alpha.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
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

    std::size_t keyDataSize = ottt_get_key_data_size(bitWidthIn, elementNum);
    key.resize_({(std::int64_t)keyDataSize});

    if (device.is_cpu())
    {
        return cpu_ottt_key_gen(key, alpha, bitWidthIn, elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_ottt_key_gen(key, alpha, bitWidthIn, elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}

std::tuple<torch::Tensor&, torch::Tensor&> ottt_lut_eval(torch::Tensor&       outE,
                                                         torch::Tensor&       outT,
                                                         const torch::Tensor& maskedX,
                                                         const torch::Tensor& key,
                                                         int                  partyId,
                                                         const torch::Tensor& lookUpTable,
                                                         std::size_t          bitWidthIn,
                                                         std::size_t          bitWidthOut)
{
    TORCH_CHECK(outE.is_contiguous());
    TORCH_CHECK(outT.is_contiguous());
    TORCH_CHECK(maskedX.is_contiguous());
    TORCH_CHECK(key.is_contiguous());
    TORCH_CHECK(lookUpTable.is_contiguous());

    TORCH_CHECK(key.dtype() == torch::kUInt8);
    auto dtype = maskedX.dtype();
    TORCH_CHECK(outE.dtype() == dtype);
    TORCH_CHECK(outT.dtype() == dtype);
    TORCH_CHECK(lookUpTable.dtype() == dtype);

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
    outE.resize_(maskedX.sizes());

    std::size_t lutNum = lookUpTable.numel() >> bitWidthIn;
    auto                      outT_sizes_raw = maskedX.sizes();
    std::vector<std::int64_t> outT_sizes(outT_sizes_raw.begin(), outT_sizes_raw.end());
    outT_sizes.push_back((std::int64_t)lutNum);
    outT.resize_(outT_sizes);

    auto device = outE.device();
    TORCH_CHECK(outT.device() == device);
    TORCH_CHECK(maskedX.device() == device);
    TORCH_CHECK(key.device() == device);
    TORCH_CHECK(lookUpTable.device() == device);

    if (device.is_cpu())
    {
        return cpu_ottt_lut_eval(outE, outT, maskedX, key, partyId, lookUpTable, bitWidthIn, elementSize, elementNum);
    }
#ifdef FAST_FSS_ENABLE_CUDA
    else if (device.is_cuda())
    {
        return cuda_ottt_lut_eval(outE, outT, maskedX, key, partyId, lookUpTable, bitWidthIn, elementSize, elementNum);
    }
#endif
    else
    {
        throw std::invalid_argument("device not supported");
    }
}
