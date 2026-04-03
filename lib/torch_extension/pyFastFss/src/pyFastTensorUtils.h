#ifndef PY_FAST_TENSOR_UTILS_H
#define PY_FAST_TENSOR_UTILS_H

#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyFastFss {

struct ValueTensorLayout
{
    std::vector<std::int64_t> logicalShape;
    std::size_t               logicalElementNum;
    std::size_t               limbNum;
    std::size_t               elementSize;
    bool                      wide;
};

inline std::size_t wide_limb_num(std::size_t bitWidth)
{
    return bitWidth <= 64 ? 1 : (bitWidth + 63) / 64;
}

inline bool is_wide_bit_width(std::size_t bitWidth)
{
    return bitWidth > 64;
}

inline std::size_t max_bit_width(std::initializer_list<std::size_t> bitWidths)
{
    std::size_t result = 0;
    for (auto bitWidth : bitWidths)
    {
        if (bitWidth > result)
        {
            result = bitWidth;
        }
    }
    return result;
}

inline std::size_t shape_numel(const std::vector<std::int64_t> &shape)
{
    if (shape.empty())
    {
        return 1;
    }

    std::size_t result = 1;
    for (auto dim : shape)
    {
        if (dim < 0)
        {
            throw std::invalid_argument("shape dimension must be non-negative");
        }
        result *= static_cast<std::size_t>(dim);
    }
    return result;
}

inline ValueTensorLayout inspect_value_tensor(const torch::Tensor &tensor, std::size_t bitWidth)
{
    ValueTensorLayout layout{};
    layout.wide = is_wide_bit_width(bitWidth);

    if (!layout.wide)
    {
        layout.logicalShape.reserve(tensor.dim());
        for (auto dim : tensor.sizes())
        {
            layout.logicalShape.push_back(dim);
        }
        layout.logicalElementNum = static_cast<std::size_t>(tensor.numel());
        layout.limbNum           = 1;
        layout.elementSize       = tensor.element_size();
        return layout;
    }

    if (tensor.dtype() != torch::kInt64)
    {
        throw std::invalid_argument("wideint tensors must use torch.int64 dtype");
    }
    if (tensor.dim() < 1)
    {
        throw std::invalid_argument("wideint tensors must have a limb dimension");
    }

    layout.limbNum = wide_limb_num(bitWidth);
    if (static_cast<std::size_t>(tensor.size(-1)) != layout.limbNum)
    {
        throw std::invalid_argument("wideint tensor last dimension must equal limb count");
    }

    layout.logicalShape.reserve(tensor.dim() - 1);
    for (std::int64_t i = 0; i < tensor.dim() - 1; ++i)
    {
        layout.logicalShape.push_back(tensor.size(i));
    }

    layout.logicalElementNum = shape_numel(layout.logicalShape);
    layout.elementSize       = layout.limbNum * sizeof(std::int64_t);
    return layout;
}

inline void assert_same_logical_shape(const ValueTensorLayout &lhs,
                                      const ValueTensorLayout &rhs,
                                      const std::string       &message)
{
    if (lhs.logicalShape != rhs.logicalShape)
    {
        throw std::invalid_argument(message);
    }
}

inline std::vector<std::int64_t> make_value_shape(const std::vector<std::int64_t> &logicalShape, std::size_t bitWidth)
{
    std::vector<std::int64_t> result = logicalShape;
    if (is_wide_bit_width(bitWidth))
    {
        result.push_back(static_cast<std::int64_t>(wide_limb_num(bitWidth)));
    }
    return result;
}

inline std::vector<std::int64_t> append_logical_dim(const std::vector<std::int64_t> &logicalShape, std::int64_t dim)
{
    std::vector<std::int64_t> result = logicalShape;
    result.push_back(dim);
    return result;
}

} // namespace pyFastFss

#endif
