import torch
from pyFastFss._C import *


def _wide_limb_num(bit_width: int) -> int:
    return 1 if bit_width <= 64 else (bit_width + 63) // 64


def mod(x: torch.Tensor, bit_width: int, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device
    if bit_width <= 64:
        assert x.element_size() * 8 >= bit_width
        if x.element_size() * 8 == bit_width:
            return out.copy_(x)
        torch.bitwise_and(x, (1 << bit_width) - 1, out=out)
        return out

    if x.dtype != torch.int64:
        raise ValueError("wideint tensors must use torch.int64 dtype")
    if x.ndim < 1:
        raise ValueError("wideint tensors must have a limb dimension")

    limb_num = _wide_limb_num(bit_width)
    if x.shape[-1] != limb_num:
        raise ValueError("wideint tensor last dimension must equal limb count")

    out.copy_(x)
    valid_high_bits = bit_width - 64 * (limb_num - 1)
    if valid_high_bits < 64:
        out[..., -1].bitwise_and_((1 << valid_high_bits) - 1)

    return out


def dtype(bit_width: int) -> torch.dtype:
    assert 0 < bit_width
    if 1 <= bit_width <= 8:
        return torch.int8
    elif 9 <= bit_width <= 16:
        return torch.int16
    elif 17 <= bit_width <= 32:
        return torch.int32
    else:
        return torch.int64


def logical_shape(x: torch.Tensor, bit_width: int) -> torch.Size:
    if bit_width <= 64:
        return x.shape
    if x.dtype != torch.int64:
        raise ValueError("wideint tensors must use torch.int64 dtype")
    limb_num = _wide_limb_num(bit_width)
    if x.ndim < 1 or x.shape[-1] != limb_num:
        raise ValueError("wideint tensor last dimension must equal limb count")
    return x.shape[:-1]


def logical_numel(x: torch.Tensor, bit_width: int) -> int:
    shape = logical_shape(x, bit_width)
    if len(shape) == 0:
        return 1
    result = 1
    for dim in shape:
        result *= dim
    return result
