import torch
from pyFastFss._C import *


def mod(x: torch.Tensor, bit_width: int, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    else:
        assert x.element_size() * 8 >= bit_width
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device
    if x.element_size() * 8 == bit_width:
        return out.copy_(x)
    torch.bitwise_and(x, (1 << bit_width) - 1, out=out)
    return out


def dtype(bit_width: int) -> torch.dtype:
    assert 0 < bit_width <= 64
    if 1 <= bit_width <= 8:
        return torch.int8
    elif 9 <= bit_width <= 16:
        return torch.int16
    elif 17 <= bit_width <= 32:
        return torch.int32
    else:
        return torch.int64
