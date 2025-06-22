import functools
import pyFastFss
import torch


def test(bit_width: int, device: torch.device):
    shape = (64, 512)
    dtype = pyFastFss.dtype(bit_width)
    info = torch.iinfo(dtype)

    element_num = functools.reduce(lambda _x, _y: _x * _y, shape)
    # key generate
    alpha = pyFastFss.mod(torch.randint(info.min, info.max, shape, device=device, dtype=dtype), bit_width)
    seed0 = torch.randint(0, 256, (element_num * 16,), device=device, dtype=torch.uint8)
    seed1 = torch.randint(0, 256, (element_num * 16,), device=device, dtype=torch.uint8)
    key = torch.empty(0, dtype=torch.uint8, device=device)
    key = pyFastFss.dpf_key_gen(
        key, alpha, torch.empty(0, device=device, dtype=dtype), seed0, seed1, bit_width, bit_width, element_num
    )
    # eval
    x = pyFastFss.mod(torch.randint(info.min, info.max, shape, device=device, dtype=dtype), bit_width)
    masked_x = pyFastFss.mod(x + alpha, bit_width)
    shared_out0 = torch.empty(0, dtype=dtype, device=device)
    shared_out1 = torch.empty(0, dtype=dtype, device=device)
    points = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=dtype, device=device)

    shared_out0 = pyFastFss.dpf_eval_multi(
        shared_out0, masked_x, key, seed0, 0, points, bit_width, bit_width, element_num
    )
    shared_out1 = pyFastFss.dpf_eval_multi(
        shared_out1, masked_x, key, seed1, 1, points, bit_width, bit_width, element_num
    )

    out = pyFastFss.mod(shared_out0 + shared_out1, bit_width).reshape(*x.shape, points.numel()).bool()
    need = torch.eq(x.reshape(*x.shape, 1), points.reshape(1, points.numel()))

    print("[TEST] grotto_eval_eq_multi:",
          "bitWidth={:>3d} device={:4}".format(bit_width, str(device)),
          "Pass" if torch.all(out == need) else "Fail")


bit_width_lst = [8, 10, 16, 18, 32, 34, 62, 64]
for i in range(len(bit_width_lst)):
    test(bit_width_lst[i], torch.device("cpu"))
for i in range(len(bit_width_lst)):
    test(bit_width_lst[i], torch.device("cuda"))
