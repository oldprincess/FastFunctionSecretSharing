import functools
import pyFastFss
import torch
import time


def test(bit_width: int, lut_bit_width: int, dtype: torch.dtype, device: torch.device):
    shape = (12, 128, 128)
    info = torch.iinfo(dtype)
    bit_width_out = dtype.itemsize * 8

    element_num = functools.reduce(lambda _x, _y: _x * _y, shape)
    # key generate
    st = time.time_ns()
    alpha = pyFastFss.mod(torch.randint(info.min, info.max, shape, device=device, dtype=dtype), bit_width)
    seed0 = torch.randint(0, 256, (element_num * 16,), device=device, dtype=torch.uint8)
    seed1 = torch.randint(0, 256, (element_num * 16,), device=device, dtype=torch.uint8)
    key = torch.empty(0, dtype=torch.uint8, device=device)
    key = pyFastFss.grotto_key_gen(key, alpha, seed0, seed1, bit_width, element_num)
    torch.cuda.synchronize()
    et = time.time_ns()
    print("\tgrotto_key_gen", (et - st) / 1e6, "ms")
    # eval
    x = pyFastFss.mod(torch.randint(info.min, info.max, shape, device=device, dtype=dtype), lut_bit_width)
    masked_x = pyFastFss.mod(x + alpha, bit_width)
    shared_out_e0 = torch.empty(0, dtype=dtype, device=device)
    shared_out_e1 = torch.empty(0, dtype=dtype, device=device)
    shared_out_t0 = torch.empty(0, dtype=dtype, device=device)
    shared_out_t1 = torch.empty(0, dtype=dtype, device=device)
    luts = torch.arange(0, 2 ** lut_bit_width, dtype=dtype, device=device)

    st = time.time_ns()
    shared_out_e0, shared_out_t0 = pyFastFss.grotto_lut_eval_ex(
        shared_out_e0, shared_out_t0, masked_x, key, seed0, 0,
        luts, lut_bit_width, bit_width, bit_width_out, element_num
    )
    torch.cuda.synchronize()
    et = time.time_ns()
    print("\tgrotto_lut_eval_ex(0)", (et - st) / 1e6, "ms")
    st = time.time_ns()
    shared_out_e1, shared_out_t1 = pyFastFss.grotto_lut_eval_ex(
        shared_out_e1, shared_out_t1, masked_x, key, seed1, 1,
        luts, lut_bit_width, bit_width, bit_width_out, element_num
    )
    torch.cuda.synchronize()
    et = time.time_ns()
    print("\tgrotto_lut_eval_ex(1)", (et - st) / 1e6, "ms")

    e = (shared_out_e0 + shared_out_e1) & 1
    t = pyFastFss.mod(shared_out_t0 + shared_out_t1, bit_width_out)
    out = pyFastFss.mod(t + e * t * (-2), bit_width_out).reshape(x.shape)
    need = luts[pyFastFss.mod(x.to(torch.int64), bit_width)]

    print("[TEST] grotto_lut_eval_ex:",
          "bitWidth={:>3d} device={:4}".format(bit_width, str(device)),
          "Pass" if torch.all(out == need) else "Fail")


dtype_lst = [torch.int8, torch.int16, torch.int32, torch.int64]
for i in range(len(dtype_lst)):
    test(dtype_lst[i].itemsize * 8, 7, dtype_lst[i], torch.device("cpu"))
for i in range(len(dtype_lst)):
    test(dtype_lst[i].itemsize * 8, 7, dtype_lst[i], torch.device("cuda"))
