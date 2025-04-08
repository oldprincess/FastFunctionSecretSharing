import math
import torch
import pyFastFss
import time


def grotto_approx_exp_key_gen(alpha: torch.Tensor,
                              bit_width_in: int,
                              bit_width_out: int,
                              float_bits_in: int,
                              float_bits_out: int):
    key0 = {}
    key1 = {}

    dtype = pyFastFss.dtype(max(bit_width_in, bit_width_out))
    device = alpha.device
    alpha = alpha.to(dtype)

    seed0 = torch.randint(0, 255, (alpha.numel() * 16,), device=device, dtype=torch.uint8)
    seed1 = torch.randint(0, 255, (alpha.numel() * 16,), device=device, dtype=torch.uint8)

    pyFastFss.mod(alpha, bit_width_in, out=alpha)

    key = torch.empty(0, dtype=torch.uint8, device=device)
    pyFastFss.grotto_key_gen(key, alpha, seed0, seed1, bit_width_in, alpha.numel())

    key0["exp"] = {"key": key, "seed": seed0}
    key1["exp"] = {"key": key, "seed": seed1}

    return key0, key1


def test_grotto_approx_exp_eval(masked_x: torch.Tensor,
                                key0: dict,
                                key1: dict,
                                bit_width_in: int,
                                bit_width_out: int,
                                float_bits_in: int,
                                float_bits_out: int):
    time_st = time.time_ns()

    dtype = pyFastFss.dtype(max(bit_width_in, bit_width_out))
    device = masked_x.device
    masked_x = masked_x.to(dtype)
    left_boundary = [math.log(i / (2 ** float_bits_in)) for i in range(1, 2 ** float_bits_in)]
    right_boundary = [math.log(i / (2 ** float_bits_in)) for i in range(1, 2 ** float_bits_in)]
    left_boundary = [torch.iinfo(dtype).min] + [round(item * pow(2, float_bits_in)) for item in left_boundary]
    right_boundary = [round(item * pow(2, float_bits_in)) - 1 for item in right_boundary] + [0]
    left_boundary = torch.tensor(left_boundary, dtype=dtype, device=device)
    right_boundary = torch.tensor(right_boundary, dtype=dtype, device=device)
    lut = torch.arange(0, 2 ** float_bits_in, dtype=dtype, device=device)

    pyFastFss.mod(masked_x, bit_width_in, out=masked_x)
    pyFastFss.mod(left_boundary, bit_width_in, out=left_boundary)
    pyFastFss.mod(right_boundary, bit_width_in, out=right_boundary)

    time_et = time.time_ns()
    print("exp init", (time_et - time_st) / 1e6, "ms")

    time_st = time.time_ns()

    shared_e0 = torch.empty(0, device=device, dtype=dtype)
    shared_t0 = torch.empty(0, device=device, dtype=dtype)
    pyFastFss.grotto_interval_lut_eval(shared_e0, shared_t0, masked_x, key0["exp"]["key"], key0["exp"]["seed"],
                                       0, left_boundary, right_boundary, lut, bit_width_in, bit_width_out,
                                       masked_x.numel())
    time_et = time.time_ns()
    print("exp eval 0", (time_et - time_st) / 1e6, "ms")

    time_st = time.time_ns()

    shared_e1 = torch.empty(0, device=device, dtype=dtype)
    shared_t1 = torch.empty(0, device=device, dtype=dtype)
    pyFastFss.grotto_interval_lut_eval(shared_e1, shared_t1, masked_x, key1["exp"]["key"], key1["exp"]["seed"],
                                       1, left_boundary, right_boundary, lut, bit_width_in, bit_width_out,
                                       masked_x.numel())
    time_et = time.time_ns()
    print("exp eval 1", (time_et - time_st) / 1e6, "ms")

    return pyFastFss.mod((shared_e0 + shared_e1) * (shared_t0 + shared_t1), bit_width_out).reshape(masked_x.shape)


def _main():
    shape = (12 * 128, 128,)
    bit_width_in = 22
    bit_width_out = 22
    float_bits_in = 10
    float_bits_out = 10

    alpha = torch.randint(0, pow(2, bit_width_in) - 1, shape)

    x = -torch.randint(0, pow(2, float_bits_in + 4) - 1, shape)

    print("------------------------")
    print("shape", shape)
    print("bit_width_in", bit_width_in, "bit_width_out", bit_width_out)

    print("------------------- cpu -----------------------")
    key0, key1 = grotto_approx_exp_key_gen(alpha, bit_width_in, bit_width_out, float_bits_in, float_bits_out)

    ret = test_grotto_approx_exp_eval(x + alpha, key0, key1, bit_width_in, bit_width_out, float_bits_in, float_bits_out)

    print(f"diff {shape}: ", torch.linalg.norm(torch.exp(x / (2 ** float_bits_in)) - ret / (2 ** float_bits_out)))

    print("------------------- cuda -----------------------")
    alpha = alpha.to("cuda")
    x = x.to("cuda")
    
    key0, key1 = grotto_approx_exp_key_gen(alpha, bit_width_in, bit_width_out, float_bits_in, float_bits_out)
    
    test_grotto_approx_exp_eval(x + alpha, key0, key1, bit_width_in, bit_width_out, float_bits_in, float_bits_out)
    


_main()
