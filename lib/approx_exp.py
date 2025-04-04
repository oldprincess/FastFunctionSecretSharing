import math
import torch
import pyFastFss
import time


def dcf_approx_exp_key_gen(alpha: torch.Tensor,
                           bit_width_in: int,
                           bit_width_out: int,
                           float_bits_in: int,
                           float_bits_out: int):
    key0 = {}
    key1 = {}

    dtype = pyFastFss.dtype(max(bit_width_in, bit_width_out))
    device = alpha.device
    alpha = alpha.to(dtype)
    left_boundary = [
        round(math.log(i / (2 ** float_bits_out), math.e) * (2 ** float_bits_in))
        for i in range(1, 2 ** float_bits_out - 1)
    ]
    right_boundary = [
        round(math.log((i + 1) / (2 ** float_bits_out), math.e) * (2 ** float_bits_in)) - 1
        for i in range(1, 2 ** float_bits_out - 1)
    ]
    left_boundary = torch.tensor(left_boundary, dtype=dtype, device=device)
    right_boundary = torch.tensor(right_boundary, dtype=dtype, device=device)
    seed0 = torch.randint(0, 255, (alpha.numel() * 16,), device=device, dtype=torch.uint8)
    seed1 = torch.randint(0, 255, (alpha.numel() * 16,), device=device, dtype=torch.uint8)

    pyFastFss.mod(alpha, bit_width_in, out=alpha)
    pyFastFss.mod(left_boundary, bit_width_in, out=left_boundary)
    pyFastFss.mod(right_boundary, bit_width_in, out=right_boundary)

    key = torch.empty(0, dtype=torch.uint8, device=device)
    z = torch.empty(0, dtype=dtype, device=device)
    pyFastFss.dcf_mic_key_gen(key, z,
                              alpha, seed0, seed1,
                              left_boundary, right_boundary,
                              bit_width_in, bit_width_out, alpha.numel())
    shared_z0 = torch.randint_like(z, torch.iinfo(dtype).min, torch.iinfo(dtype).max)
    shared_z1 = z - shared_z0

    shared_z0 = shared_z0.reshape(*alpha.shape, left_boundary.shape[0])
    shared_z1 = shared_z1.reshape(*alpha.shape, left_boundary.shape[0])

    key0["exp"] = {"key": key, "shared_z": shared_z0, "seed": seed0}
    key1["exp"] = {"key": key, "shared_z": shared_z1, "seed": seed1}

    print(key.numel(), z.numel())

    return key0, key1


def test_dcf_approx_exp_eval(masked_x: torch.Tensor,
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
    left_boundary = [
        round(math.log(i / (2 ** float_bits_out), math.e) * (2 ** float_bits_in))
        for i in range(1, 2 ** float_bits_out - 1)
    ]
    right_boundary = [
        round(math.log((i + 1) / (2 ** float_bits_out), math.e) * (2 ** float_bits_in)) - 1
        for i in range(1, 2 ** float_bits_out - 1)
    ]
    left_boundary = torch.tensor(left_boundary, dtype=dtype, device=device)
    right_boundary = torch.tensor(right_boundary, dtype=dtype, device=device)

    pyFastFss.mod(masked_x, bit_width_in, out=masked_x)
    pyFastFss.mod(left_boundary, bit_width_in, out=left_boundary)
    pyFastFss.mod(right_boundary, bit_width_in, out=right_boundary)

    time_et = time.time_ns()
    print("exp init", (time_et - time_st) / 1e6, "ms")

    time_st = time.time_ns()

    shared_out0 = torch.empty(0, device=device, dtype=dtype)
    pyFastFss.dcf_mic_eval(shared_out0, masked_x, key0["exp"]["key"], key0["exp"]["shared_z"], key0["exp"]["seed"],
                           0, left_boundary, right_boundary, bit_width_in, bit_width_out, masked_x.numel())

    time_et = time.time_ns()
    print("exp eval 0", (time_et - time_st) / 1e6, "ms")

    time_st = time.time_ns()

    shared_out1 = torch.empty(0, device=device, dtype=dtype)
    pyFastFss.dcf_mic_eval(shared_out1, masked_x, key1["exp"]["key"], key1["exp"]["shared_z"], key1["exp"]["seed"],
                           1, left_boundary, right_boundary, bit_width_in, bit_width_out, masked_x.numel())

    time_et = time.time_ns()
    print("exp eval 1", (time_et - time_st) / 1e6, "ms")


def _main():
    shape = (12 * 128, 128, )
    bit_width_in = 20
    bit_width_out = 20
    float_bits_in = 10
    float_bits_out = 10

    alpha = torch.randint(0, pow(2, bit_width_in) - 1, shape)

    x = torch.randint(0, pow(2, bit_width_in) - 1, shape)

    print("------------------------")
    print("shape", shape)
    print("bit_width_in", bit_width_in, "bit_width_out", bit_width_out)

    # print("------------------- cpu -----------------------")
    # key0, key1 = dcf_approx_exp_key_gen(alpha, bit_width_in, bit_width_out, float_bits_in, float_bits_out)
    #
    # test_dcf_approx_exp_eval(x + alpha, key0, key1, bit_width_in, bit_width_out, float_bits_in, float_bits_out)

    print("------------------- cuda -----------------------")
    alpha = alpha.to("cuda")
    x = x.to("cuda")

    key0, key1 = dcf_approx_exp_key_gen(alpha, bit_width_in, bit_width_out, float_bits_in, float_bits_out)

    test_dcf_approx_exp_eval(x + alpha, key0, key1, bit_width_in, bit_width_out, float_bits_in, float_bits_out)


_main()
