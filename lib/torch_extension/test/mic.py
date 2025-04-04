import torch
import pyFastFss


def run_test(device: str, dtype: torch.dtype, bit_width_in: int, bit_width_out: int, element_num: int,
             left_boundary: list, right_boundary: list):
    # =======================================================
    # =======================================================
    # =======================================================

    left_boundary = torch.tensor(left_boundary, dtype=dtype, device=device)
    right_boundary = torch.tensor(right_boundary, dtype=dtype, device=device)

    alpha = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (element_num,), dtype=dtype, device=device)
    alpha = pyFastFss.mod(alpha, bit_width_in)
    seed0 = torch.randint(0, 255, (element_num * 16,), dtype=torch.uint8, device=device)
    seed1 = torch.randint(0, 255, (element_num * 16,), dtype=torch.uint8, device=device)

    # =======================================================
    # =======================================================
    # =======================================================

    key = torch.empty(0, dtype=torch.uint8, device=device)
    z = torch.empty(0, dtype=dtype, device=device)
    pyFastFss.dcf_mic_key_gen(
        key, z, alpha, seed0, seed1, left_boundary, right_boundary, bit_width_in, bit_width_out, element_num
    )

    shared_z0 = torch.randint_like(z, torch.iinfo(dtype).min, torch.iinfo(dtype).max)
    shared_z1 = z - shared_z0

    # =======================================================
    # =======================================================
    # =======================================================

    x = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (element_num,), dtype=dtype, device=device)
    x = pyFastFss.mod(x, bit_width_in)
    masked_x = pyFastFss.mod((x + alpha), bit_width_in)

    shared_out0 = torch.empty(0, dtype=dtype, device=device)
    pyFastFss.dcf_mic_eval(shared_out0, masked_x, key, shared_z0, seed0, 0,
                           left_boundary, right_boundary, bit_width_in, bit_width_out, element_num
                           )

    shared_out1 = torch.empty(0, dtype=dtype, device=device)
    pyFastFss.dcf_mic_eval(shared_out1, masked_x, key, shared_z1, seed1, 1,
                           left_boundary, right_boundary, bit_width_in, bit_width_out, element_num
                           )
    # =======================================================
    # =======================================================
    # =======================================================

    interval_num = left_boundary.numel()

    out = pyFastFss.mod((shared_out0 + shared_out1), bit_width_out)
    out = out.reshape(element_num, interval_num).to(torch.bool)

    need_out = torch.logical_and(
        torch.greater_equal(x.reshape(element_num, 1), left_boundary.reshape(1, interval_num)),
        torch.less_equal(x.reshape(element_num, 1), right_boundary.reshape(1, interval_num))
    )

    print(
        f"dtype={str(dtype):12s} device={device:6s} bit_width_in={bit_width_in:3d}, "
        f"bit_width_out={bit_width_out:3d}, element_num={element_num}:  ",
        "pass" if torch.equal(out, need_out) else "fail"
    )

    if not torch.equal(out, need_out):
        print(x)
        print(out)
        print(need_out)


run_test("cpu", torch.int8, 6, 6, 1024 - 1,
         [0] + [2 ** i for i in range(5)], [0] + [2 ** (i + 1) - 1 for i in range(5)]
         )
run_test("cuda", torch.int8, 6, 6, 1024 - 1,
         [0] + [2 ** i for i in range(5)], [0] + [2 ** (i + 1) - 1 for i in range(5)]
         )
run_test("cpu", torch.int16, 14, 11, 1024 - 1,
         [0] + [2 ** i for i in range(13)], [0] + [2 ** (i + 1) - 1 for i in range(13)]
         )
run_test("cuda", torch.int16, 14, 11, 1024 - 1,
         [0] + [2 ** i for i in range(13)], [0] + [2 ** (i + 1) - 1 for i in range(13)]
         )

run_test("cpu", torch.int32, 31, 11, 1024 - 1,
         [0] + [2 ** i for i in range(31)], [0] + [2 ** (i + 1) - 1 for i in range(31)]
         )
run_test("cuda", torch.int32, 31, 11, 1024 - 1,
         [0] + [2 ** i for i in range(31)], [0] + [2 ** (i + 1) - 1 for i in range(31)]
         )
run_test("cpu", torch.int64, 61, 11, 1024 - 1,
         [0] + [2 ** i for i in range(60)], [0] + [2 ** (i + 1) - 1 for i in range(60)]
         )
run_test("cuda", torch.int64, 61, 11, 1024 - 1,
         [0] + [2 ** i for i in range(60)], [0] + [2 ** (i + 1) - 1 for i in range(60)]
         )
