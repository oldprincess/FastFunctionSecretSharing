import torch
import pyFastFss


def run_test(device: str, dtype: torch.dtype, bit_width_in: int, bit_width_out: int, element_num: int):
    # =======================================================
    # =======================================================
    # =======================================================

    alpha = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (element_num,), dtype=dtype, device=device)
    alpha = pyFastFss.mod(alpha, bit_width_in)
    beta = torch.empty(0, dtype=dtype, device=device)
    seed0 = torch.randint(0, 255, (element_num * 16,), dtype=torch.uint8, device=device)
    seed1 = torch.randint(0, 255, (element_num * 16,), dtype=torch.uint8, device=device)

    # =======================================================
    # =======================================================
    # =======================================================

    key = torch.empty(0, dtype=torch.uint8, device=device)
    pyFastFss.dcf_key_gen(key, alpha, beta, seed0, seed1, bit_width_in, bit_width_out, element_num)

    # =======================================================
    # =======================================================
    # =======================================================

    x = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (element_num,), dtype=dtype, device=device)
    masked_x = pyFastFss.mod((x + alpha), bit_width_in)

    shared_out0 = torch.empty(0, dtype=dtype, device=device)
    pyFastFss.dcf_eval(shared_out0, masked_x, key, seed0, 0, bit_width_in, bit_width_out, element_num)

    shared_out1 = torch.empty(0, dtype=dtype, device=device)
    pyFastFss.dcf_eval(shared_out1, masked_x, key, seed1, 1, bit_width_in, bit_width_out, element_num)
    # =======================================================
    # =======================================================
    # =======================================================

    out = pyFastFss.mod((shared_out0 + shared_out1), bit_width_out)

    print(
        f"dtype={str(dtype):12s} device={device:6s} bit_width_in={bit_width_in:3d}, "
        f"bit_width_out={bit_width_out:3d}, element_num={element_num}:",
        "pass" if (torch.equal(out.to(torch.bool), torch.less(masked_x, alpha))) else "fail"
    )


run_test("cpu", torch.int8, 6, 6, 1024 - 1)
run_test("cuda", torch.int8, 6, 6, 1024 - 1)
run_test("cpu", torch.int16, 15, 8, 1024 - 1)
run_test("cuda", torch.int16, 15, 8, 1024 - 1)
run_test("cpu", torch.int32, 31, 25, 1024 - 1)
run_test("cuda", torch.int32, 31, 25, (1024 - 1))
run_test("cpu", torch.int64, 62, 25, 1024 - 1)
run_test("cuda", torch.int64, 62, 25, 1024 - 1)
