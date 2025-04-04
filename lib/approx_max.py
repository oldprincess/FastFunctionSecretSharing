import math
import torch
import pyFastFss
import time


def dcf_approx_max_key_gen(alpha: torch.Tensor,
                           bit_width_in: int,
                           bit_width_out: int,
                           bit_width_trunc: int,
                           left_boundary: torch.Tensor,
                           right_boundary: torch.Tensor):
    assert bit_width_trunc < bit_width_in
    assert left_boundary.shape == right_boundary.shape and len(left_boundary.shape) == 1

    batch_dims = alpha.shape[:-1]
    n = alpha.shape[-1]
    m = left_boundary.shape[0]

    key0 = {}
    key1 = {}

    # ==========================================================================
    # ======================= Step1: ars + onehot ==============================
    # ==========================================================================

    onehot_bit_width_in = bit_width_in - bit_width_trunc
    onehot_bit_width_out = math.ceil(math.log2(n + 1))
    onehot_dtype = pyFastFss.dtype(max(onehot_bit_width_in, onehot_bit_width_out))
    onehot_device = alpha.device
    onehot_r_in = torch.bitwise_right_shift(alpha, bit_width_trunc).to(onehot_dtype)
    onehot_left_boundary = torch.bitwise_right_shift(left_boundary, bit_width_trunc).to(onehot_dtype)
    onehot_right_boundary = torch.bitwise_right_shift(right_boundary, bit_width_trunc).to(onehot_dtype)
    onehot_seed0 = torch.randint(0, 255, (onehot_r_in.numel() * 16,), device=onehot_device, dtype=torch.uint8)
    onehot_seed1 = torch.randint(0, 255, (onehot_r_in.numel() * 16,), device=onehot_device, dtype=torch.uint8)

    pyFastFss.mod(onehot_r_in, bit_width_in - bit_width_trunc, out=onehot_r_in)
    pyFastFss.mod(onehot_left_boundary, bit_width_in - bit_width_trunc, out=onehot_left_boundary)
    pyFastFss.mod(onehot_right_boundary, bit_width_in - bit_width_trunc, out=onehot_right_boundary)

    onehot_key = torch.empty(0, dtype=torch.uint8, device=onehot_device)
    onehot_z = torch.empty(0, dtype=onehot_dtype, device=onehot_device)
    pyFastFss.dcf_mic_key_gen(onehot_key, onehot_z,
                              onehot_r_in, onehot_seed0, onehot_seed1,
                              onehot_left_boundary, onehot_right_boundary,
                              onehot_bit_width_in, onehot_bit_width_out, onehot_r_in.numel())
    onehot_shared_z0 = torch.randint_like(onehot_z, torch.iinfo(onehot_dtype).min, torch.iinfo(onehot_dtype).max)
    onehot_shared_z1 = onehot_z - onehot_shared_z0

    onehot_shared_z0 = onehot_shared_z0.reshape(*batch_dims, n, m)
    onehot_shared_z1 = onehot_shared_z1.reshape(*batch_dims, n, m)

    key0["onehot"] = {"key": onehot_key, "shared_z": onehot_shared_z0, "seed": onehot_seed0}
    key1["onehot"] = {"key": onehot_key, "shared_z": onehot_shared_z1, "seed": onehot_seed1}

    # ==========================================================================
    # ======================= Step2: bucket ====================================
    # ==========================================================================

    bucket_bit_width_in = onehot_bit_width_out
    bucket_bit_width_out = bucket_bit_width_in
    bucket_dtype = onehot_dtype
    bucket_device = onehot_device
    bucket_r_in0 = torch.randint(torch.iinfo(bucket_dtype).min, torch.iinfo(bucket_dtype).max, (*batch_dims, m),
                                 device=bucket_device, dtype=bucket_dtype)
    bucket_r_in1 = torch.randint(torch.iinfo(bucket_dtype).min, torch.iinfo(bucket_dtype).max, (*batch_dims, m),
                                 device=bucket_device, dtype=bucket_dtype)
    bucket_r_in = bucket_r_in0 + bucket_r_in1

    key0["bucket"] = {"bucket_r_in": bucket_r_in0}
    key1["bucket"] = {"bucket_r_in": bucket_r_in1}

    # ==========================================================================
    # ======================= Step3: greater0 ==================================
    # ==========================================================================

    gt0_bit_width_in = bucket_bit_width_out
    gt0_bit_width_out = m
    gt0_dtype = pyFastFss.dtype(max(gt0_bit_width_in, gt0_bit_width_out))
    gt0_device = bucket_device
    gt0_r_in = bucket_r_in.to(gt0_dtype)
    gt0_left_boundary = torch.tensor([1], dtype=gt0_dtype, device=gt0_device)
    gt0_right_boundary = torch.tensor([-1], dtype=gt0_dtype, device=gt0_device)

    pyFastFss.mod(gt0_r_in, gt0_bit_width_in, out=gt0_r_in)
    pyFastFss.mod(gt0_left_boundary, gt0_bit_width_in, out=gt0_left_boundary)
    pyFastFss.mod(gt0_right_boundary, gt0_bit_width_in, out=gt0_right_boundary)

    gt0_seed0 = torch.randint(0, 255, (gt0_r_in.numel() * 16,), device=gt0_device, dtype=torch.uint8)
    gt0_seed1 = torch.randint(0, 255, (gt0_r_in.numel() * 16,), device=gt0_device, dtype=torch.uint8)

    gt0_key = torch.empty(0, dtype=torch.uint8, device=gt0_device)
    gt0_z = torch.empty(0, dtype=gt0_dtype, device=gt0_device)
    pyFastFss.dcf_mic_key_gen(gt0_key, gt0_z, gt0_r_in, gt0_seed0, gt0_seed1,
                              gt0_left_boundary, gt0_right_boundary,
                              gt0_bit_width_in, gt0_bit_width_out, gt0_r_in.numel())

    gt0_shared_z0 = torch.randint_like(gt0_z, torch.iinfo(gt0_dtype).min, torch.iinfo(gt0_dtype).max)
    gt0_shared_z1 = gt0_z - gt0_shared_z0

    gt0_shared_z0 = gt0_shared_z0.reshape(*batch_dims, m)
    gt0_shared_z1 = gt0_shared_z1.reshape(*batch_dims, m)

    key0["gt0"] = {"key": gt0_key, "shared_z": gt0_shared_z0, "seed": gt0_seed0}
    key1["gt0"] = {"key": gt0_key, "shared_z": gt0_shared_z1, "seed": gt0_seed1}

    # ==========================================================================
    # ========================== Step4: y ======================================
    # ==========================================================================

    y_bit_width_in = gt0_bit_width_out
    y_bit_width_out = y_bit_width_in
    y_dtype = gt0_dtype
    y_device = gt0_device
    y_r_in0 = torch.randint(torch.iinfo(y_dtype).min, torch.iinfo(y_dtype).max,
                            (*batch_dims,), device=y_device, dtype=y_dtype)
    y_r_in1 = torch.randint(torch.iinfo(y_dtype).min, torch.iinfo(y_dtype).max,
                            (*batch_dims,), device=y_device, dtype=y_dtype)
    y_r_in = y_r_in0 + y_r_in1

    key0["y"] = {"y_r_in": y_r_in0}
    key1["y"] = {"y_r_in": y_r_in1}

    # ==========================================================================
    # ==================== Step5: interval lut =================================
    # ==========================================================================

    interval_lut_bit_width_in = y_bit_width_out
    interval_lut_bit_width_out = bit_width_out
    interval_lut_dtype = pyFastFss.dtype(max(interval_lut_bit_width_in, interval_lut_bit_width_out))
    interval_lut_device = y_device
    interval_lut_r_in = y_r_in.to(interval_lut_dtype)
    interval_lut_left_boundary = torch.tensor([2 ** i for i in range(m)],
                                              dtype=interval_lut_dtype,
                                              device=interval_lut_device)
    interval_lut_right_boundary = torch.tensor([2 ** (i + 1) - 1 for i in range(m)],
                                               dtype=interval_lut_dtype,
                                               device=interval_lut_device)

    pyFastFss.mod(interval_lut_r_in, interval_lut_bit_width_in, out=interval_lut_r_in)
    pyFastFss.mod(interval_lut_left_boundary, interval_lut_bit_width_in, out=interval_lut_left_boundary)
    pyFastFss.mod(interval_lut_right_boundary, interval_lut_bit_width_in, out=interval_lut_right_boundary)

    interval_lut_seed0 = torch.randint(0, 255, (interval_lut_r_in.numel() * 16,),
                                       device=interval_lut_device, dtype=torch.uint8)
    interval_lut_seed1 = torch.randint(0, 255, (interval_lut_r_in.numel() * 16,),
                                       device=interval_lut_device, dtype=torch.uint8)

    interval_lut_key = torch.empty(0, dtype=torch.uint8, device=interval_lut_device)
    interval_lut_z = torch.empty(0, dtype=interval_lut_dtype, device=interval_lut_device)
    pyFastFss.dcf_mic_key_gen(interval_lut_key, interval_lut_z, interval_lut_r_in,
                              interval_lut_seed0, interval_lut_seed1,
                              interval_lut_left_boundary, interval_lut_right_boundary,
                              interval_lut_bit_width_in, interval_lut_bit_width_out, interval_lut_r_in.numel())

    interval_lut_shared_z0 = torch.randint_like(interval_lut_z,
                                                torch.iinfo(interval_lut_dtype).min,
                                                torch.iinfo(interval_lut_dtype).max)
    interval_lut_shared_z1 = interval_lut_z - interval_lut_shared_z0

    interval_lut_shared_z0 = interval_lut_shared_z0.reshape(*batch_dims, m)
    interval_lut_shared_z1 = interval_lut_shared_z1.reshape(*batch_dims, m)

    key0["interval_lut"] = {"key": interval_lut_key, "shared_z": interval_lut_shared_z0, "seed": interval_lut_seed0}
    key1["interval_lut"] = {"key": interval_lut_key, "shared_z": interval_lut_shared_z1, "seed": interval_lut_seed1}

    return key0, key1


def test_dcf_approx_max_eval(masked_x: torch.Tensor, key0: dict, key1: dict,
                             bit_width_in: int, bit_width_out: int, bit_width_trunc: int,
                             left_boundary: torch.Tensor, right_boundary: torch.Tensor) -> torch.Tensor:
    assert bit_width_trunc < bit_width_in
    assert left_boundary.shape == right_boundary.shape and len(left_boundary.shape) == 1

    batch_dims = masked_x.shape[:-1]
    n = masked_x.shape[-1]
    m = left_boundary.shape[0]

    # ==========================================================================
    # ======================= Step1: ars + onehot ==============================
    # ==========================================================================

    time_st = time.time_ns()

    onehot_bit_width_in = bit_width_in - bit_width_trunc
    onehot_bit_width_out = math.ceil(math.log2(n + 1))
    onehot_dtype = pyFastFss.dtype(max(onehot_bit_width_in, onehot_bit_width_out))
    onehot_device = masked_x.device
    onehot_masked_in = torch.bitwise_right_shift(masked_x, bit_width_trunc).to(onehot_dtype)
    onehot_left_boundary = torch.bitwise_right_shift(left_boundary, bit_width_trunc).to(onehot_dtype)
    onehot_right_boundary = torch.bitwise_right_shift(right_boundary, bit_width_trunc).to(onehot_dtype)

    pyFastFss.mod(onehot_masked_in, onehot_bit_width_in, out=onehot_masked_in)
    pyFastFss.mod(onehot_left_boundary, onehot_bit_width_in, out=onehot_left_boundary)
    pyFastFss.mod(onehot_right_boundary, onehot_bit_width_in, out=onehot_right_boundary)

    time_et = time.time_ns()
    print("onehot init", (time_et - time_st) / 1e6, "ms")

    time_st = time.time_ns()

    onehot_shared_out0 = torch.empty(0, device=onehot_device, dtype=onehot_dtype)
    pyFastFss.dcf_mic_eval(onehot_shared_out0, onehot_masked_in,
                           key0["onehot"]["key"], key0["onehot"]["shared_z"], key0["onehot"]["seed"],
                           0, onehot_left_boundary, onehot_right_boundary,
                           onehot_bit_width_in, onehot_bit_width_out, onehot_masked_in.numel()
                           )
    onehot_shared_out0 = onehot_shared_out0.reshape(*batch_dims, n, m)

    time_et = time.time_ns()
    print("onehot0", (time_et - time_st) / 1e6, "ms")

    time_st = time.time_ns()

    onehot_shared_out1 = torch.empty(0, device=onehot_device, dtype=onehot_dtype)
    pyFastFss.dcf_mic_eval(onehot_shared_out1, onehot_masked_in,
                           key1["onehot"]["key"], key1["onehot"]["shared_z"], key1["onehot"]["seed"],
                           1, onehot_left_boundary, onehot_right_boundary,
                           onehot_bit_width_in, onehot_bit_width_out, onehot_masked_in.numel()
                           )
    onehot_shared_out1 = onehot_shared_out1.reshape(*batch_dims, n, m)

    time_et = time.time_ns()
    print("onehot0", (time_et - time_st) / 1e6, "ms")

    # print("onehot", onehot_bit_width_in, onehot_bit_width_out)
    # print(pyFastFss.mod(onehot_shared_out0 + onehot_shared_out1, onehot_bit_width_out))

    # ==========================================================================
    # ======================= Step2: bucket ====================================
    # ==========================================================================

    bucket_bit_width_in = onehot_bit_width_out
    bucket_bit_width_out = bucket_bit_width_in
    bucket_dtype = onehot_dtype
    bucket_device = onehot_device

    shared_bucket0 = onehot_shared_out0.sum(-2)
    shared_bucket1 = onehot_shared_out1.sum(-2)

    masked_bucket0 = shared_bucket0 + key0["bucket"]["bucket_r_in"]
    masked_bucket1 = shared_bucket1 + key1["bucket"]["bucket_r_in"]

    masked_bucket = masked_bucket0 + masked_bucket1

    # print("bucket", bucket_bit_width_in, bucket_bit_width_out)
    # print(pyFastFss.mod(shared_bucket0 + shared_bucket1, bucket_bit_width_out))

    # ==========================================================================
    # ======================= Step3: greater0 ==================================
    # ==========================================================================

    time_st = time.time_ns()

    gt0_bit_width_in = bucket_bit_width_out
    gt0_bit_width_out = m
    gt0_dtype = pyFastFss.dtype(max(gt0_bit_width_in, gt0_bit_width_out))
    gt0_device = bucket_device
    gt0_masked_in = masked_bucket.to(gt0_dtype)
    gt0_left_boundary = torch.tensor([1], dtype=gt0_dtype, device=gt0_device)
    gt0_right_boundary = pyFastFss.mod(torch.tensor([-1], dtype=gt0_dtype, device=gt0_device), gt0_bit_width_in)

    pyFastFss.mod(gt0_masked_in, gt0_bit_width_in, out=gt0_masked_in)
    pyFastFss.mod(gt0_left_boundary, gt0_bit_width_in, out=gt0_left_boundary)
    pyFastFss.mod(gt0_right_boundary, gt0_bit_width_in, out=gt0_right_boundary)

    time_et = time.time_ns()
    print("gt0 init", (time_et - time_st) / 1e6, "ms")

    time_st = time.time_ns()

    gt0_shared_out0 = torch.empty(0, device=gt0_device, dtype=gt0_dtype)
    pyFastFss.dcf_mic_eval(gt0_shared_out0, gt0_masked_in,
                           key0["gt0"]["key"], key0["gt0"]["shared_z"], key0["gt0"]["seed"],
                           0, gt0_left_boundary, gt0_right_boundary,
                           gt0_bit_width_in, gt0_bit_width_out, gt0_masked_in.numel()
                           )
    gt0_shared_out0 = gt0_shared_out0.reshape(*batch_dims, m)

    time_et = time.time_ns()
    print("gt0 0", (time_et - time_st) / 1e6, "ms")

    time_st = time.time_ns()

    gt0_shared_out1 = torch.empty(0, device=gt0_device, dtype=gt0_dtype)
    pyFastFss.dcf_mic_eval(gt0_shared_out1, gt0_masked_in,
                           key1["gt0"]["key"], key1["gt0"]["shared_z"], key1["gt0"]["seed"],
                           1, gt0_left_boundary, gt0_right_boundary,
                           gt0_bit_width_in, gt0_bit_width_out, gt0_masked_in.numel()
                           )
    gt0_shared_out1 = gt0_shared_out1.reshape(*batch_dims, m)

    time_et = time.time_ns()
    print("gt0 1", (time_et - time_st) / 1e6, "ms")

    # print("gt0", gt0_bit_width_in, gt0_bit_width_out)
    # print(pyFastFss.mod(gt0_shared_out0 + gt0_shared_out1, gt0_bit_width_out))

    # ==========================================================================
    # ========================== Step4: y ======================================
    # ==========================================================================

    y_bit_width_in = gt0_bit_width_out
    y_bit_width_out = y_bit_width_in
    y_dtype = gt0_dtype
    y_device = gt0_device

    shared_y0 = torch.zeros((*batch_dims,), dtype=y_dtype, device=y_device)
    shared_y1 = torch.zeros((*batch_dims,), dtype=y_dtype, device=y_device)
    for i in range(m):
        shared_y0 = torch.add(shared_y0, torch.bitwise_left_shift(gt0_shared_out0[:, i], i))
    for i in range(m):
        shared_y1 = torch.add(shared_y1, torch.bitwise_left_shift(gt0_shared_out1[:, i], i))

    masked_y0 = shared_y0 + key0["y"]["y_r_in"]
    masked_y1 = shared_y1 + key1["y"]["y_r_in"]

    masked_y = masked_y0 + masked_y1

    # ==========================================================================
    # ==================== Step5: interval lut =================================
    # ==========================================================================

    time_st = time.time_ns()

    interval_lut_bit_width_in = y_bit_width_out
    interval_lut_bit_width_out = bit_width_out
    interval_lut_dtype = pyFastFss.dtype(max(interval_lut_bit_width_in, interval_lut_bit_width_out))
    interval_lut_device = y_device
    interval_lut_masked_in = masked_y.to(interval_lut_dtype)
    interval_lut_left_boundary = torch.tensor([2 ** i for i in range(m)],
                                              dtype=interval_lut_dtype,
                                              device=interval_lut_device)
    interval_lut_right_boundary = torch.tensor([2 ** (i + 1) - 1 for i in range(m)],
                                               dtype=interval_lut_dtype,
                                               device=interval_lut_device)

    pyFastFss.mod(interval_lut_masked_in, interval_lut_bit_width_in, out=interval_lut_masked_in)
    pyFastFss.mod(interval_lut_left_boundary, interval_lut_bit_width_in, out=interval_lut_left_boundary)
    pyFastFss.mod(interval_lut_right_boundary, interval_lut_bit_width_in, out=interval_lut_right_boundary)

    time_et = time.time_ns()
    print("interval lut init", (time_et - time_st) / 1e6, "ms")

    time_st = time.time_ns()

    interval_lut_shared_out0 = torch.empty(0, device=interval_lut_device, dtype=interval_lut_dtype)
    pyFastFss.dcf_mic_eval(interval_lut_shared_out0, interval_lut_masked_in,
                           key0["interval_lut"]["key"], key0["interval_lut"]["shared_z"], key0["interval_lut"]["seed"],
                           0, interval_lut_left_boundary, interval_lut_right_boundary,
                           interval_lut_bit_width_in, interval_lut_bit_width_out, interval_lut_masked_in.numel()
                           )
    interval_lut_shared_out0 = interval_lut_shared_out0.reshape(*batch_dims, m)

    time_et = time.time_ns()
    print("interval lut 0", (time_et - time_st) / 1e6, "ms")

    time_st = time.time_ns()

    interval_lut_shared_out1 = torch.empty(0, device=interval_lut_device, dtype=interval_lut_dtype)
    pyFastFss.dcf_mic_eval(interval_lut_shared_out1, interval_lut_masked_in,
                           key1["interval_lut"]["key"], key1["interval_lut"]["shared_z"], key1["interval_lut"]["seed"],
                           1, interval_lut_left_boundary, interval_lut_right_boundary,
                           interval_lut_bit_width_in, interval_lut_bit_width_out, interval_lut_masked_in.numel()
                           )
    interval_lut_shared_out1 = interval_lut_shared_out1.reshape(*batch_dims, m)

    time_et = time.time_ns()
    print("interval lut 1", (time_et - time_st) / 1e6, "ms")

    # print("interval", interval_lut_bit_width_in, interval_lut_bit_width_out)
    # print(pyFastFss.mod(interval_lut_shared_out0 + interval_lut_shared_out1, interval_lut_bit_width_out))

    interval_lut_shared_out0 = (interval_lut_shared_out0 * right_boundary).sum(-1)
    interval_lut_shared_out1 = (interval_lut_shared_out1 * right_boundary).sum(-1)

    # print("out: ", pyFastFss.mod(interval_lut_shared_out0 + interval_lut_shared_out1, bit_width_out))

    return pyFastFss.mod(interval_lut_shared_out0 + interval_lut_shared_out1, bit_width_out)


def _main():
    shape = (12 * 128, 128,)
    # shape = (12 * 256, 256,)
    # shape = (16 * 128, 128,)
    # shape = (1, 1024,)
    bit_width_in = 20
    bit_width_out = bit_width_in
    bit_width_trunc = 10
    left_boundary = torch.tensor(
        [item * pow(2, 12) for item in [4 * i for i in range(16)]]
    )
    right_boundary = torch.tensor(
        [item * pow(2, 12) - 1 for item in [4 * (i + 1) for i in range(16)]]
    )
    alpha = torch.randint(0, pow(2, bit_width_in) - 1, shape)
    x = pyFastFss.mod(torch.randint(0, pow(2, bit_width_in) - 1, shape), bit_width_in)

    left_boundary = left_boundary.to("cuda")
    right_boundary = right_boundary.to("cuda")
    alpha = alpha.to("cuda")
    x = x.to("cuda")

    print("------------------------")
    print("shape", shape)
    print("bit_width_in", bit_width_in, "bit_width_out", bit_width_out, "bit_width_trunc", bit_width_trunc)

    print("====================== cuda ===========================")
    print("-------------------------------")
    key0, key1 = dcf_approx_max_key_gen(
        alpha, bit_width_in, bit_width_out, bit_width_trunc, left_boundary, right_boundary
    )
    print("-------------------------------")
    out_cuda = test_dcf_approx_max_eval(x + alpha, key0, key1, bit_width_in, bit_width_out, bit_width_trunc,
                                        left_boundary, right_boundary)

    print("====================== cpu ===========================")
    left_boundary = left_boundary.to("cpu")
    right_boundary = right_boundary.to("cpu")
    alpha = alpha.to("cpu")
    x = x.to("cpu")
    print("-------------------------------")
    key0, key1 = dcf_approx_max_key_gen(
        alpha, bit_width_in, bit_width_out, bit_width_trunc, left_boundary, right_boundary
    )
    print("-------------------------------")
    out_cpu = test_dcf_approx_max_eval(x + alpha, key0, key1, bit_width_in, bit_width_out, bit_width_trunc,
                                       left_boundary, right_boundary)
    print("-------------------------------")
    print("out(cpu) = out(cuda)", torch.equal(out_cpu, out_cuda.to("cpu")))


_main()
