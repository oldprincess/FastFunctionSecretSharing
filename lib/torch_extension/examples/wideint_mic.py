import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyFastFss


def limb_num(bit_width: int) -> int:
    return (bit_width + 63) // 64


def modulus(bit_width: int) -> int:
    return 1 << bit_width


def to_signed_limb(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


def from_signed_limb(value: int) -> int:
    return value & ((1 << 64) - 1)


def pack_values(values: list[int], logical_shape: tuple[int, ...], bit_width: int, device: torch.device) -> torch.Tensor:
    limbs = limb_num(bit_width)
    flat = []
    mask = modulus(bit_width) - 1
    for value in values:
        value &= mask
        for i in range(limbs):
            flat.append(to_signed_limb((value >> (64 * i)) & ((1 << 64) - 1)))
    return torch.tensor(flat, dtype=torch.int64, device=device).reshape(*logical_shape, limbs)


def unpack_values(tensor: torch.Tensor, bit_width: int) -> list[int]:
    limbs = tensor.shape[-1]
    flat = tensor.reshape(-1, limbs).tolist()
    values = []
    mask = modulus(bit_width) - 1
    for limb_row in flat:
        value = 0
        for i, limb in enumerate(limb_row):
            value |= from_signed_limb(limb) << (64 * i)
        values.append(value & mask)
    return values


def format_hex_list(values: list[int], bit_width: int) -> list[str]:
    width = math.ceil(bit_width / 4)
    return [f"0x{value:0{width}x}" for value in values]


def run_case(bit_width: int) -> None:
    device = torch.device("cpu")
    dtype = pyFastFss.dtype(bit_width)
    logical_shape = (2, 3)
    element_num = math.prod(logical_shape)
    interval_bounds = [
        (3, 9),
        (10, 20),
        (21, 42),
        (43, 63),
    ]
    interval_num = len(interval_bounds)
    ring_mod = modulus(bit_width)
    rng = random.Random(20260403 + bit_width)

    x_values = [3, 8, 12, 24, 44, 60]
    alpha_values = [rng.randrange(ring_mod) for _ in range(element_num)]
    masked_values = [(x + a) % ring_mod for x, a in zip(x_values, alpha_values)]

    x = pyFastFss.mod(pack_values(x_values, logical_shape, bit_width, device), bit_width)
    alpha = pyFastFss.mod(pack_values(alpha_values, logical_shape, bit_width, device), bit_width)
    masked_x = pyFastFss.mod(pack_values(masked_values, logical_shape, bit_width, device), bit_width)

    left_bound = pyFastFss.mod(
        pack_values([left for left, _ in interval_bounds], (interval_num,), bit_width, device),
        bit_width,
    )
    right_bound = pyFastFss.mod(
        pack_values([right for _, right in interval_bounds], (interval_num,), bit_width, device),
        bit_width,
    )

    key = torch.empty(0, dtype=torch.uint8, device=device)
    z = torch.empty(0, dtype=dtype, device=device)
    seed0 = torch.randint(0, 255, (element_num * 16,), dtype=torch.uint8, device=device)
    seed1 = torch.randint(0, 255, (element_num * 16,), dtype=torch.uint8, device=device)

    pyFastFss.dcf_mic_key_gen(key, z, alpha, seed0, seed1, left_bound, right_bound, bit_width, bit_width, element_num)

    z_values = unpack_values(z, bit_width)
    shared_z0_values = [rng.randrange(ring_mod) for _ in range(len(z_values))]
    shared_z1_values = [(z_value - share0) % ring_mod for z_value, share0 in zip(z_values, shared_z0_values)]
    shared_z0 = pack_values(shared_z0_values, logical_shape + (interval_num,), bit_width, device)
    shared_z1 = pack_values(shared_z1_values, logical_shape + (interval_num,), bit_width, device)

    shared_out0 = torch.empty(0, dtype=dtype, device=device)
    shared_out1 = torch.empty(0, dtype=dtype, device=device)

    pyFastFss.dcf_mic_eval(
        shared_out0,
        masked_x,
        key,
        shared_z0,
        seed0,
        0,
        left_bound,
        right_bound,
        bit_width,
        bit_width,
        element_num,
    )
    pyFastFss.dcf_mic_eval(
        shared_out1,
        masked_x,
        key,
        shared_z1,
        seed1,
        1,
        left_bound,
        right_bound,
        bit_width,
        bit_width,
        element_num,
    )

    out = pyFastFss.mod(shared_out0 + shared_out1, bit_width)
    out_values = unpack_values(out, bit_width)
    normalized_values = [value & 1 for value in out_values]
    expected_values = [
        1 if left <= x_value <= right else 0
        for x_value in x_values
        for left, right in interval_bounds
    ]

    print(f"bit_width = {bit_width}")
    print(f"dtype = {dtype}, logical_shape = {logical_shape}, limb_num = {limb_num(bit_width)}")
    print("x =", format_hex_list(x_values, bit_width))
    print("intervals =", [(hex(left), hex(right)) for left, right in interval_bounds])
    print("out.shape =", tuple(out.shape))
    print("raw wide outputs =", format_hex_list(out_values, bit_width))
    print("membership =", torch.tensor(normalized_values, dtype=torch.int64).reshape(*logical_shape, interval_num))
    print()

    if normalized_values != expected_values:
        raise RuntimeError(f"wideint MIC example failed for {bit_width} bits")


if __name__ == "__main__":
    for bit_width in (128, 196, 256):
        run_case(bit_width)
