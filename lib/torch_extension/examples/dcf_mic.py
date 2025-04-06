import pyFastFss
import torch

device = torch.device('cpu')
dtype = torch.int32
bit_width_in = 32
bit_width_out = 32


x = torch.tensor(
    [-3, -1, -3, -1, -4, 20, 42, -32], dtype=dtype, device=device
)
x = pyFastFss.mod(x, bit_width_in)

element_num = x.numel()

key = torch.empty(0, dtype=torch.uint8, device=device)
z = torch.empty(0, dtype=dtype, device=device)
seed0 = torch.randint(0, 255, (element_num * 16,), dtype=torch.uint8, device=device)
seed1 = torch.randint(0, 255, (element_num * 16,), dtype=torch.uint8, device=device)
left_bound = torch.tensor(
    [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50], dtype=dtype, device=device
)
right_bound = torch.tensor(
    [-41, -31, -21, -11, -1, 9, 19, 29, 39, 49, 59], dtype=dtype, device=device
)

left_bound = pyFastFss.mod(left_bound, bit_width_in)
right_bound = pyFastFss.mod(right_bound, bit_width_in)

alpha = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, (element_num,), dtype=dtype, device=device)
pyFastFss.dcf_mic_key_gen(key, z, alpha, seed0, seed1, left_bound, right_bound, bit_width_in, bit_width_out, element_num)

shared_z0 = torch.randint_like(z, torch.iinfo(dtype).min, torch.iinfo(dtype).max)
shared_z1 = z - shared_z0


shared_out0 = torch.empty(0, dtype=dtype, device=device)
shared_out1 = torch.empty(0, dtype=dtype, device=device)

masked_x = torch.add(x, alpha)

pyFastFss.dcf_mic_eval(shared_out0, masked_x, key, shared_z0, seed0,
                       0, left_bound, right_bound, bit_width_in, bit_width_out, element_num)
pyFastFss.dcf_mic_eval(shared_out1, masked_x, key, shared_z1, seed1,
                       1, left_bound, right_bound, bit_width_in, bit_width_out, element_num)

out = pyFastFss.mod(shared_out0 + shared_out1, bit_width_out).reshape(element_num, -1)

print(x)
print(left_bound)
print(right_bound)
print(out)
