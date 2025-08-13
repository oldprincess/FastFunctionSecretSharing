# benchmark

## cuda

compile .cu benchmark

```shell
# lut ex (optimized)
nvcc benchmark/cuda/grotto_lut_ex.cu -o cuda_benchmark_grotto_lut_ex.exe -std=c++17 -I include src/cuda/grotto.cu src/cuda/prng.cu src/cuda/config.cpp --expt-relaxed-constexpr
# lut ex2 (default)
nvcc benchmark/cuda/grotto_lut_ex2.cu -o cuda_benchmark_grotto_lut_ex2.exe -std=c++17 -I include src/cuda/grotto.cu src/cuda/prng.cu src/cuda/config.cpp --expt-relaxed-constexpr
```

The experiment was conducted with input dimensions of (128, 3072), a bit width of 20, and a LUT size of $2^9$, across 10 iterations.

```shell
./cuda_benchmark_grotto_lut_ex.exe 128*3072 20 9 10
./cuda_benchmark_grotto_lut_ex2.exe 128*3072 20 9 10
```

query metrics

```shell
# powershell
.\ncu.exe --query-metrics | Select-String -Pattern "branch"
# unix

```

| metric| |description|
|:-|:-|:-|
|sm__sass_average_branch_targets_threads_uniform||proportion of branch targets where all active threads selected the|
|sm__sass_branch_targets_threads_divergent||incremented only when there are two or more active threads with|
|smsp__average_warps_issue_stalled_branch_resolving_per_issue_active|inst|average # of warps resident per issue cycle, waiting for a branch|
|smsp__average_warp_latency_issue_stalled_branch_resolving|inst/warp|average # of warp cycles spent waiting for a branch target address to|

analyze using ncu

```shell
# lut ex (optimized)
ncu --metrics sm__sass_average_branch_targets_threads_uniform,sm__sass_branch_targets_threads_divergent,smsp__average_warps_issue_stalled_branch_resolving_per_issue_active,smsp__average_warp_latency_issue_stalled_branch_resolving --kernel-name grottoLutEvalKernel_ex cuda_benchmark_grotto_lut_ex.exe 128*3072 20 9 1
# lut ex2 (default)
ncu --metrics sm__sass_average_branch_targets_threads_uniform,sm__sass_branch_targets_threads_divergent,smsp__average_warps_issue_stalled_branch_resolving_per_issue_active,smsp__average_warp_latency_issue_stalled_branch_resolving --kernel-name grottoLutEvalKernel_ex2 cuda_benchmark_grotto_lut_ex2.exe 128*3072 20 9 1
```
