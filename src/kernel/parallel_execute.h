#ifndef SRC_KERNEL_PARALLEL_EXECUTE_H
#define SRC_KERNEL_PARALLEL_EXECUTE_H

#include <FastFss/errors.h>

#include "../impl/def.h"

namespace FastFss::kernel {

#ifndef __CUDACC__

#include <FastFss/cpu/config.h>
#include <omp.h>

template <class Task>
static inline int parallel_execute(Task task)
{
    int ret = (int)task.check();
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = 0; i < (std::int64_t)(task.elementNum); ++i)
    {
        task((std::size_t)i);
    }
    return FAST_FSS_SUCCESS;
}

#else

#include <FastFss/cuda/config.h>
#include <cuda_runtime.h>

template <class Task>
static __global__ void for_each_kernel(std::size_t n, Task task)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t s = blockDim.x * gridDim.x;
    for (; i < n; i += s)
    {
        task(i);
    }
}

template <class Task>
static inline int parallel_execute(Task task)
{
    int ret = (int)task.check();
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    std::size_t block = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t grid  = (task.elementNum + block - 1) / block;
    if (grid > CUDA_MAX_GRID_DIM)
    {
        grid = CUDA_MAX_GRID_DIM;
    }
    cudaStream_t stream = 0;
    if (task.cudaStreamPtr)
    {
        stream = *(cudaStream_t *)(task.cudaStreamPtr);
    }

    for_each_kernel<<<grid, block, 0, stream>>>(task.elementNum, task);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        return FAST_FSS_RUNTIME_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

#endif

}; // namespace FastFss::kernel

#endif
