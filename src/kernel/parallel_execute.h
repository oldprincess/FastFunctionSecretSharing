#ifndef SRC_KERNEL_PARALLEL_EXECUTE_H
#define SRC_KERNEL_PARALLEL_EXECUTE_H

#include <FastFss/errors.h>

#include <cstddef>
#include <type_traits>

#include "../impl/def.h"

#ifndef __CUDACC__
#include <FastFss/cpu/config.h>
#include <omp.h>

#include <vector>
#else
#include <FastFss/cuda/config.h>
#include <cuda_runtime.h>
#endif

namespace FastFss::kernel {

namespace detail {

template <typename Task, typename = void>
struct has_strided_support : std::false_type
{
};

template <typename Task>
struct has_strided_support<
    Task,
    std::void_t<decltype(std::declval<const Task &>().getStridedWorkerCount(std::declval<std::size_t>())),
                decltype(std::declval<const Task &>().getWorkspaceSize(std::declval<std::size_t>())),
                decltype(std::declval<Task &>().setWorkspace(nullptr, std::declval<std::size_t>())),
                decltype(std::declval<const Task &>().run(std::declval<std::size_t>(), std::declval<std::size_t>()))>>
    : std::true_type
{
};

template <typename Task>
inline constexpr bool has_strided_support_v = has_strided_support<Task>::value;

} // namespace detail

#ifndef __CUDACC__

template <class Task>
static inline int parallel_execute_regular(Task task)
{
    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (std::int64_t i = 0; i < (std::int64_t)(task.elementNum); ++i)
    {
        task((std::size_t)i);
    }
    return FAST_FSS_SUCCESS;
}

template <class Task>
static inline int parallel_execute_strided(Task task)
{
    std::size_t               workerCount = task.getStridedWorkerCount((std::size_t)FastFss_cpu_getNumThreads());
    std::vector<std::uint8_t> workspace(task.getWorkspaceSize(workerCount));

    omp_set_num_threads(FastFss_cpu_getNumThreads());
#pragma omp parallel for
    for (int i = 0; i < (int)workerCount; i++)
    {
        task.run((std::size_t)i, workerCount, workspace.data());
    }
    return FAST_FSS_SUCCESS;
}

template <class Task>
static inline int parallel_execute(Task task)
{
    int ret = (int)task.check();
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    if constexpr (detail::has_strided_support_v<Task>)
    {
        if (task.elementNum < (std::size_t)FastFss_cpu_getNumThreads())
        {
            return parallel_execute_strided(task);
        }
    }
    return parallel_execute_regular(task);
}

#else

template <class Task>
static __global__ void for_each_kernel(Task task)
{
    std::size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t stride = blockDim.x * gridDim.x;
    std::size_t n      = task.elementNum;
    for (; idx < n; idx += stride)
    {
        task(idx);
    }
}

template <class Task>
static __global__ void strided_kernel(Task task, void *workSpace)
{
    std::size_t workerIdx   = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t workerCount = blockDim.x * gridDim.x;
    task.run(workerIdx, workerCount, workSpace);
}

template <class Task>
static inline int parallel_execute_regular(Task task, cudaStream_t stream)
{
    int         maxGridDim = FastFss_cuda_getGridDim();
    std::size_t block      = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t grid       = (task.elementNum + block - 1) / block;
    grid                   = (grid > maxGridDim) ? maxGridDim : grid;

    for_each_kernel<<<grid, block, 0, stream>>>(task);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        return FAST_FSS_RUNTIME_ERROR;
    }
    return FAST_FSS_SUCCESS;
}

template <class Task>
static inline int parallel_execute_strided(Task task, cudaStream_t stream)
{
    std::size_t block       = CUDA_DEFAULT_BLOCK_DIM;
    std::size_t workerCount = task.getStridedWorkerCount(block * FastFss_cuda_getGridDim());
    std::size_t grid        = (workerCount + block - 1) / block;

    std::size_t workspaceSize = task.getWorkspaceSize(workerCount);
    void       *workspace     = nullptr;
    if (cudaMallocAsync(&workspace, workspaceSize, stream) != cudaSuccess)
    {
        return FAST_FSS_RUNTIME_ERROR;
    }
    strided_kernel<<<grid, block, 0, stream>>>(task, workspace);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        cudaFreeAsync(workspace, stream);
        return FAST_FSS_RUNTIME_ERROR;
    }
    cudaFreeAsync(workspace, stream);
    return FAST_FSS_SUCCESS;
}

template <class Task>
static inline int parallel_execute(Task task)
{
    int ret = (int)task.check();
    if (ret != FAST_FSS_SUCCESS)
    {
        return ret;
    }

    cudaStream_t stream = cudaStreamPerThread;
    if (task.cudaStreamPtr)
    {
        stream = *(cudaStream_t *)(task.cudaStreamPtr);
    }

    if constexpr (detail::has_strided_support_v<Task>)
    {
        if (task.elementNum < CUDA_DEFAULT_BLOCK_DIM * FastFss_cuda_getGridDim())
        {
            return parallel_execute_strided(task, stream);
        }
    }
    return parallel_execute_regular(task, stream);
}

#endif

}; // namespace FastFss::kernel

#endif
