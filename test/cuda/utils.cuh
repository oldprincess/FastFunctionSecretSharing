#ifndef TEST_CUDA_UTILS_CUH
#define TEST_CUDA_UTILS_CUH

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(expression, do_something)                          \
    if (expression != cudaSuccess)                                    \
    {                                                                 \
        std::printf("[error] %s in %s:%d\n",                          \
                    cudaGetErrorString(cudaGetLastError()), __FILE__, \
                    __LINE__);                                        \
        do_something                                                  \
    }

#define CUDA_ERR_CHECK(do_something)                                  \
    if (cudaDeviceSynchronize() != cudaSuccess)                       \
    {                                                                 \
        std::printf("[error] %s in %s:%d\n",                          \
                    cudaGetErrorString(cudaGetLastError()), __FILE__, \
                    __LINE__);                                        \
        do_something                                                  \
    }

namespace FastFss::cuda {

inline void memcpy_cpu2gpu(void*       deviceDst,
                           const void* hostSrc,
                           std::size_t size) noexcept
{
    CUDA_CHECK(cudaMemcpy(deviceDst, hostSrc, size, cudaMemcpyHostToDevice),
               { std::exit(-1); });
}

inline void memcpy_gpu2cpu(void*       hostDst,
                           const void* deviceSrc,
                           std::size_t size) noexcept
{
    CUDA_CHECK(cudaMemcpy(hostDst, deviceSrc, size, cudaMemcpyDeviceToHost),
               { std::exit(-1); });
}

inline void memcpy_gpu2gpu(void*       deviceDst,
                           const void* deviceSrc,
                           std::size_t size) noexcept
{
    CUDA_CHECK(cudaMemcpy(deviceDst, deviceSrc, size, cudaMemcpyDeviceToDevice),
               { std::exit(-1); });
}

inline void* malloc_host(std::size_t size) noexcept
{
    void* host_ptr = NULL;
    CUDA_CHECK(cudaMallocHost(&host_ptr, size), { std::exit(-1); });
    return host_ptr;
}

inline void* malloc_gpu(std::size_t size) noexcept
{
    void* device_ptr = NULL;
    CUDA_CHECK(cudaMalloc((void**)&device_ptr, size), { std::exit(-1); });
    return device_ptr;
}

inline void malloc_gpu_inplace(void** device_ptr, std::size_t size) noexcept
{
    CUDA_CHECK(cudaMalloc(device_ptr, size), { std::exit(-1); });
}

inline void free_gpu(void* device_ptr)
{
    CUDA_CHECK(cudaFree(device_ptr), { std::exit(-1); });
}

inline void free_host(void* host_ptr)
{
    CUDA_CHECK(cudaFreeHost(host_ptr), { std::exit(-1); });
}

inline int memcmp_gpu(const void* devicePtr1,
                      const void* devicePtr2,
                      std::size_t size)
{
    int   ret;
    void* host_ptr1 = std::malloc(size);
    void* host_ptr2 = std::malloc(size);

    if (host_ptr1 == NULL || host_ptr2 == NULL)
    {
        std::printf("[err] %s:%d\n", __FILE__, __LINE__);
        std::exit(-1);
    }

    memcpy_gpu2cpu(host_ptr1, devicePtr1, size);
    memcpy_gpu2cpu(host_ptr2, devicePtr2, size);

    ret = std::memcmp(host_ptr1, host_ptr2, size);

    std::free(host_ptr1);
    std::free(host_ptr2);

    return ret;
}

inline void memset_gpu(void* deviceDst, int value, std::size_t size)
{
    CUDA_CHECK(cudaMemset(deviceDst, value, size), { std::exit(-1); });
}

} // namespace FastFss::cuda

#endif