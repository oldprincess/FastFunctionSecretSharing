#ifndef FAST_FSS_TESTS_COMMON_CUDA_TEST_UTILS_CUH
#define FAST_FSS_TESTS_COMMON_CUDA_TEST_UTILS_CUH

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#define FAST_FSS_CUDA_CHECK(expression, do_something)                                                     \
    if ((expression) != cudaSuccess)                                                                      \
    {                                                                                                     \
        std::printf("[error] %s in %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
        do_something                                                                                      \
    }

namespace FastFss::tests::cuda {

inline void memcpy_cpu_to_gpu(void *deviceDst, const void *hostSrc, std::size_t size)
{
    FAST_FSS_CUDA_CHECK(cudaMemcpy(deviceDst, hostSrc, size, cudaMemcpyHostToDevice), { std::exit(-1); });
}

inline void memcpy_gpu_to_cpu(void *hostDst, const void *deviceSrc, std::size_t size)
{
    FAST_FSS_CUDA_CHECK(cudaMemcpy(hostDst, deviceSrc, size, cudaMemcpyDeviceToHost), { std::exit(-1); });
}

inline void free_gpu(void *ptr)
{
    FAST_FSS_CUDA_CHECK(cudaFree(ptr), { std::exit(-1); });
}

inline std::unique_ptr<void, void (*)(void *)> make_gpu_buffer(std::size_t size)
{
    void *ptr = nullptr;
    FAST_FSS_CUDA_CHECK(cudaMalloc(&ptr, size), { std::exit(-1); });
    return {ptr, free_gpu};
}

inline std::unique_ptr<void, void (*)(void *)> make_gpu_buffer(void *hostPtr, std::size_t size)
{
    auto buffer = make_gpu_buffer(size);
    memcpy_cpu_to_gpu(buffer.get(), hostPtr, size);
    return buffer;
}

class StreamPair
{
public:
    StreamPair()
    {
        FAST_FSS_CUDA_CHECK(cudaStreamCreate(&streams_[0]), { std::exit(-1); });
        FAST_FSS_CUDA_CHECK(cudaStreamCreate(&streams_[1]), { std::exit(-1); });
    }

    ~StreamPair()
    {
        if (streams_[0] != nullptr)
        {
            cudaStreamDestroy(streams_[0]);
        }
        if (streams_[1] != nullptr)
        {
            cudaStreamDestroy(streams_[1]);
        }
    }

    void *party_stream(std::size_t index)
    {
        return &streams_[index];
    }
    void synchronize(std::size_t index)
    {
        FAST_FSS_CUDA_CHECK(cudaStreamSynchronize(streams_[index]), { std::exit(-1); });
    }
    void synchronize_all()
    {
        synchronize(0);
        synchronize(1);
    }

private:
    cudaStream_t streams_[2]{};
};

} // namespace FastFss::tests::cuda

#endif
