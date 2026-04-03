#include <FastFss/cuda/prng.h>
#include <FastFss/errors.h>
#include <cuda_runtime.h>

#include <cstdio>

#include "../impl/aes.h"

#define CUDA_CHECK(expression, do_something)                                                              \
    if ((expression) != cudaSuccess)                                                                      \
    {                                                                                                     \
        std::printf("[error] %s in %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
        do_something                                                                                      \
    }

#define CUDA_ERR_CHECK(do_something)                                                                      \
    if (cudaPeekAtLastError() != cudaSuccess)                                                             \
    {                                                                                                     \
        std::printf("[error] %s in %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
        do_something                                                                                      \
    }

namespace FastFss::cuda {

struct Prng
{
public:
    std::uint8_t seed[16];
    std::uint8_t counterTmp[16];
    std::uint8_t counter[16];
};

static __global__ void aes128_ctr_kernel(const void *seed,
                                         void       *counterOut,
                                         const void *counter,
                                         void       *dst,
                                         std::size_t bytesNum)
{
    std::size_t idx    = threadIdx.x + blockIdx.x * blockDim.x;
    std::size_t stride = blockDim.x * gridDim.x;

    __shared__ impl::AES128 aes128ctx;
    if (threadIdx.x == 0)
    {
        aes128ctx.set_enc_key(seed);
    }
    __syncthreads();

    std::size_t          blockNum   = bytesNum / 16;
    std::size_t          bytesRem   = bytesNum % 16;
    const std::uint64_t *counterPtr = (const std::uint64_t *)counter;
    std::uint8_t        *dstPtr     = (std::uint8_t *)dst;

    std::size_t i = idx;
    for (; i < blockNum; i += stride)
    {
        std::size_t dstOffest = i * 16;

        std::uint64_t curCounter[2];
        curCounter[0] = counterPtr[0] + i;
        curCounter[1] = counterPtr[1] + (curCounter[0] < i);

        aes128ctx.enc_block(dstPtr + dstOffest, curCounter);
    }

    if (i == blockNum)
    {
        if (bytesRem != 0)
        {
            std::uint8_t  outputBufer[16];
            std::uint64_t curCounter[2];
            curCounter[0] = counterPtr[0] + blockNum;
            curCounter[1] = counterPtr[1] + (curCounter[0] < blockNum);

            std::size_t dstOffest = blockNum * 16;

            aes128ctx.enc_block(outputBufer, curCounter);
            for (std::size_t i = 0; i < bytesRem; i++)
            {
                dstPtr[dstOffest + i] = outputBufer[i];
            }
            blockNum += 1;
        }
        std::uint64_t *counterOutPtr = (std::uint64_t *)counterOut;

        counterOutPtr[0] = counterPtr[0] + blockNum;
        counterOutPtr[1] = counterPtr[1] + (counterPtr[0] < blockNum);
    }
}

} // namespace FastFss::cuda

void *FastFss_cuda_prngInit()
{
    void       *prng = nullptr;
    cudaError_t err  = cudaMalloc(&prng, sizeof(FastFss::cuda::Prng));
    if (err != cudaSuccess)
    {
        return nullptr;
    }
    return prng;
}

void FastFss_cuda_prngRelease(void *prng)
{
    if (prng == nullptr)
    {
        return;
    }
    cudaFree(prng);
}

int FastFss_cuda_prngSetCurrentSeed(void *prng, const void *hostSeed128bit, const void *hostCounter128bit)
{
    if (prng == nullptr || hostSeed128bit == nullptr)
    {
        return FAST_FSS_PRNG_INPUT_INVALID_ARGUMENT;
    }
    FastFss::cuda::Prng *prngObj = (FastFss::cuda::Prng *)prng;
    CUDA_CHECK(cudaMemcpy(prngObj->seed, hostSeed128bit, 16, cudaMemcpyHostToDevice),
               { return FAST_FSS_RUNTIME_ERROR; });
    if (hostCounter128bit != nullptr)
    {
        CUDA_CHECK(cudaMemcpy(prngObj->counter, hostCounter128bit, 16, cudaMemcpyHostToDevice),
                   { return FAST_FSS_RUNTIME_ERROR; });
    }
    else
    {
        CUDA_CHECK(cudaMemset(prngObj->counter, 0, 16), { return FAST_FSS_RUNTIME_ERROR; });
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_prngGetCurrentSeed(const void *prng, void *hostSeed128bit, void *hostCounter128bit)
{
    if (prng == nullptr || hostSeed128bit == nullptr)
    {
        return FAST_FSS_PRNG_INPUT_INVALID_ARGUMENT;
    }
    auto deviceSeed    = ((const FastFss::cuda::Prng *)prng)->seed;
    auto deviceCounter = ((const FastFss::cuda::Prng *)prng)->counter;

    CUDA_CHECK(cudaMemcpy(hostSeed128bit, deviceSeed, 16, cudaMemcpyDeviceToHost), { return FAST_FSS_RUNTIME_ERROR; });
    if (hostCounter128bit != nullptr)
    {
        CUDA_CHECK(cudaMemcpy(hostCounter128bit, deviceCounter, 16, cudaMemcpyDeviceToHost),
                   { return FAST_FSS_RUNTIME_ERROR; });
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_cuda_prngGen(void  *prng,
                         void  *deviceDst,
                         size_t bitWidth,
                         size_t elementSize,
                         size_t elementNum,
                         void  *cudaStreamPtr)
{
    if (prng == nullptr || deviceDst == nullptr)
    {
        return FAST_FSS_PRNG_INPUT_INVALID_ARGUMENT;
    }
    if (bitWidth == 0 || bitWidth > elementSize * 8)
    {
        return FAST_FSS_PRNG_INVALID_BIT_WIDTH;
    }
    FastFss::cuda::Prng *prngObj = (FastFss::cuda::Prng *)prng;

    int BLOCK_SIZE = 512;
    int GRID_SIZE  = (elementNum * elementSize + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);

    cudaStream_t stream = (cudaStreamPtr) ? *(cudaStream_t *)cudaStreamPtr : 0;

    FastFss::cuda::aes128_ctr_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>( //
        prngObj->seed,                                                      //
        prngObj->counterTmp,                                                //
        prngObj->counter,                                                   //
        deviceDst,                                                          //
        elementNum * elementSize                                            //
    );                                                                      //
    CUDA_ERR_CHECK({ return FAST_FSS_RUNTIME_ERROR; });
    CUDA_CHECK(cudaMemcpy(prngObj->counter, prngObj->counterTmp, 16, cudaMemcpyDeviceToDevice),
               { return FAST_FSS_RUNTIME_ERROR; });

    return FAST_FSS_SUCCESS;
}