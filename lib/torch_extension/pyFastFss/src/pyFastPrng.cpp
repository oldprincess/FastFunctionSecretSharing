#include "pyFastPrng.h"

#include <FastFss/cpu/prng.h>
#include <FastFss/cuda/prng.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/python.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define ERR_LOG(fmt, ...)                                                  \
    std::fprintf(stderr, "[FastFss PRNG] " fmt ". %s:%d\n", ##__VA_ARGS__, \
                 __FILE__, __LINE__)

#define ARG_ASSERT(exp)                                    \
    if (!(exp))                                            \
    {                                                      \
        ERR_LOG("assert fail: " #exp);                     \
        throw std::invalid_argument("assert fail: " #exp); \
    }

#define CHECK_ERROR_CODE(ret, func)             \
    if (ret != 0)                               \
    {                                           \
        ERR_LOG(func "ret = %d", ret);          \
        throw std::runtime_error(func " fail"); \
    }

namespace pyFastFss {

Prng::Prng(torch::Device device) : device_{device}
{
    if (device.type() == torch::kCPU)
    {
        ctx_ = FastFss_cpu_prngInit();
        if (ctx_ == nullptr)
        {
            throw std::runtime_error("FastFss_cpu_prngInit failed");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        ctx_ = FastFss_cuda_prngInit();
        if (ctx_ == nullptr)
        {
            throw std::runtime_error("FastFss_cuda_prngInit failed");
        }
    }
    else
    {
        throw std::invalid_argument("Unsupported device type");
    }
}

Prng::~Prng()
{
    if (ctx_ != nullptr)
    {
        if (device_.type() == torch::kCPU)
        {
            FastFss_cpu_prngRelease(ctx_);
            ctx_ = nullptr;
        }
        else if (device_.type() == torch::kCUDA)
        {
            FastFss_cuda_prngRelease(ctx_);
            ctx_ = nullptr;
        }
    }
}

void Prng::set_current_seed(py::bytes seed128bit, py::bytes counter128bit)
{
    std::string seed128bit_str    = seed128bit;
    std::string counter128bit_str = counter128bit;
    if (seed128bit_str.size() != 16 || counter128bit_str.size() != 16)
    {
        ERR_LOG("seed128bit_str.size() != 16 || " //
                "counter128bit_str.size() != 16");
        throw std::runtime_error(                           //
            "seed128bit and counter128bit must be 16 bytes" //
        );                                                  //
    }
    if (device_.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_prngSetCurrentSeed(                 //
            ctx_, seed128bit_str.data(), counter128bit_str.data() //
        );                                                        //
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_prngSetCurrentSeed ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_prngSetCurrentSeed failed");
        }
    }
    else if (device_.type() == torch::kCUDA)
    {
        int ret = FastFss_cuda_prngSetCurrentSeed(                //
            ctx_, seed128bit_str.data(), counter128bit_str.data() //
        );                                                        //

        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_prngSetCurrentSeed ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_prngSetCurrentSeed failed");
        }
    }
    else
    {
        ERR_LOG("Unsupported device type");
        throw std::invalid_argument("Unsupported device type");
    }
}

py::tuple Prng::get_current_seed() const
{
    std::uint8_t seed128bit[16];
    std::uint8_t counter128bit[16];
    if (device_.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_prngGetCurrentSeed( //
            ctx_, seed128bit, counter128bit       //
        );                                        //
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_prngGetCurrentSeed failed ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_prngGetCurrentSeed failed");
        }
    }
    else if (device_.type() == torch::kCUDA)
    {
        int ret = FastFss_cuda_prngGetCurrentSeed( //
            ctx_, seed128bit, counter128bit        //
        );                                         //
        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_prngGetCurrentSeed failed ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_prngGetCurrentSeed failed");
        }
    }
    else
    {
        ERR_LOG("Unsupported device type");
        throw std::invalid_argument("Unsupported device type");
    }
    return py::make_tuple(                         //
        py::bytes((const char *)seed128bit, 16),   //
        py::bytes((const char *)counter128bit, 16) //
    );                                             //
}

torch::Device Prng::device() const
{
    return device_;
}

Prng &Prng::to_(torch::Device device)
{
    // early check
    if (device.type() == torch::kCPU)
    {
        if (device_.type() == torch::kCPU)
        {
            return *this;
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        if (device_.type() == torch::kCUDA)
        {
            return *this;
        }
    }
    else
    {
        ERR_LOG("Unsupported device type");
        throw std::invalid_argument("Unsupported device type");
    }
    //
    std::uint8_t seed128bit[16];
    std::uint8_t counter128bit[16];
    if (device_.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_prngGetCurrentSeed( //
            ctx_, seed128bit, counter128bit       //
        );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_prngGetCurrentSeed failed ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_prngGetCurrentSeed failed");
        }
    }
    if (device_.type() == torch::kCUDA)
    {
        int ret = FastFss_cuda_prngGetCurrentSeed( //
            ctx_, seed128bit, counter128bit        //
        );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_prngGetCurrentSeed failed ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_prngGetCurrentSeed failed");
        }
    }

    //
    int   ret    = 0;
    void *newCtx = nullptr;
    if (device.type() == torch::kCPU)
    {
        newCtx = FastFss_cpu_prngInit();
    }
    if (device.type() == torch::kCUDA)
    {
        newCtx = FastFss_cuda_prngInit();
    }
    if (newCtx == nullptr)
    {
        throw std::runtime_error("FastFss_prngInit failed");
    }

    if (device.type() == torch::kCPU)
    {
        ret = FastFss_cpu_prngSetCurrentSeed(newCtx, seed128bit, counter128bit);
    }
    if (device.type() == torch::kCUDA)
    {
        ret =
            FastFss_cuda_prngSetCurrentSeed(newCtx, seed128bit, counter128bit);
    }
    if (ret != 0)
    {
        throw std::runtime_error("FastFss_prngSetCurrentSeed failed");
    }
    // copy
    if (device_.type() == torch::kCPU)
    {
        FastFss_cpu_prngRelease(ctx_);
    }
    if (device_.type() == torch::kCUDA)
    {
        FastFss_cuda_prngRelease(ctx_);
    }
    ctx_    = newCtx;
    device_ = device;

    return *this;
}

torch::Tensor &Prng::rand_(torch::Tensor &out, std::size_t bitWidth)
{
    ARG_ASSERT(out.is_contiguous());
    ARG_ASSERT(device_.type() == out.device().type());

    std::size_t elementSize = out.dtype().itemsize();
    ARG_ASSERT(bitWidth <= elementSize * 8);

    if (device_.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_prngGen(ctx_,                   //
                                      out.mutable_data_ptr(), //
                                      bitWidth,               //
                                      elementSize,            //
                                      out.numel()             //
        );                                                    //
        CHECK_ERROR_CODE(ret, "FastFss_cpu_prngGen");
    }
    else if (device_.type() == torch::kCUDA)
    {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int ret = FastFss_cuda_prngGen(ctx_,                   //
                                       out.mutable_data_ptr(), //
                                       bitWidth,               //
                                       elementSize,            //
                                       out.numel(),            //
                                       &stream                 //
        );                                                     //
        CHECK_ERROR_CODE(ret, "FastFss_cuda_prngGen");
    }
    else
    {
        throw std::runtime_error("Unsupported device type");
    }
    return out;
}

}; // namespace pyFastFss