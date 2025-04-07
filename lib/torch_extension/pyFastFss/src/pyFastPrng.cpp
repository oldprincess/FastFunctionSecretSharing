#include <FastFss/cpu/prng.h>
#include <FastFss/cuda/prng.h>
#include <torch/python.h>

#include "pyFastFss.h"

#define ERR_LOG(fmt, ...)                                                  \
    std::fprintf(stderr, "[FastFss PRNG] " fmt ". %s:%d\n", ##__VA_ARGS__, \
                 __FILE__, __LINE__)

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

void Prng::to_(torch::Device device)
{
    // early check
    if (device.type() == torch::kCPU)
    {
        if (device_.type() == torch::kCPU)
        {
            return;
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        if (device_.type() == torch::kCUDA)
        {
            return;
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
}
torch::Tensor Prng::rand_(torch::Tensor out, std::size_t bitWidth)
{
    if (!out.is_contiguous())
    {
        ERR_LOG("out tensor must be contiguous");
        throw std::invalid_argument("out tensor must be contiguous");
    }
    if (device_.type() != out.device().type())
    {
        ERR_LOG("device type mismatch");
        throw std::invalid_argument("device type mismatch");
    }
    std::size_t elementSize = out.dtype().itemsize();
    if (bitWidth > elementSize * 8)
    {
        ERR_LOG("bitWidth = %zu is too large", bitWidth);
        throw std::invalid_argument("bitWidth is too large");
    }

    int ret = -1;
    if (device_.type() == torch::kCPU)
    {
        ret = FastFss_cpu_prngGen(ctx_,                   //
                                  out.mutable_data_ptr(), //
                                  bitWidth,               //
                                  elementSize,            //
                                  out.numel()             //
        );                                                //
    }
    if (device_.type() == torch::kCUDA)
    {
        ret = FastFss_cuda_prngGen(ctx_,                   //
                                   out.mutable_data_ptr(), //
                                   bitWidth,               //
                                   elementSize,            //
                                   out.numel()             //
        );                                                 //
    }
    if (ret != 0)
    {
        throw std::runtime_error("FastFss_prngGen failed");
    }
    return out;
}

}; // namespace pyFastFss