#include "pyFastDcf.h"

#include <FastFss/cpu/dcf.h>
#include <FastFss/cpu/mic.h>
#include <FastFss/cuda/dcf.h>
#include <FastFss/cuda/mic.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define ERR_LOG(fmt, ...)                                                 \
    std::fprintf(stderr, "[FastFss DCF] " fmt ". %s:%d\n", ##__VA_ARGS__, \
                 __FILE__, __LINE__)

namespace pyFastFss {

std::size_t dcf_get_key_data_size(std::size_t bitWidthIn,
                                  std::size_t bitWidthOut,
                                  std::size_t elementSize,
                                  std::size_t elementNum)
{
    int result = FastFss_cpu_dcfGetKeyDataSize(bitWidthIn, bitWidthOut,
                                               elementSize, elementNum);
    if (result < 0)
    {
        ERR_LOG("FastFss_cpu_dcfGetKeyDataSize ret = %d", result);
        throw std::runtime_error("FastFss_cpu_dcfGetKeyDataSize fail");
    }
    return (std::size_t)result;
}

std::size_t dcf_get_zipped_key_data_size(std::size_t bitWidthIn,
                                         std::size_t bitWidthOut,
                                         std::size_t elementSize,
                                         std::size_t elementNum)
{
    int result = FastFss_cpu_dcfGetZippedKeyDataSize(bitWidthIn, bitWidthOut,
                                                     elementSize, elementNum);
    if (result < 0)
    {
        ERR_LOG("FastFss_cpu_dcfGetZippedKeyDataSize ret = %d", result);
        throw std::runtime_error("FastFss_cpu_dcfGetZippedKeyDataSize fail");
    }
    return (std::size_t)result;
}

void dcf_key_zip(torch::Tensor zippedKeyOut,
                 torch::Tensor key,
                 std::size_t   bitWidthIn,
                 std::size_t   bitWidthOut,
                 std::size_t   elementNum)
{
    auto device = key.device();

    if (zippedKeyOut.device() != device)
    {
        ERR_LOG("zippedKeyOut.device() != device");
        throw std::invalid_argument("zippedKeyOut.device() != device");
    }
    if (zippedKeyOut.dtype() != torch::kUInt8 || key.dtype() != torch::kUInt8)
    {
        ERR_LOG("zippedKeyOut.dtype() != torch::kUInt8 || "
                "key.dtype() != torch::kUInt8");
        throw std::invalid_argument("zippedKeyOut.dtype() != torch::kUInt8 || "
                                    "key.dtype() != torch::kUInt8");
    }

    std::size_t zippedKeyDataSize =
        dcf_get_zipped_key_data_size(bitWidthIn, bitWidthOut, 1, elementNum);
    if ((std::size_t)zippedKeyOut.numel() != zippedKeyDataSize)
    {
        zippedKeyOut.resize_({(std::int64_t)zippedKeyDataSize});
    }

    void*       zippedKeyOutPtr = zippedKeyOut.mutable_data_ptr();
    std::size_t inputDataSize   = zippedKeyOut.numel();
    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dcfKeyZip(&zippedKeyOutPtr,         //
                                        &inputDataSize,           //
                                        key.const_data_ptr(),     //
                                        (std::size_t)key.numel(), //
                                        bitWidthIn,               //
                                        bitWidthOut,              //
                                        1,                        //
                                        elementNum                //
        );
        if (ret < 0)
        {
            ERR_LOG("FastFss_cpu_dcfKeyZip ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_dcfKeyZip fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        int ret = FastFss_cuda_dcfKeyZip(&zippedKeyOutPtr,         //
                                         &inputDataSize,           //
                                         key.const_data_ptr(),     //
                                         (std::size_t)key.numel(), //
                                         bitWidthIn,               //
                                         bitWidthOut,              //
                                         1,                        //
                                         elementNum                //
        );
        if (ret < 0)
        {
            ERR_LOG("FastFss_cuda_dcfKeyZip ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_dcfKeyZip fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

void dcf_key_unzip(torch::Tensor keyOut,
                   torch::Tensor zippedKey,
                   std::size_t   bitWidthIn,
                   std::size_t   bitWidthOut,
                   std::size_t   elementNum)
{
    auto device = zippedKey.device();

    if (keyOut.device() != device)
    {
        ERR_LOG("keyOut.device() != device");
        throw std::invalid_argument("keyOut.device() != device");
    }
    if (keyOut.dtype() != torch::kUInt8 || zippedKey.dtype() != torch::kUInt8)
    {
        ERR_LOG("keyOut.dtype() != torch::kUInt8 || "
                "zippedKey.dtype() != torch::kUInt8");
        throw std::invalid_argument("keyOut.dtype() != torch::kUInt8 || "
                                    "zippedKey.dtype() != torch::kUInt8");
    }

    std::size_t keyDataSize =
        dcf_get_key_data_size(bitWidthIn, bitWidthOut, 1, elementNum);
    if ((std::size_t)keyOut.numel() != keyDataSize)
    {
        keyOut.resize_({(std::int64_t)keyDataSize});
    }

    void*       keyOutPtr     = keyOut.mutable_data_ptr();
    std::size_t inputDataSize = (std::size_t)keyOut.numel();
    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dcfKeyUnzip(&keyOutPtr,                     //
                                          &inputDataSize,                 //
                                          zippedKey.const_data_ptr(),     //
                                          (std::size_t)zippedKey.numel(), //
                                          bitWidthIn,                     //
                                          bitWidthOut,                    //
                                          1,                              //
                                          elementNum                      //
        );
        if (ret < 0)
        {
            ERR_LOG("FastFss_cpu_dcfKeyUnzip ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_dcfKeyUnzip fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        int ret = FastFss_cuda_dcfKeyUnzip(&keyOutPtr,                     //
                                           &inputDataSize,                 //
                                           zippedKey.const_data_ptr(),     //
                                           (std::size_t)zippedKey.numel(), //
                                           bitWidthIn,                     //
                                           bitWidthOut,                    //
                                           1,                              //
                                           elementNum                      //
        );
        if (ret < 0)
        {
            ERR_LOG("FastFss_cuda_dcfKeyUnzip ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_dcfKeyUnzip fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}
void dcf_key_gen(torch::Tensor keyOut,
                 torch::Tensor alpha,
                 torch::Tensor beta,
                 torch::Tensor seed0,
                 torch::Tensor seed1,
                 std::size_t   bitWidthIn,
                 std::size_t   bitWidthOut,
                 std::size_t   elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    if (!keyOut.is_contiguous() || !alpha.is_contiguous() ||
        !beta.is_contiguous() || !seed0.is_contiguous() ||
        !seed1.is_contiguous())
    {
        ERR_LOG("tensor must be contiguous");
        throw std::invalid_argument("tensor must be contiguous");
    }

    if ((std::size_t)alpha.numel() != elementNum)
    {
        ERR_LOG("alpha.numel() != elementNum");
        throw std::invalid_argument("alpha.numel() != elementNum");
    }
    if ((std::size_t)beta.numel() != 0)
    {
        if ((std::size_t)beta.numel() != elementNum)
        {
            ERR_LOG("beta.numel() != elementNum");
            throw std::invalid_argument("beta.numel() != elementNum");
        }
    }
    if ((std::size_t)seed0.numel() != 16 * elementNum ||
        (std::size_t)seed1.numel() != 16 * elementNum)
    {
        ERR_LOG("seed0.numel() != 16 * elementNum || "
                "seed1.numel() != 16 * elementNum");
        throw std::invalid_argument("seed0.numel() != 16 * elementNum || "
                                    "seed1.numel() != 16 * elementNum");
    }

    if (keyOut.dtype() != torch::kUInt8 || seed0.dtype() != torch::kUInt8 ||
        seed1.dtype() != torch::kUInt8)
    {
        ERR_LOG("keyOut.dtype() != torch::kUInt8 || " //
                "seed0.dtype() != torch::kUInt8 || "  //
                "seed1.dtype() != torch::kUInt8"      //
        );                                            //
        throw std::invalid_argument(                  //
            "keyOut.dtype() != torch::kUInt8 || "     //
            "seed0.dtype() != torch::kUInt8 || "      //
            "seed1.dtype() != torch::kUInt8"          //
        );                                            //
    }
    if (alpha.dtype() != beta.dtype())
    {
        ERR_LOG("alpha.dtype() != beta.dtype()");
        throw std::invalid_argument("alpha.dtype() != beta.dtype()");
    }

    std::size_t elementSize = alpha.element_size();
    if (bitWidthIn > elementSize * 8 || bitWidthOut > elementSize * 8)
    {
        ERR_LOG(                               //
            "bitWidthIn > elementSize * 8 || " //
            "bitWidthOut > elementSize *8"     //
        );                                     //
        throw std::invalid_argument(           //
            "bitWidthIn > elementSize * 8 || " //
            "bitWidthOut > elementSize *8"     //
        );                                     //
    }

    auto device = alpha.device();
    if (keyOut.device() != device || beta.device() != device ||
        seed0.device() != device || seed1.device() != device)
    {
        ERR_LOG("device must be same");
        throw std::invalid_argument("device must be same");
    }

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    std::size_t dcfKeyDataSize = dcf_get_key_data_size(  //
        bitWidthIn, bitWidthOut, elementSize, elementNum //
    );                                                   //

    if ((std::size_t)keyOut.numel() != dcfKeyDataSize)
    {
        keyOut.resize_({(std::int64_t)dcfKeyDataSize});
    }

    void*       dcfKeyPtr    = keyOut.mutable_data_ptr();
    std::size_t inputKeySize = (std::size_t)keyOut.numel();
    if (device.type() == torch::kCPU)
    {
        int ret =
            FastFss_cpu_dcfKeyGen(&dcfKeyPtr,                               //
                                  &inputKeySize,                            //
                                  alpha.const_data_ptr(),                   //
                                  (std::size_t)alpha.numel() * elementSize, //
                                  beta.const_data_ptr(),                    //
                                  (std::size_t)beta.numel() * elementSize,  //
                                  seed0.const_data_ptr(),                   //
                                  (std::size_t)seed0.numel(),               //
                                  seed1.const_data_ptr(),                   //
                                  (std::size_t)seed1.numel(),               //
                                  bitWidthIn,                               //
                                  bitWidthOut,                              //
                                  elementSize,                              //
                                  elementNum                                //
            );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cpu_dcfKeyGen ret = %d", ret);
            throw std::runtime_error("FastFss_cpu_dcfKeyGen fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        int ret =
            FastFss_cuda_dcfKeyGen(&dcfKeyPtr,                               //
                                   &inputKeySize,                            //
                                   alpha.const_data_ptr(),                   //
                                   (std::size_t)alpha.numel() * elementSize, //
                                   beta.const_data_ptr(),                    //
                                   (std::size_t)beta.numel() * elementSize,  //
                                   seed0.const_data_ptr(),                   //
                                   (std::size_t)seed0.numel(),               //
                                   seed1.const_data_ptr(),                   //
                                   (std::size_t)seed1.numel(),               //
                                   bitWidthIn,                               //
                                   bitWidthOut,                              //
                                   elementSize,                              //
                                   elementNum                                //
            );
        if (ret != 0)
        {
            ERR_LOG("FastFss_cuda_dcfKeyGen ret = %d", ret);
            throw std::runtime_error("FastFss_cuda_dcfKeyGen fail");
        }
    }
    else
    {
        ERR_LOG("device must be CPU or CUDA");
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

void dcf_eval(torch::Tensor sharedOut,
              torch::Tensor maskedX,
              torch::Tensor key,
              torch::Tensor seed,
              int           partyId,
              std::size_t   bitWidthIn,
              std::size_t   bitWidthOut,
              std::size_t   elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    if (!sharedOut.is_contiguous() || !maskedX.is_contiguous() ||
        !key.is_contiguous() || !seed.is_contiguous())
    {
        ERR_LOG("tensor must be contiguous");
        throw std::invalid_argument("tensor must be contiguous");
    }

    if ((std::size_t)maskedX.numel() != elementNum)
    {
        ERR_LOG("maskedX.numel() != elementNum");
        throw std::invalid_argument("maskedX.numel() != elementNum");
    }

    if ((std::size_t)seed.numel() != 16 * elementNum)
    {
        std::fprintf(stderr,                                               //
                     "[FastFss] seed.numel() != 16 * elementNum. %s:%d\n", //
                     __FILE__, __LINE__                                    //
        );                                                                 //
        throw std::invalid_argument("seed.numel() != 16 * elementNum");
    }

    if (key.dtype() != torch::kUInt8 || seed.dtype() != torch::kUInt8)
    {
        std::fprintf(stderr,                                                //
                     "[FastFss] key.dtype() != torch::kUInt8 || "           //
                     "seed.dtype() != torch::kUInt8. %s:%d\n",              //
                     __FILE__, __LINE__                                     //
        );                                                                  //
        throw std::invalid_argument(                                        //
            "key.dtype() != torch::kUInt8 || seed.dtype() != torch::kUInt8" //
        );                                                                  //
    }
    if (sharedOut.dtype() != maskedX.dtype())
    {
        std::fprintf(
            stderr,                                                    //
            "[FastFss] sharedOut.dtype() != maskedX.dtype(). %s:%d\n", //
            __FILE__, __LINE__                                         //
        );                                                             //
        throw std::invalid_argument("sharedOut.dtype() != maskedX.dtype()");
    }

    std::size_t elementSize = maskedX.element_size();
    if (bitWidthIn > elementSize * 8 || bitWidthOut > elementSize * 8)
    {
        std::fprintf(stderr,                                      //
                     "[FastFss] bitWidthIn <= elementSize *8 && " //
                     "bitWidthOut <= elementSize *8. %s:%d\n",    //
                     __FILE__, __LINE__                           //
        );                                                        //
        throw std::invalid_argument(                              //
            "bitWidthIn > elementSize * 8 || "                    //
            "bitWidthOut > elementSize *8"                        //
        );                                                        //
    }

    auto device = maskedX.device();
    if (sharedOut.device() != device || key.device() != device ||
        seed.device() != device)
    {
        std::fprintf(stderr,                                   //
                     "[FastFss] device must be same. %s:%d\n", //
                     __FILE__, __LINE__                        //
        );
        throw std::invalid_argument("device must be same");
    }

    if ((std::size_t)key.numel() !=
        dcf_get_key_data_size(bitWidthIn, bitWidthOut, elementSize, elementNum))
    {
        std::fprintf(
            stderr,                                                    //
            "[FastFss] key.numel() != dcf_get_key_data_size. %s:%d\n", //
            __FILE__, __LINE__                                         //
        );                                                             //
        throw std::invalid_argument("key.numel() != dcf_get_key_data_size");
    }

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    if (sharedOut.numel() != maskedX.numel())
    {
        sharedOut.resize_({maskedX.numel()});
    }

    if (device.type() == torch::kCPU)
    {
        int ret =
            FastFss_cpu_dcfEval(sharedOut.mutable_data_ptr(),               //
                                maskedX.const_data_ptr(),                   //
                                (std::size_t)maskedX.numel() * elementSize, //
                                key.const_data_ptr(),                       //
                                (std::size_t)key.numel(),                   //
                                seed.const_data_ptr(),                      //
                                (std::size_t)seed.numel(),                  //
                                partyId,                                    //
                                bitWidthIn,                                 //
                                bitWidthOut,                                //
                                elementSize,                                //
                                elementNum);
        if (ret != 0)
        {
            std::fprintf(stderr,                                            //
                         "[FastFss] FastFss_cpu_dcfEval ret = %d. %s:%d\n", //
                         ret, __FILE__, __LINE__                            //
            );                                                              //
            throw std::runtime_error("FastFss_cpu_dcfEval fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        int ret =
            FastFss_cuda_dcfEval(sharedOut.mutable_data_ptr(),               //
                                 maskedX.const_data_ptr(),                   //
                                 (std::size_t)maskedX.numel() * elementSize, //
                                 key.const_data_ptr(),                       //
                                 (std::size_t)key.numel(),                   //
                                 seed.const_data_ptr(),                      //
                                 (std::size_t)seed.numel(),                  //
                                 partyId,                                    //
                                 bitWidthIn,                                 //
                                 bitWidthOut,                                //
                                 elementSize,                                //
                                 elementNum);
        if (ret != 0)
        {
            std::fprintf(stderr,                                             //
                         "[FastFss] FastFss_cuda_dcfEval ret = %d. %s:%d\n", //
                         ret, __FILE__, __LINE__                             //
            );                                                               //
            throw std::runtime_error("FastFss_cuda_dcfEval fail");
        }
    }
    else
    {
        std::fprintf(                                                     //
            stderr,                                                       //
            "[FastFss] device must be CPU or CUDA. device = %s. %s:%d\n", //
            device.str().c_str(), __FILE__, __LINE__                      //
        );
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

} // namespace pyFastFss
