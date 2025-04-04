#include <FastFss/cpu/dcf.h>
#include <FastFss/cpu/mic.h>
#include <FastFss/cuda/dcf.h>
#include <FastFss/cuda/mic.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "pyFastFss.h"

namespace pyFastFss {

std::size_t dcf_mic_get_key_data_size(std::size_t bitWidthIn,
                                      std::size_t bitWidthOut,
                                      std::size_t elementSize,
                                      std::size_t elementNum)
{
    int result = FastFss_cpu_dcfMICGetKeyDataSize(bitWidthIn, bitWidthOut,
                                                  elementSize, elementNum);
    if (result < 0)
    {
        std::fprintf(
            stderr,                                                         //
            "[FastFss] FastFss_cpu_dcfMICGetKeyDataSize ret = %d. %s:%d\n", //
            result, __FILE__, __LINE__                                      //
        );                                                                  //
        throw std::runtime_error("FastFss_cpu_dcfMICGetKeyDataSize fail");
    }
    return (std::size_t)result;
}

std::size_t dcf_mic_get_zipped_key_data_size(std::size_t bitWidthIn,
                                             std::size_t bitWidthOut,
                                             std::size_t elementSize,
                                             std::size_t elementNum)
{
    int result = FastFss_cpu_dcfMICGetZippedKeyDataSize(
        bitWidthIn, bitWidthOut, elementSize, elementNum);
    if (result < 0)
    {
        std::fprintf(
            stderr,                                                       //
            "[FastFss] FastFss_cpu_dcfMICGetZippedKeyDataSize ret = %d. " //
            "%s:%d\n",                                                    //
            result, __FILE__, __LINE__                                    //
        );                                                                //
        throw std::runtime_error("FastFss_cpu_dcfMICGetZippedKeyDataSize fail");
    }
    return (std::size_t)result;
}

void dcf_mic_key_zip(torch::Tensor zippedKeyOut,
                     torch::Tensor key,
                     std::size_t   bitWidthIn,
                     std::size_t   bitWidthOut,
                     std::size_t   elementNum)
{
    auto device = key.device();

    if (zippedKeyOut.device() != device)
    {
        std::fprintf(stderr,                                             //
                     "[FastFss] zippedKeyOut.device() != device. %s:%d", //
                     __FILE__, __LINE__                                  //
        );
        throw std::invalid_argument("zippedKeyOut.device() != device");
    }
    if (zippedKeyOut.dtype() != torch::kUInt8 || key.dtype() != torch::kUInt8)
    {
        std::fprintf(stderr,                                               //
                     "[FastFss] zippedKeyOut.dtype() != torch::kUInt8 || " //
                     "key.dtype() != torch::kUInt8. %s:%d\n",              //
                     __FILE__, __LINE__                                    //
        );
        throw std::invalid_argument("zippedKeyOut.dtype() != torch::kUInt8 || "
                                    "key.dtype() != torch::kUInt8");
    }

    std::size_t zippedKeyDataSize = dcf_mic_get_zipped_key_data_size(
        bitWidthIn, bitWidthOut, 1, elementNum);
    if ((std::size_t)zippedKeyOut.numel() != zippedKeyDataSize)
    {
        zippedKeyOut.resize_({(std::int64_t)zippedKeyDataSize});
    }

    void*       zippedKeyOutPtr = zippedKeyOut.mutable_data_ptr();
    std::size_t inputDataSize   = (std::size_t)zippedKeyOut.numel();
    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dcfMICKeyZip(&zippedKeyOutPtr,         //
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
            std::fprintf(stderr,
                         "[FastFss] FastFss_cpu_dcfMICKeyZip ret = %d. %s:%d\n",
                         ret, __FILE__, __LINE__);
            throw std::runtime_error("FastFss_cpu_dcfMICKeyZip fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        int ret = FastFss_cuda_dcfMICKeyZip(&zippedKeyOutPtr,         //
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
            std::fprintf(
                stderr, "[FastFss] FastFss_cuda_dcfMICKeyZip ret = %d. %s:%d\n",
                ret, __FILE__, __LINE__);
            throw std::runtime_error("FastFss_cuda_dcfMICKeyZip fail");
        }
    }
    else
    {
        std::fprintf(stderr,                                          //
                     "[FastFss] device must be CPU or CUDA. %s:%d\n", //
                     __FILE__, __LINE__                               //
        );
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}

void dcf_mic_key_unzip(torch::Tensor keyOut,
                       torch::Tensor zippedKey,
                       std::size_t   bitWidthIn,
                       std::size_t   bitWidthOut,
                       std::size_t   elementNum)
{
    auto device = zippedKey.device();

    if (keyOut.device() != device)
    {
        std::fprintf(stderr,                                       //
                     "[FastFss] keyOut.device() != device. %s:%d", //
                     __FILE__, __LINE__                            //
        );
        throw std::invalid_argument("keyOut.device() != device");
    }
    if (keyOut.dtype() != torch::kUInt8 || zippedKey.dtype() != torch::kUInt8)
    {
        std::fprintf(stderr,                                         //
                     "[FastFss] keyOut.dtype() != torch::kUInt8 || " //
                     "zippedKey.dtype() != torch::kUInt8. %s:%d\n",  //
                     __FILE__, __LINE__                              //
        );
        throw std::invalid_argument("keyOut.dtype() != torch::kUInt8 || "
                                    "zippedKey.dtype() != torch::kUInt8");
    }

    std::size_t keyDataSize =
        dcf_mic_get_key_data_size(bitWidthIn, bitWidthOut, 1, elementNum);
    if ((std::size_t)keyOut.numel() != keyDataSize)
    {
        keyOut.resize_({(std::int64_t)keyDataSize});
    }

    void*       keyOutPtr     = keyOut.mutable_data_ptr();
    std::size_t inputDataSize = (std::size_t)keyOut.numel();
    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dcfMICKeyUnzip(&keyOutPtr,                     //
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
            std::fprintf(
                stderr,
                "[FastFss] FastFss_cpu_dcfMICKeyUnzip ret = %d. %s:%d\n", ret,
                __FILE__, __LINE__);
            throw std::runtime_error("FastFss_cpu_dcfMICKeyUnzip fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        int ret = FastFss_cuda_dcfMICKeyUnzip(&keyOutPtr,                     //
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
            std::fprintf(
                stderr,
                "[FastFss] FastFss_cuda_dcfMICKeyUnzip ret = %d. %s:%d\n", ret,
                __FILE__, __LINE__);
            throw std::runtime_error("FastFss_cuda_dcfMICKeyUnzip fail");
        }
    }
    else
    {
        std::fprintf(stderr,                                          //
                     "[FastFss] device must be CPU or CUDA. %s:%d\n", //
                     __FILE__, __LINE__                               //
        );
        throw std::invalid_argument("device must be CPU or CUDA");
    }
}
void dcf_mic_key_gen(torch::Tensor keyOut,
                     torch::Tensor zOut,
                     torch::Tensor alpha,
                     torch::Tensor seed0,
                     torch::Tensor seed1,
                     torch::Tensor leftBoundary,
                     torch::Tensor rightBoundary,
                     std::size_t   bitWidthIn,
                     std::size_t   bitWidthOut,
                     std::size_t   elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    if (!keyOut.is_contiguous() || !zOut.is_contiguous() ||
        !alpha.is_contiguous() || !seed0.is_contiguous() ||
        !seed1.is_contiguous() || !leftBoundary.is_contiguous() ||
        !rightBoundary.is_contiguous())
    {
        std::fprintf(stderr,                                         //
                     "[FastFss] tensor must be contiguous. %s:%d\n", //
                     __FILE__, __LINE__);                            //
        throw std::invalid_argument("tensor must be contiguous");
    }

    if ((std::size_t)alpha.numel() != elementNum)
    {
        std::fprintf(stderr,                                            //
                     "[FastFss]  alpha.numel() != elementNum. %s:%d\n", //
                     __FILE__, __LINE__                                 //
        );                                                              //
        throw std::invalid_argument("alpha.numel() != elementNum");
    }

    if ((std::size_t)seed0.numel() != 16 * elementNum ||
        (std::size_t)seed1.numel() != 16 * elementNum)
    {
        std::fprintf(stderr, //
                     "[FastFss] seed0.numel() != 16 * elementNum || "
                     "seed1.numel() != 16 * elementNum. %s:%d\n", //
                     __FILE__, __LINE__                           //
        );                                                        //
        throw std::invalid_argument("seed0.numel() != 16 * elementNum || "
                                    "seed1.numel() != 16 * elementNum");
    }

    if (keyOut.dtype() != torch::kUInt8 || seed0.dtype() != torch::kUInt8 ||
        seed1.dtype() != torch::kUInt8)
    {
        std::fprintf(
            stderr,                                                      //
            "[FastFss] keyOut.dtype seed0.dtype seed1.dtype must be "    //
            "torch::kUInt8. %s:%d\n",                                    //
            __FILE__, __LINE__                                           //
        );                                                               //
        throw std::invalid_argument(                                     //
            "keyOut.dtype seed0.dtype seed1.dtype must be torch::kUInt8" //
        );                                                               //
    }
    if (alpha.dtype() != zOut.dtype() ||
        alpha.dtype() != leftBoundary.dtype() ||
        alpha.dtype() != rightBoundary.dtype())
    {
        std::fprintf(stderr,                                                  //
                     "[FastFss] alpha zOut leftBoundary rightBoundary dtype " //
                     "must be same. %s:%d\n",                                 //
                     __FILE__, __LINE__                                       //
        );                                                                    //
        throw std::invalid_argument(
            "alpha zOut leftBoundary rightBoundary dtype must be same");
    }

    std::size_t elementSize = alpha.element_size();
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

    auto device = alpha.device();
    if (keyOut.device() != device || zOut.device() != device ||
        seed0.device() != device || seed1.device() != device ||
        leftBoundary.device() != device || rightBoundary.device() != device)
    {
        std::fprintf(stderr,                                   //
                     "[FastFss] device must be same. %s:%d\n", //
                     __FILE__, __LINE__                        //
        );
        throw std::invalid_argument("device must be same");
    }

    if (leftBoundary.numel() != rightBoundary.numel())
    {
        std::fprintf(
            stderr,                                                    //
            "[FastFss] intervalNum != rightBoundary.numel(). %s:%d\n", //
            __FILE__, __LINE__                                         //
        );
        throw std::invalid_argument("intervalNum != rightBoundary.numel()");
    }

    std::size_t intervalNum = (std::size_t)leftBoundary.numel();

    if ((std::size_t)zOut.numel() != intervalNum * elementNum)
    {
        zOut.resize_({(std::int64_t)(intervalNum * elementNum)});
    }

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    std::size_t dcfMICKeyDataSize = dcf_mic_get_key_data_size( //
        bitWidthIn, bitWidthOut, elementSize, elementNum       //
    );                                                         //

    if ((std::size_t)keyOut.numel() != dcfMICKeyDataSize)
    {
        keyOut.resize_({(std::int64_t)dcfMICKeyDataSize});
    }

    void*       dcfKeyPtr    = keyOut.mutable_data_ptr();
    std::size_t inputKeySize = (std::size_t)keyOut.numel();
    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dcfMICKeyGen(
            &dcfKeyPtr,                                       //
            &inputKeySize,                                    //
            zOut.mutable_data_ptr(),                          //
            (std::size_t)zOut.numel() * elementSize,          //
            alpha.const_data_ptr(),                           //
            (std::size_t)alpha.numel() * elementSize,         //
            seed0.const_data_ptr(),                           //
            (std::size_t)seed0.numel(),                       //
            seed1.const_data_ptr(),                           //
            (std::size_t)seed1.numel(),                       //
            leftBoundary.const_data_ptr(),                    //
            (std::size_t)leftBoundary.numel() * elementSize,  //
            rightBoundary.const_data_ptr(),                   //
            (std::size_t)rightBoundary.numel() * elementSize, //
            bitWidthIn,                                       //
            bitWidthOut,                                      //
            elementSize,                                      //
            elementNum                                        //
        );
        if (ret != 0)
        {
            std::fprintf(
                stderr,                                                 //
                "[FastFss] FastFss_cpu_dcfMICKeyGen ret = %d. %s:%d\n", //
                ret, __FILE__, __LINE__                                 //
            );                                                          //
            throw std::runtime_error("FastFss_cpu_dcfMICKeyGen fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        int ret = FastFss_cuda_dcfMICKeyGen(
            &dcfKeyPtr,                                       //
            &inputKeySize,                                    //
            zOut.mutable_data_ptr(),                          //
            (std::size_t)zOut.numel() * elementSize,          //
            alpha.const_data_ptr(),                           //
            (std::size_t)alpha.numel() * elementSize,         //
            seed0.const_data_ptr(),                           //
            (std::size_t)seed0.numel(),                       //
            seed1.const_data_ptr(),                           //
            (std::size_t)seed1.numel(),                       //
            leftBoundary.const_data_ptr(),                    //
            (std::size_t)leftBoundary.numel() * elementSize,  //
            rightBoundary.const_data_ptr(),                   //
            (std::size_t)rightBoundary.numel() * elementSize, //
            bitWidthIn,                                       //
            bitWidthOut,                                      //
            elementSize,                                      //
            elementNum                                        //
        );
        if (ret != 0)
        {
            std::fprintf( //
                stderr,
                "[FastFss] FastFss_cuda_dcfMICKeyGen ret = %d. %s:%d\n", //
                ret, __FILE__, __LINE__                                  //
            );                                                           //
            throw std::runtime_error("FastFss_cuda_dcfMICKeyGen fail");
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

void dcf_mic_eval(torch::Tensor sharedOut,
                  torch::Tensor maskedX,
                  torch::Tensor key,
                  torch::Tensor sharedZ,
                  torch::Tensor seed,
                  int           partyId,
                  torch::Tensor leftBoundary,
                  torch::Tensor rightBoundary,
                  std::size_t   bitWidthIn,
                  std::size_t   bitWidthOut,
                  std::size_t   elementNum)
{
    // =====================================================
    // ===================== Check Input ===================
    // =====================================================

    if (!sharedOut.is_contiguous() || !maskedX.is_contiguous() ||
        !key.is_contiguous() || !sharedZ.is_contiguous() ||
        !seed.is_contiguous() || !leftBoundary.is_contiguous() ||
        !rightBoundary.is_contiguous())
    {
        std::fprintf(stderr,                                         //
                     "[FastFss] tensor must be contiguous. %s:%d\n", //
                     __FILE__, __LINE__);                            //
        throw std::invalid_argument("tensor must be contiguous");
    }

    if ((std::size_t)maskedX.numel() != elementNum)
    {
        std::fprintf(stderr,                                              //
                     "[FastFss]  maskedX.numel() != elementNum. %s:%d\n", //
                     __FILE__, __LINE__                                   //
        );                                                                //
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
        std::fprintf(
            stderr,                                                          //
            "[FastFss] key.dtype seed.dtype must be torch::kUInt8. %s:%d\n", //
            __FILE__, __LINE__                                               //
        );                                                                   //
        throw std::invalid_argument(                                         //
            "key.dtype seed.dtype must be torch::kUInt8"                     //
        );                                                                   //
    }

    if (maskedX.dtype() != sharedOut.dtype() ||
        maskedX.dtype() != sharedZ.dtype() ||
        maskedX.dtype() != leftBoundary.dtype() ||
        maskedX.dtype() != rightBoundary.dtype())
    {
        std::fprintf(stderr,                                             //
                     "[FastFss] maskedX sharedOut sharedZ leftBoundary " //
                     "rightBoundary dtype "                              //
                     "must be same. %s:%d\n",                            //
                     __FILE__, __LINE__                                  //
        );                                                               //
        throw std::invalid_argument("maskedX sharedOut sharedZ leftBoundary "
                                    "rightBoundary dtype must be same");
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
        sharedZ.device() != device || seed.device() != device ||
        leftBoundary.device() != device || rightBoundary.device() != device)
    {
        std::fprintf(stderr,                                   //
                     "[FastFss] device must be same. %s:%d\n", //
                     __FILE__, __LINE__                        //
        );
        throw std::invalid_argument("device must be same");
    }

    if (leftBoundary.numel() != rightBoundary.numel())
    {
        std::fprintf(
            stderr,                                                    //
            "[FastFss] intervalNum != rightBoundary.numel(). %s:%d\n", //
            __FILE__, __LINE__                                         //
        );
        throw std::invalid_argument("intervalNum != rightBoundary.numel()");
    }

    std::size_t intervalNum = (std::size_t)leftBoundary.numel();

    if ((std::size_t)sharedZ.numel() != intervalNum * elementNum)
    {
        std::fprintf(
            stderr,                                                           //
            "[FastFss] sharedZ.numel() != intervalNum * elementNum. %s:%d\n", //
            __FILE__, __LINE__                                                //
        );
        throw std::invalid_argument(                      //
            "sharedZ.numel() != intervalNum * elementNum" //
        );                                                //
    }

    if ((std::size_t)key.numel() !=
        dcf_mic_get_key_data_size(bitWidthIn, bitWidthOut, elementSize,
                                  elementNum))
    {
        std::fprintf(
            stderr,                                                        //
            "[FastFss] key.numel() != dcf_mic_get_key_data_size. %s:%d\n", //
            __FILE__, __LINE__                                             //
        );
        throw std::invalid_argument(                   //
            "key.numel() != dcf_mic_get_key_data_size" //
        );
    }

    // =====================================================
    // ===================== FastFss =======================
    // =====================================================

    if ((std::size_t)sharedOut.numel() != intervalNum * elementNum)
    {
        sharedOut.resize_({(std::int64_t)(intervalNum * elementNum)});
    }

    if (device.type() == torch::kCPU)
    {
        int ret = FastFss_cpu_dcfMICEval(                     //
            sharedOut.mutable_data_ptr(),                     //
            (std::size_t)sharedOut.numel() * elementSize,     //
            maskedX.const_data_ptr(),                         //
            (std::size_t)maskedX.numel() * elementSize,       //
            key.const_data_ptr(),                             //
            (std::size_t)key.numel(),                         //
            sharedZ.const_data_ptr(),                         //
            (std::size_t)sharedZ.numel() * elementSize,       //
            seed.const_data_ptr(),                            //
            (std::size_t)seed.numel(),                        //
            partyId,                                          //
            leftBoundary.const_data_ptr(),                    //
            (std::size_t)leftBoundary.numel() * elementSize,  //
            rightBoundary.const_data_ptr(),                   //
            (std::size_t)rightBoundary.numel() * elementSize, //
            bitWidthIn,                                       //
            bitWidthOut,                                      //
            elementSize,                                      //
            elementNum);
        if (ret != 0)
        {
            std::fprintf(
                stderr,                                               //
                "[FastFss] FastFss_cpu_dcfMICEval ret = %d. %s:%d\n", //
                ret, __FILE__, __LINE__                               //
            );                                                        //
            throw std::runtime_error("FastFss_cpu_dcfMICEval fail");
        }
    }
    else if (device.type() == torch::kCUDA)
    {
        int ret = FastFss_cuda_dcfMICEval(                    //
            sharedOut.mutable_data_ptr(),                     //
            (std::size_t)sharedOut.numel() * elementSize,     //
            maskedX.const_data_ptr(),                         //
            (std::size_t)maskedX.numel() * elementSize,       //
            key.const_data_ptr(),                             //
            (std::size_t)key.numel(),                         //
            sharedZ.const_data_ptr(),                         //
            (std::size_t)sharedZ.numel() * elementSize,       //
            seed.const_data_ptr(),                            //
            (std::size_t)seed.numel(),                        //
            partyId,                                          //
            leftBoundary.const_data_ptr(),                    //
            (std::size_t)leftBoundary.numel() * elementSize,  //
            rightBoundary.const_data_ptr(),                   //
            (std::size_t)rightBoundary.numel() * elementSize, //
            bitWidthIn,                                       //
            bitWidthOut,                                      //
            elementSize,                                      //
            elementNum);
        if (ret != 0)
        {
            std::fprintf(
                stderr,                                                //
                "[FastFss] FastFss_cuda_dcfMICEval ret = %d. %s:%d\n", //
                ret, __FILE__, __LINE__                                //
            );                                                         //
            throw std::runtime_error("FastFss_cuda_dcfMICEval fail");
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
