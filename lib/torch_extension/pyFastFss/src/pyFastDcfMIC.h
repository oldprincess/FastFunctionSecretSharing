#ifndef PY_FAST_DCF_MIC_H
#define PY_FAST_DCF_MIC_H

#include <torch/python.h>
#include <torch/torch.h>

namespace pyFastFss {

// ==========================================
// ==================== MIC =================
// ==========================================

std::size_t dcf_mic_get_key_data_size(std::size_t bitWidthIn,
                                      std::size_t bitWidthOut,
                                      std::size_t elementSize,
                                      std::size_t elementNum);

py::tuple dcf_mic_key_gen(torch::Tensor&       keyOut,
                          torch::Tensor&       zOut,
                          const torch::Tensor& alpha,
                          const torch::Tensor& seed0,
                          const torch::Tensor& seed1,
                          const torch::Tensor& leftBoundary,
                          const torch::Tensor& rightBoundary,
                          std::size_t          bitWidthIn,
                          std::size_t          bitWidthOut,
                          std::size_t          elementNum);

torch::Tensor& dcf_mic_eval(torch::Tensor&       sharedOut,
                            const torch::Tensor& maskedX,
                            const torch::Tensor& key,
                            const torch::Tensor& sharedZ,
                            const torch::Tensor& seed,
                            int                  partyId,
                            const torch::Tensor& leftBoundary,
                            const torch::Tensor& rightBoundary,
                            std::size_t          bitWidthIn,
                            std::size_t          bitWidthOut,
                            std::size_t          elementNum);

}; // namespace pyFastFss

#endif