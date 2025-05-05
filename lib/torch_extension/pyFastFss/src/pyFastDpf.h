#ifndef PY_FAST_DPF_H
#define PY_FAST_DPF_H

#include <torch/python.h>
#include <torch/torch.h>

namespace pyFastFss {

// ==========================================
// ==================== DPF =================
// ==========================================

std::size_t dpf_get_key_data_size(std::size_t bitWidthIn,
                                  std::size_t bitWidthOut,
                                  std::size_t elementSize,
                                  std::size_t elementNum);

torch::Tensor& dpf_key_gen(torch::Tensor&       keyOut,
                           const torch::Tensor& alpha,
                           const torch::Tensor& beta,
                           const torch::Tensor& seed0,
                           const torch::Tensor& seed1,
                           std::size_t          bitWidthIn,
                           std::size_t          bitWidthOut,
                           std::size_t          elementNum);

torch::Tensor& dpf_eval(torch::Tensor&       sharedOut,
                        const torch::Tensor& maskedX,
                        const torch::Tensor& key,
                        const torch::Tensor& seed,
                        int                  partyId,
                        std::size_t          bitWidthIn,
                        std::size_t          bitWidthOut,
                        std::size_t          elementNum);

torch::Tensor& dpf_eval_multi(torch::Tensor&       sharedOut,
                              const torch::Tensor& maskedX,
                              const torch::Tensor& key,
                              const torch::Tensor& seed,
                              int                  partyId,
                              const torch::Tensor& point,
                              std::size_t          bitWidthIn,
                              std::size_t          bitWidthOut,
                              std::size_t          elementNum);

}; // namespace pyFastFss

#endif