#ifndef PY_FAST_DCF_H
#define PY_FAST_DCF_H

#include <torch/python.h>
#include <torch/torch.h>

namespace pyFastFss {

// ==========================================
// ==================== DCF =================
// ==========================================

std::size_t dcf_get_key_data_size(std::size_t bitWidthIn,
                                  std::size_t bitWidthOut,
                                  std::size_t elementSize,
                                  std::size_t elementNum);

torch::Tensor& dcf_key_gen(torch::Tensor&       keyOut,
                           const torch::Tensor& alpha,
                           const torch::Tensor& beta,
                           const torch::Tensor& seed0,
                           const torch::Tensor& seed1,
                           std::size_t          bitWidthIn,
                           std::size_t          bitWidthOut,
                           std::size_t          elementNum);

torch::Tensor& dcf_eval(torch::Tensor&      sharedOut,
                        const torch::Tensor maskedX,
                        const torch::Tensor key,
                        const torch::Tensor seed,
                        int                 partyId,
                        std::size_t         bitWidthIn,
                        std::size_t         bitWidthOut,
                        std::size_t         elementNum);

}; // namespace pyFastFss

#endif