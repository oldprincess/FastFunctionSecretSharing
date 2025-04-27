#ifndef PY_FAST_ONEHOT_H
#define PY_FAST_ONEHOT_H

#include <torch/python.h>
#include <torch/torch.h>

namespace pyFastFss {

// ==========================================
// ==================== ONEHOT ==============
// ==========================================

std::size_t onehot_get_key_data_size(std::size_t bitWidthIn,
                                     std::size_t elementNum);

torch::Tensor& onehot_key_gen(torch::Tensor&       keyInOut,
                              const torch::Tensor& alpha,
                              std::size_t          bitWidthIn,
                              std::size_t          elementNum);

py::tuple onehot_lut_eval(torch::Tensor&       sharedOutE,
                          torch::Tensor&       sharedOutT,
                          const torch::Tensor& maskedX,
                          const torch::Tensor& key,
                          int                  partyId,
                          const torch::Tensor& lookUpTable,
                          std::size_t          bitWidthIn,
                          std::size_t          bitWidthOut,
                          std::size_t          elementNum);

}; // namespace pyFastFss

#endif