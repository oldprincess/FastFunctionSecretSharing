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

void dcf_key_gen(torch::Tensor keyOut,
                 torch::Tensor alpha,
                 torch::Tensor beta,
                 torch::Tensor seed0,
                 torch::Tensor seed1,
                 std::size_t   bitWidthIn,
                 std::size_t   bitWidthOut,
                 std::size_t   elementNum);

void dcf_eval(torch::Tensor sharedOut,
              torch::Tensor maskedX,
              torch::Tensor key,
              torch::Tensor seed,
              int           partyId,
              std::size_t   bitWidthIn,
              std::size_t   bitWidthOut,
              std::size_t   elementNum);

}; // namespace pyFastFss

#endif