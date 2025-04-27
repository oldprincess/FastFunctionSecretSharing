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

void dcf_mic_key_gen(torch::Tensor keyOut,
                     torch::Tensor zOut,
                     torch::Tensor alpha,
                     torch::Tensor seed0,
                     torch::Tensor seed1,
                     torch::Tensor leftBoundary,
                     torch::Tensor rightBoundary,
                     std::size_t   bitWidthIn,
                     std::size_t   bitWidthOut,
                     std::size_t   elementNum);

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
                  std::size_t   elementNum);

}; // namespace pyFastFss

#endif