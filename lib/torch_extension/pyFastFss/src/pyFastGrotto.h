#ifndef PY_FAST_GROTTO_H
#define PY_FAST_GROTTO_H

#include <torch/python.h>
#include <torch/torch.h>

namespace pyFastFss {

// ==========================================
// =================== GROTTO ===============
// ==========================================

std::size_t grotto_get_key_data_size(std::size_t bitWidthIn,
                                     std::size_t elementSize,
                                     std::size_t elementNum);

torch::Tensor& grotto_key_gen(torch::Tensor&       keyOut,
                              const torch::Tensor& alpha,
                              const torch::Tensor& seed0,
                              const torch::Tensor& seed1,
                              std::size_t          bitWidthIn,
                              std::size_t          elementNum);

torch::Tensor& grotto_eq_eval(torch::Tensor&       sharedOut,
                              const torch::Tensor& maskedX,
                              const torch::Tensor& key,
                              const torch::Tensor& seed,
                              int                  partyId,
                              std::size_t          bitWidthIn,
                              std::size_t          elementNum);

torch::Tensor& grotto_eq_multi_eval(torch::Tensor&       sharedOut,
                                    const torch::Tensor& maskedX,
                                    const torch::Tensor& key,
                                    const torch::Tensor& seed,
                                    int                  partyId,
                                    const torch::Tensor& point,
                                    std::size_t          bitWidthIn,
                                    std::size_t          elementNum);

torch::Tensor& grotto_eval(torch::Tensor&       sharedOut,
                           const torch::Tensor& maskedX,
                           const torch::Tensor& key,
                           const torch::Tensor& seed,
                           bool                 equalBound,
                           int                  partyId,
                           std::size_t          bitWidthIn,
                           std::size_t          elementNum);

torch::Tensor& grotto_mic_eval(torch::Tensor&       sharedBooleanOut,
                               const torch::Tensor& maskedX,
                               const torch::Tensor& key,
                               const torch::Tensor& seed,
                               int                  partyId,
                               const torch::Tensor& leftBoundary,
                               const torch::Tensor& rightBoundary,
                               std::size_t          bitWidthIn,
                               std::size_t          elementNum);

py::tuple grotto_lut_eval(torch::Tensor&       sharedOutE,
                          torch::Tensor&       sharedOutT,
                          const torch::Tensor& maskedX,
                          const torch::Tensor& key,
                          const torch::Tensor& seed,
                          int                  partyId,
                          const torch::Tensor& lookUpTable,
                          std::size_t          bitWidthIn,
                          std::size_t          bitWidthOut,
                          std::size_t          elementNum);

py::tuple grotto_lut_eval_ex(torch::Tensor&       sharedOutE,
                             torch::Tensor&       sharedOutT,
                             const torch::Tensor& maskedX,
                             const torch::Tensor& key,
                             const torch::Tensor& seed,
                             int                  partyId,
                             const torch::Tensor& lookUpTable,
                             std::size_t          lutBitWidth,
                             std::size_t          bitWidthIn,
                             std::size_t          bitWidthOut,
                             std::size_t          elementNum,
                             bool                 doubleCache);

py::tuple grotto_lut_eval_ex2(torch::Tensor&       sharedOutE,
                              torch::Tensor&       sharedOutT,
                              const torch::Tensor& maskedX,
                              const torch::Tensor& key,
                              const torch::Tensor& seed,
                              int                  partyId,
                              const torch::Tensor& points,
                              const torch::Tensor& lookUpTable,
                              std::size_t          bitWidthIn,
                              std::size_t          bitWidthOut,
                              std::size_t          elementNum);

py::tuple grotto_interval_lut_eval(torch::Tensor&       sharedOutE,
                                   torch::Tensor&       sharedOutT,
                                   const torch::Tensor& maskedX,
                                   const torch::Tensor& key,
                                   const torch::Tensor& seed,
                                   int                  partyId,
                                   const torch::Tensor& leftBoundary,
                                   const torch::Tensor& rightBoundary,
                                   const torch::Tensor& lookUpTable,
                                   std::size_t          bitWidthIn,
                                   std::size_t          bitWidthOut,
                                   std::size_t          elementNum);

}; // namespace pyFastFss

#endif