#ifndef PY_FAST_FSS_CSRC_SPLINE_SPLINE_H
#define PY_FAST_FSS_CSRC_SPLINE_SPLINE_H

#include <torch/types.h>

#include <tuple>

std::size_t spline_get_key_data_size(std::size_t degree,
                                     std::size_t intervalNum,
                                     std::size_t bitWidthIn,
                                     std::size_t bitWidthOut,
                                     std::size_t elementSize,
                                     std::size_t elementNum);

std::tuple<torch::Tensor&, torch::Tensor&, torch::Tensor&> spline_key_gen(torch::Tensor&       key,
                                                                          torch::Tensor&       e,
                                                                          torch::Tensor&       beta,
                                                                          const torch::Tensor& alpha,
                                                                          const torch::Tensor& seed0,
                                                                          const torch::Tensor& seed1,
                                                                          const torch::Tensor& coefficients,
                                                                          std::size_t          degree,
                                                                          const torch::Tensor& leftEndpoints,
                                                                          const torch::Tensor& rightEndpoints,
                                                                          std::size_t          intervalNum,
                                                                          std::size_t          bitWidthIn,
                                                                          std::size_t          bitWidthOut);

torch::Tensor& spline_eval(torch::Tensor&       out,
                           const torch::Tensor& maskedX,
                           const torch::Tensor& key,
                           const torch::Tensor& sharedE,
                           const torch::Tensor& sharedBeta,
                           const torch::Tensor& seed,
                           int                  partyId,
                           const torch::Tensor& leftEndpoints,
                           const torch::Tensor& rightEndpoints,
                           std::size_t          intervalNum,
                           std::size_t          degree,
                           std::size_t          bitWidthIn,
                           std::size_t          bitWidthOut);

#endif
