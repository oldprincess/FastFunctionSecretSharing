#ifndef PY_FAST_FSS_CSRC_GROTTO_GROTTO_H
#define PY_FAST_FSS_CSRC_GROTTO_GROTTO_H

#include <torch/types.h>

#include <tuple>

std::size_t grotto_get_key_data_size(std::size_t bitWidthIn, std::size_t elementSize, std::size_t elementNum);

torch::Tensor& grotto_key_gen(torch::Tensor&       key,
                              const torch::Tensor& alpha,
                              const torch::Tensor& seed0,
                              const torch::Tensor& seed1,
                              std::size_t          bitWidthIn);

torch::Tensor& grotto_eq_eval(torch::Tensor&       out,
                              const torch::Tensor& maskedX,
                              const torch::Tensor& key,
                              const torch::Tensor& seed,
                              int                  partyId,
                              std::size_t          bitWidthIn);

torch::Tensor& grotto_eval(torch::Tensor&       out,
                           const torch::Tensor& maskedX,
                           const torch::Tensor& key,
                           const torch::Tensor& seed,
                           bool                 equalBound,
                           int                  partyId,
                           std::size_t          bitWidthIn);

torch::Tensor& grotto_mic_eval(torch::Tensor&       out,
                               const torch::Tensor& maskedX,
                               const torch::Tensor& key,
                               const torch::Tensor& seed,
                               int                  partyId,
                               const torch::Tensor& leftEndpoints,
                               const torch::Tensor& rightEndpoints,
                               std::size_t          bitWidthIn);

std::tuple<torch::Tensor&, torch::Tensor&> grotto_interval_lut_eval(torch::Tensor&       outE,
                                                                    torch::Tensor&       outT,
                                                                    const torch::Tensor& maskedX,
                                                                    const torch::Tensor& key,
                                                                    const torch::Tensor& seed,
                                                                    int                  partyId,
                                                                    const torch::Tensor& leftEndpoints,
                                                                    const torch::Tensor& rightEndpoints,
                                                                    const torch::Tensor& lookUpTable,
                                                                    std::size_t          bitWidthIn,
                                                                    std::size_t          bitWidthOut);

#endif
