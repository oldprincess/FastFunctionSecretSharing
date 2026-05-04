#ifndef PY_FAST_FSS_CSRC_MIC_MIC_H
#define PY_FAST_FSS_CSRC_MIC_MIC_H

#include <torch/types.h>

#include <tuple>

std::size_t dcf_mic_get_key_data_size(std::size_t bitWidthIn,
                                      std::size_t bitWidthOut,
                                      std::size_t elementSize,
                                      std::size_t elementNum);

std::tuple<torch::Tensor&, torch::Tensor&> dcf_mic_key_gen(torch::Tensor&       key,
                                                           torch::Tensor&       z,
                                                           const torch::Tensor& alpha,
                                                           const torch::Tensor& seed0,
                                                           const torch::Tensor& seed1,
                                                           const torch::Tensor& leftEndpoints,
                                                           const torch::Tensor& rightEndpoints,
                                                           std::size_t          bitWidthIn,
                                                           std::size_t          bitWidthOut);

torch::Tensor& dcf_mic_eval(torch::Tensor&       out,
                            const torch::Tensor& maskedX,
                            const torch::Tensor& key,
                            const torch::Tensor& sharedZ,
                            const torch::Tensor& seed,
                            int                  partyId,
                            const torch::Tensor& leftEndpoints,
                            const torch::Tensor& rightEndpoints,
                            std::size_t          bitWidthIn,
                            std::size_t          bitWidthOut);

#endif
