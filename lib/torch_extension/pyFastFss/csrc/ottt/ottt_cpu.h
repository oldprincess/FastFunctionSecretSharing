#ifndef PY_FAST_FSS_CSRC_OTTT_OTTT_CPU_H
#define PY_FAST_FSS_CSRC_OTTT_OTTT_CPU_H

#include <torch/types.h>

#include <tuple>

torch::Tensor& cpu_ottt_key_gen(torch::Tensor&       key,
                                const torch::Tensor& alpha,
                                std::size_t          bitWidthIn,
                                std::size_t          elementSize,
                                std::size_t          elementNum);

std::tuple<torch::Tensor&, torch::Tensor&> cpu_ottt_lut_eval(torch::Tensor&       outE,
                                                             torch::Tensor&       outT,
                                                             const torch::Tensor& maskedX,
                                                             const torch::Tensor& key,
                                                             int                  partyId,
                                                             const torch::Tensor& lookUpTable,
                                                             std::size_t          bitWidthIn,
                                                             std::size_t          elementSize,
                                                             std::size_t          elementNum);

#endif
