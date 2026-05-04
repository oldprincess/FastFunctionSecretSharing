#ifndef PY_FAST_FSS_CSRC_DCF_DCF_CPU_H
#define PY_FAST_FSS_CSRC_DCF_DCF_CPU_H

#include <torch/types.h>

torch::Tensor& cpu_dcf_key_gen(torch::Tensor&       key,
                               const torch::Tensor& alpha,
                               const torch::Tensor& beta,
                               const torch::Tensor& seed0,
                               const torch::Tensor& seed1,
                               std::size_t          bitWidthIn,
                               std::size_t          bitWidthOut,
                               std::size_t          groupSize,
                               std::size_t          elementSize,
                               std::size_t          elementNum);

torch::Tensor& cpu_dcf_eval(torch::Tensor&       out,
                            const torch::Tensor& maskedX,
                            const torch::Tensor& key,
                            const torch::Tensor& seed,
                            int                  partyId,
                            std::size_t          bitWidthIn,
                            std::size_t          bitWidthOut,
                            std::size_t          groupSize,
                            std::size_t          elementSize,
                            std::size_t          elementNum);

#endif
