#ifndef PY_FAST_FSS_H
#define PY_FAST_FSS_H

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

std::size_t dcf_get_zipped_key_data_size(std::size_t bitWidthIn,
                                         std::size_t bitWidthOut,
                                         std::size_t elementSize,
                                         std::size_t elementNum);

void dcf_key_zip(torch::Tensor zippedKeyOut,
                 torch::Tensor key,
                 std::size_t   bitWidthIn,
                 std::size_t   bitWidthOut,
                 std::size_t   elementNum);

void dcf_key_unzip(torch::Tensor keyOut,
                   torch::Tensor zippedKey,
                   std::size_t   bitWidthIn,
                   std::size_t   bitWidthOut,
                   std::size_t   elementNum);

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

// ==========================================
// ==================== MIC =================
// ==========================================

std::size_t dcf_mic_get_key_data_size(std::size_t bitWidthIn,
                                      std::size_t bitWidthOut,
                                      std::size_t elementSize,
                                      std::size_t elementNum);

std::size_t dcf_mic_get_zipped_key_data_size(std::size_t bitWidthIn,
                                             std::size_t bitWidthOut,
                                             std::size_t elementSize,
                                             std::size_t elementNum);

void dcf_mic_key_zip(torch::Tensor zippedKeyOut,
                     torch::Tensor key,
                     std::size_t   bitWidthIn,
                     std::size_t   bitWidthOut,
                     std::size_t   elementNum);

void dcf_mic_key_unzip(torch::Tensor keyOut,
                       torch::Tensor zippedKey,
                       std::size_t   bitWidthIn,
                       std::size_t   bitWidthOut,
                       std::size_t   elementNum);

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

// ==========================================
// =================== PRNG =================
// ==========================================

class Prng
{
private:
    void*         ctx_ = nullptr;
    torch::Device device_;

public:
    Prng(torch::Device device = torch::Device("cpu"));
    ~Prng();

public:
    void set_current_seed(py::bytes seed128bit, py::bytes counter128bit);

    py::tuple get_current_seed() const;

public:
    torch::Device device() const;

public:
    void to(torch::Device device);

    void rand(torch::Tensor out, std::size_t bitWidth);
};

}; // namespace pyFastFss

#endif