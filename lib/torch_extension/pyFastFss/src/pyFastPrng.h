#ifndef PY_FAST_PRNG_H
#define PY_FAST_PRNG_H

#include <torch/python.h>
#include <torch/torch.h>

namespace pyFastFss {

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
    Prng& to_(torch::Device device);

    torch::Tensor& rand_(torch::Tensor& out, std::size_t bitWidth);
};

}; // namespace pyFastFss

#endif