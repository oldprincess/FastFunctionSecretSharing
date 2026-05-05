#include <FastFss/cpu/config.h>
#include <FastFss/cuda/config.h>

#include "pyFastFss.h"

namespace pyFastFss {

void bind_config(py::module_ &m)
{
    m.def(
        "set_num_threads", [](int num) { FastFss_cpu_setNumThreads(num); }, py::arg("num"),
        R"doc((num: int) -> None)doc");
    m.def(
        "get_num_threads", []() { return FastFss_cpu_getNumThreads(); }, R"doc(() -> int)doc");

#ifdef FAST_FSS_ENABLE_CUDA
    m.def(
        "set_grid_dim", [](int dim) { FastFss_cuda_setGridDim(dim); }, py::arg("dim"),
        R"doc((dim: int) -> None)doc");
    m.def(
        "get_grid_dim", []() { return FastFss_cuda_getGridDim(); }, R"doc(() -> int)doc");
    m.def("set_fine_grain_parallel_grid_dim_threshold",
          [](int dim) { FastFss_cuda_setFineGrainParallelGridDimThreshold(dim); }, py::arg("dim"),
          R"doc((dim: int) -> None)doc");
    m.def("get_fine_grain_parallel_grid_dim_threshold",
          []() { return FastFss_cuda_getFineGrainParallelGridDimThreshold(); }, R"doc(() -> int)doc");
#endif
}

} // namespace pyFastFss
