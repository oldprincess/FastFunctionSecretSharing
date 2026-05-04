#include <FastFss/cpu/config.h>

#include "pyFastFss.h"

namespace pyFastFss {

void bind_config(py::module_ &m)
{
    m.def(
        "set_num_threads", [](int num) { FastFss_cpu_setNumThreads(num); }, py::arg("num"),
        R"doc((num: int) -> None)doc");
    m.def(
        "get_num_threads", []() { return FastFss_cpu_getNumThreads(); }, R"doc(() -> int)doc");
}

} // namespace pyFastFss
