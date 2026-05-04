#include "pyFastFss.h"
#include "dcf/dcf.h"

namespace pyFastFss {

void bind_dcf(py::module_ &m)
{
    m.def("dcf_get_key_data_size", &dcf_get_key_data_size, py::arg("bitWidthIn"), py::arg("bitWidthOut"),
          py::arg("groupSize"), py::arg("elementSize"), py::arg("elementNum"),
          R"doc((bitWidthIn: int, bitWidthOut: int, groupSize: int, elementSize: int, elementNum: int) -> int)doc");
    m.def(
        "dcf_key_gen", &dcf_key_gen, py::arg("keyOut"), py::arg("alpha"), py::arg("beta"), py::arg("seed0"),
        py::arg("seed1"), py::arg("bitWidthIn"), py::arg("bitWidthOut"), py::arg("groupSize"),
        R"doc((keyOut: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, seed0: torch.Tensor, seed1: torch.Tensor, bitWidthIn: int, bitWidthOut: int, groupSize: int) -> torch.Tensor)doc");
    m.def(
        "dcf_eval", &dcf_eval, py::arg("sharedOut"), py::arg("maskedX"), py::arg("key"), py::arg("seed"),
        py::arg("partyId"), py::arg("bitWidthIn"), py::arg("bitWidthOut"), py::arg("groupSize"),
        R"doc((sharedOut: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, seed: torch.Tensor, partyId: int, bitWidthIn: int, bitWidthOut: int, groupSize: int) -> torch.Tensor)doc");
}

} // namespace pyFastFss
