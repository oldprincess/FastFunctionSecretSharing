#include "pyFastFss.h"
#include "dpf/dpf.h"

namespace pyFastFss {

void bind_dpf(py::module_ &m)
{
    m.def("dpf_get_key_data_size", &dpf_get_key_data_size, py::arg("bitWidthIn"), py::arg("bitWidthOut"),
          py::arg("groupSize"), py::arg("elementSize"), py::arg("elementNum"),
          R"doc((bitWidthIn: int, bitWidthOut: int, groupSize: int, elementSize: int, elementNum: int) -> int)doc");
    m.def(
        "dpf_key_gen", &dpf_key_gen, py::arg("keyOut"), py::arg("alpha"), py::arg("beta"), py::arg("seed0"),
        py::arg("seed1"), py::arg("bitWidthIn"), py::arg("bitWidthOut"), py::arg("groupSize"),
        R"doc((keyOut: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, seed0: torch.Tensor, seed1: torch.Tensor, bitWidthIn: int, bitWidthOut: int, groupSize: int) -> torch.Tensor)doc");
    m.def(
        "dpf_eval", &dpf_eval, py::arg("sharedOut"), py::arg("maskedX"), py::arg("key"), py::arg("seed"),
        py::arg("partyId"), py::arg("bitWidthIn"), py::arg("bitWidthOut"), py::arg("groupSize"),
        R"doc((sharedOut: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, seed: torch.Tensor, partyId: int, bitWidthIn: int, bitWidthOut: int, groupSize: int) -> torch.Tensor)doc");
    m.def(
        "dpf_eval_all", &dpf_eval_all, py::arg("sharedOut"), py::arg("maskedX"), py::arg("key"), py::arg("seed"),
        py::arg("partyId"), py::arg("bitWidthIn"), py::arg("bitWidthOut"), py::arg("groupSize"),
        R"doc((sharedOut: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, seed: torch.Tensor, partyId: int, bitWidthIn: int, bitWidthOut: int, groupSize: int) -> torch.Tensor)doc");
    m.def(
        "dpf_multi_eval", &dpf_eval_multi, py::arg("sharedOut"), py::arg("maskedX"), py::arg("key"), py::arg("seed"),
        py::arg("partyId"), py::arg("point"), py::arg("bitWidthIn"), py::arg("bitWidthOut"), py::arg("groupSize"),
        R"doc((sharedOut: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, seed: torch.Tensor, partyId: int, point: torch.Tensor, bitWidthIn: int, bitWidthOut: int, groupSize: int) -> torch.Tensor)doc");
}

} // namespace pyFastFss
