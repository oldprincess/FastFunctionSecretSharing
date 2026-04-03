#include "pyFastFss.h"

namespace pyFastFss {

void bind_dcf_mic(py::module_ &m)
{
    m.def("dcf_mic_get_key_data_size", &pyFastFss::dcf_mic_get_key_data_size, py::arg("bitWidthIn"),
          py::arg("bitWidthOut"), py::arg("elementSize"), py::arg("elementNum"),
          R"doc((bitWidthIn: int, bitWidthOut: int, elementSize: int, elementNum: int) -> int)doc");
    m.def(
        "dcf_mic_key_gen", &pyFastFss::dcf_mic_key_gen, py::arg("keyOut"), py::arg("zOut"), py::arg("alpha"),
        py::arg("seed0"), py::arg("seed1"), py::arg("leftBoundary"), py::arg("rightBoundary"), py::arg("bitWidthIn"),
        py::arg("bitWidthOut"), py::arg("elementNum"),
        R"doc((keyOut: torch.Tensor, zOut: torch.Tensor, alpha: torch.Tensor, seed0: torch.Tensor, seed1: torch.Tensor, leftBoundary: torch.Tensor, rightBoundary: torch.Tensor, bitWidthIn: int, bitWidthOut: int, elementNum: int) -> tuple[torch.Tensor, torch.Tensor])doc");
    m.def(
        "dcf_mic_eval", &pyFastFss::dcf_mic_eval, py::arg("sharedOut"), py::arg("maskedX"), py::arg("key"),
        py::arg("sharedZ"), py::arg("seed"), py::arg("partyId"), py::arg("leftBoundary"), py::arg("rightBoundary"),
        py::arg("bitWidthIn"), py::arg("bitWidthOut"), py::arg("elementNum"),
        R"doc((sharedOut: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, sharedZ: torch.Tensor, seed: torch.Tensor, partyId: int, leftBoundary: torch.Tensor, rightBoundary: torch.Tensor, bitWidthIn: int, bitWidthOut: int, elementNum: int) -> torch.Tensor)doc");
}

} // namespace pyFastFss
