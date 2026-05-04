#include "pyFastFss.h"
#include "spline/spline.h"

namespace pyFastFss {

void bind_spline(py::module_& m)
{
    m.def("spline_get_key_data_size", &spline_get_key_data_size, py::arg("degree"), py::arg("intervalNum"),
          py::arg("bitWidthIn"), py::arg("bitWidthOut"), py::arg("elementSize"), py::arg("elementNum"),
          R"doc((degree: int, intervalNum: int, bitWidthIn: int, bitWidthOut: int, elementSize: int, elementNum: int) -> int)doc");
    m.def("spline_key_gen", &spline_key_gen, py::arg("keyOut"), py::arg("eOut"), py::arg("betaOut"),
          py::arg("alpha"), py::arg("seed0"), py::arg("seed1"), py::arg("coefficients"), py::arg("degree"),
          py::arg("leftEndpoints"), py::arg("rightEndpoints"), py::arg("intervalNum"), py::arg("bitWidthIn"),
          py::arg("bitWidthOut"),
          R"doc((keyOut: torch.Tensor, eOut: torch.Tensor, betaOut: torch.Tensor, alpha: torch.Tensor, seed0: torch.Tensor, seed1: torch.Tensor, coefficients: torch.Tensor, degree: int, leftEndpoints: torch.Tensor, rightEndpoints: torch.Tensor, intervalNum: int, bitWidthIn: int, bitWidthOut: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor])doc");
    m.def("spline_eval", &spline_eval, py::arg("sharedOut"), py::arg("maskedX"), py::arg("key"), py::arg("sharedE"),
          py::arg("sharedBeta"), py::arg("seed"), py::arg("partyId"), py::arg("leftEndpoints"),
          py::arg("rightEndpoints"), py::arg("intervalNum"), py::arg("degree"), py::arg("bitWidthIn"),
          py::arg("bitWidthOut"),
          R"doc((sharedOut: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, sharedE: torch.Tensor, sharedBeta: torch.Tensor, seed: torch.Tensor, partyId: int, leftEndpoints: torch.Tensor, rightEndpoints: torch.Tensor, degree: int, intervalNum: int, bitWidthIn: int, bitWidthOut: int) -> torch.Tensor)doc");
}

} // namespace pyFastFss
