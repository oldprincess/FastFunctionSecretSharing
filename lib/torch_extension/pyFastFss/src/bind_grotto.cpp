#include "pyFastFss.h"

namespace pyFastFss {

void bind_grotto(py::module_ &m)
{
    m.def("grotto_get_key_data_size", &pyFastFss::grotto_get_key_data_size, py::arg("bitWidthIn"),
          py::arg("elementSize"), py::arg("elementNum"),
          R"doc((bitWidthIn: int, elementSize: int, elementNum: int) -> int)doc");
    m.def(
        "grotto_key_gen", &pyFastFss::grotto_key_gen, py::arg("keyOut"), py::arg("alpha"), py::arg("seed0"),
        py::arg("seed1"), py::arg("bitWidthIn"), py::arg("elementNum"),
        R"doc((keyOut: torch.Tensor, alpha: torch.Tensor, seed0: torch.Tensor, seed1: torch.Tensor, bitWidthIn: int, elementNum: int) -> torch.Tensor)doc");
    m.def(
        "grotto_eval", &pyFastFss::grotto_eval, py::arg("sharedOut"), py::arg("maskedX"), py::arg("key"),
        py::arg("seed"), py::arg("equalBound"), py::arg("partyId"), py::arg("bitWidthIn"), py::arg("elementNum"),
        R"doc((sharedOut: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, seed: torch.Tensor, equalBound: bool, partyId: int, bitWidthIn: int, elementNum: int) -> torch.Tensor)doc");
    m.def(
        "grotto_eq_eval", &pyFastFss::grotto_eq_eval, py::arg("sharedOut"), py::arg("maskedX"), py::arg("key"),
        py::arg("seed"), py::arg("partyId"), py::arg("bitWidthIn"), py::arg("elementNum"),
        R"doc((sharedOut: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, seed: torch.Tensor, partyId: int, bitWidthIn: int, elementNum: int) -> torch.Tensor)doc");
    m.def(
        "grotto_mic_eval", &pyFastFss::grotto_mic_eval, py::arg("sharedOut"), py::arg("maskedX"), py::arg("key"),
        py::arg("seed"), py::arg("partyId"), py::arg("leftBoundary"), py::arg("rightBoundary"), py::arg("bitWidthIn"),
        py::arg("elementNum"),
        R"doc((sharedOut: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, seed: torch.Tensor, partyId: int, leftBoundary: torch.Tensor, rightBoundary: torch.Tensor, bitWidthIn: int, elementNum: int) -> torch.Tensor)doc");
    m.def(
        "grotto_interval_lut_eval", &pyFastFss::grotto_interval_lut_eval, py::arg("sharedOutE"), py::arg("sharedOutT"),
        py::arg("maskedX"), py::arg("key"), py::arg("seed"), py::arg("partyId"), py::arg("leftBoundary"),
        py::arg("rightBoundary"), py::arg("lookUpTable"), py::arg("bitWidthIn"), py::arg("bitWidthOut"),
        py::arg("elementNum"),
        R"doc((sharedOutE: torch.Tensor, sharedOutT: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, seed: torch.Tensor, partyId: int, leftBoundary: torch.Tensor, rightBoundary: torch.Tensor, lookUpTable: torch.Tensor, bitWidthIn: int, bitWidthOut: int, elementNum: int) -> tuple[torch.Tensor, torch.Tensor])doc");
}

} // namespace pyFastFss
