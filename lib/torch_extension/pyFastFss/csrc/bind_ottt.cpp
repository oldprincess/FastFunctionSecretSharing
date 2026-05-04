#include "pyFastFss.h"
#include "ottt/ottt.h"

namespace pyFastFss {

void bind_ottt(py::module_ &m)
{
    m.def("ottt_get_key_data_size", &ottt_get_key_data_size, py::arg("bitWidthIn"), py::arg("elementNum"),
          R"doc((bitWidthIn: int, elementNum: int) -> int)doc");
    m.def("ottt_key_gen", &ottt_key_gen, py::arg("keyInOut"), py::arg("alpha"), py::arg("bitWidthIn"),
          R"doc((keyInOut: torch.Tensor, alpha: torch.Tensor, bitWidthIn: int) -> torch.Tensor)doc");
    m.def(
        "ottt_lut_eval", &ottt_lut_eval, py::arg("sharedOutE"), py::arg("sharedOutT"), py::arg("maskedX"),
        py::arg("key"), py::arg("partyId"), py::arg("lookUpTable"), py::arg("bitWidthIn"), py::arg("bitWidthOut"),
        R"doc((sharedOutE: torch.Tensor, sharedOutT: torch.Tensor, maskedX: torch.Tensor, key: torch.Tensor, partyId: int, lookUpTable: torch.Tensor, bitWidthIn: int, bitWidthOut: int) -> tuple[torch.Tensor, torch.Tensor])doc");
}

} // namespace pyFastFss
