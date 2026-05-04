#include "pyFastFss.h"
#include "pyFastPrng.h"

namespace pyFastFss {

void bind_prng(py::module_ &m)
{
    auto prngClass = py::class_<pyFastFss::Prng>(m, "Prng");
    prngClass.def(py::init<torch::Device>(), py::arg("device") = torch::Device("cpu"),
                  R"doc((device: torch.device = torch.device("cpu")) -> Prng)doc");
    prngClass.def("device", &pyFastFss::Prng::device, R"doc(() -> torch.device)doc");
    prngClass.def("get_current_seed", &pyFastFss::Prng::get_current_seed, R"doc(() -> tuple[bytes, bytes])doc");
    prngClass.def("set_current_seed", &pyFastFss::Prng::set_current_seed, py::arg("seed128bit"),
                  py::arg("counter128bit"), R"doc((seed128bit: bytes, counter128bit: bytes) -> None)doc");
    prngClass.def("to_", &pyFastFss::Prng::to_, py::arg("device"), R"doc((device: torch.device) -> Prng)doc");
    prngClass.def("rand_", &pyFastFss::Prng::rand_, py::arg("out"), py::arg("bitWidth"),
                  R"doc((out: torch.Tensor, bitWidth: int) -> torch.Tensor)doc");
}

} // namespace pyFastFss
