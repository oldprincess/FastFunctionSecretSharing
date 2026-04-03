#ifndef PY_FAST_FSS_H
#define PY_FAST_FSS_H

#include <torch/extension.h>

#include "pyFastDcf.h"
#include "pyFastDcfMIC.h"
#include "pyFastDpf.h"
#include "pyFastGrotto.h"
#include "pyFastOttt.h"
#include "pyFastPrng.h"

namespace pyFastFss {

void bind_config(py::module_ &m);
void bind_dcf(py::module_ &m);
void bind_dpf(py::module_ &m);
void bind_grotto(py::module_ &m);
void bind_dcf_mic(py::module_ &m);
void bind_ottt(py::module_ &m);
void bind_prng(py::module_ &m);

} // namespace pyFastFss

#endif
