#include "pyFastFss.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pyFastFss::bind_config(m);
    pyFastFss::bind_dcf(m);
    pyFastFss::bind_dpf(m);
    pyFastFss::bind_grotto(m);
    pyFastFss::bind_dcf_mic(m);
    pyFastFss::bind_ottt(m);
    pyFastFss::bind_spline(m);
    pyFastFss::bind_prng(m);
}
