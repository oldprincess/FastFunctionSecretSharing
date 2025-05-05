#include "pyFastFss.h"

#include <FastFss/cpu/config.h>
#include <torch/extension.h>

// ======================================================
// ======================================================
// ======================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "set_num_threads", [](int num) { FastFss_cpu_setNumThreads(num); },
        py::arg("num"),
        R"(
Args:
    num (int):   number of threads

Returns:
    None
)");

    m.def(
        "get_num_threads", []() { return FastFss_cpu_getNumThreads(); },
        R"(
Returns:
    int: number of threads
)");

    // ==========================================
    // ==================== DCF =================
    // ==========================================

    m.def("dcf_get_key_data_size",           //
          &pyFastFss::dcf_get_key_data_size, //
          py::arg("bitWidthIn"),             //
          py::arg("bitWidthOut"),            //
          py::arg("elementSize"),            //
          py::arg("elementNum"),             //
          R"(
Get DCF key data size.

Args:
    bitWidthIn (int):       bit width of input data
    bitWidthOut (int):      bit width of output data
    elementSize (int):      element size of input data
    elementNum (int):       element number of input data

Returns:
    int:             DCF key data size

Raises:
    ValueError:   If the input argument is invalid.
        )");

    m.def(                       //
        "dcf_key_gen",           //
        &pyFastFss::dcf_key_gen, //
        py::arg("keyOut"),       //
        py::arg("alpha"),        //
        py::arg("beta"),         //
        py::arg("seed0"),        //
        py::arg("seed1"),        //
        py::arg("bitWidthIn"),   //
        py::arg("bitWidthOut"),  //
        py::arg("elementNum"),   //
        R"(
Generate DCF key.

Args:
    keyOut (torch.Tensor):  DCF key tensor
    alpha (torch.Tensor):   alpha tensor
    beta (torch.Tensor):    beta tensor
    seed0 (torch.Tensor):   seed0 tensor
    seed1 (torch.Tensor):   seed1 tensor
    bitWidthIn (int):       bit width of input data
    bitWidthOut (int):      bit width of output data
    elementSize (int):      element size of input data
    elementNum (int):       element number of input data

Returns:
    torch.Tensor: keyOut

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_dcfKeyGen or FastFss_cuda_dcfKeyGen fail.
        )");

    m.def("dcf_eval",             //
          &pyFastFss::dcf_eval,   //
          py::arg("sharedOut"),   //
          py::arg("maskedX"),     //
          py::arg("key"),         //
          py::arg("seed"),        //
          py::arg("partyId"),     //
          py::arg("bitWidthIn"),  //
          py::arg("bitWidthOut"), //
          py::arg("elementNum"),  //
          R"(
Evaluate DCF.

Args:
    sharedOut (torch.Tensor):  shared tensor
    maskedX (torch.Tensor):    masked tensor
    key (torch.Tensor):        key tensor
    seed (torch.Tensor):       seed tensor
    partyId (int):             party id
    bitWidthIn (int):          bit width of input data
    bitWidthOut (int):         bit width of output data
    elementNum (int):          element number of input data
    
Returns:
    torch.Tensor: sharedOut

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_dcfEval or FastFss_cuda_dcfEval fail.
        )");

    // ==========================================
    // ==================== DPF =================
    // ==========================================

    m.def("dpf_get_key_data_size",           //
          &pyFastFss::dpf_get_key_data_size, //
          py::arg("bitWidthIn"),             //
          py::arg("bitWidthOut"),            //
          py::arg("elementSize"),            //
          py::arg("elementNum"),             //
          R"(
Get DPF key data size.

Args:
  bitWidthIn (int):       bit width of input data
  bitWidthOut (int):      bit width of output data
  elementSize (int):      element size of input data
  elementNum (int):       element number of input data

Returns:
  int:             DPF key data size

Raises:
  ValueError:   If the input argument is invalid.
      )");

    m.def(                       //
        "dpf_key_gen",           //
        &pyFastFss::dpf_key_gen, //
        py::arg("keyOut"),       //
        py::arg("alpha"),        //
        py::arg("beta"),         //
        py::arg("seed0"),        //
        py::arg("seed1"),        //
        py::arg("bitWidthIn"),   //
        py::arg("bitWidthOut"),  //
        py::arg("elementNum"),   //
        R"(
Generate DCF key.

Args:
  keyOut (torch.Tensor):  DPF key tensor
  alpha (torch.Tensor):   alpha tensor
  beta (torch.Tensor):    beta tensor
  seed0 (torch.Tensor):   seed0 tensor
  seed1 (torch.Tensor):   seed1 tensor
  bitWidthIn (int):       bit width of input data
  bitWidthOut (int):      bit width of output data
  elementSize (int):      element size of input data
  elementNum (int):       element number of input data

Returns:
  torch.Tensor: keyOut

Raises:
  ValueError:   If the input argument is invalid.
  RuntimeError: If the FastFss_cpu_dpfKeyGen or FastFss_cuda_dpfKeyGen fail.
      )");

    m.def("dpf_eval",             //
          &pyFastFss::dpf_eval,   //
          py::arg("sharedOut"),   //
          py::arg("maskedX"),     //
          py::arg("key"),         //
          py::arg("seed"),        //
          py::arg("partyId"),     //
          py::arg("bitWidthIn"),  //
          py::arg("bitWidthOut"), //
          py::arg("elementNum"),  //
          R"(
Evaluate DCF.

Args:
  sharedOut (torch.Tensor):  shared tensor
  maskedX (torch.Tensor):    masked tensor
  key (torch.Tensor):        key tensor
  seed (torch.Tensor):       seed tensor
  partyId (int):             party id
  bitWidthIn (int):          bit width of input data
  bitWidthOut (int):         bit width of output data
  elementNum (int):          element number of input data
  
Returns:
  torch.Tensor: sharedOut

Raises:
  ValueError:   If the input argument is invalid.
  RuntimeError: If the FastFss_cpu_dpfEval or FastFss_cuda_dpfEval fail.
      )");

    m.def("dpf_eval_multi",           //
          &pyFastFss::dpf_eval_multi, //
          py::arg("sharedOut"),       //
          py::arg("maskedX"),         //
          py::arg("key"),             //
          py::arg("seed"),            //
          py::arg("partyId"),         //
          py::arg("point"),           //
          py::arg("bitWidthIn"),      //
          py::arg("bitWidthOut"),     //
          py::arg("elementNum"),      //
          R"(
Evaluate DCF.

Args:
  sharedOut (torch.Tensor):  shared tensor
  maskedX (torch.Tensor):    masked tensor
  key (torch.Tensor):        key tensor
  seed (torch.Tensor):       seed tensor
  partyId (int):             party id
  point (torch.Tensor):      point list
  bitWidthIn (int):          bit width of input data
  bitWidthOut (int):         bit width of output data
  elementNum (int):          element number of input data
  
Returns:
  torch.Tensor: sharedOut

Raises:
  ValueError:   If the input argument is invalid.
  RuntimeError: If the FastFss_cpu_dpfEvalMulti or FastFss_cuda_dpfEvalMulti fail.
      )");

    // ===========================================
    // ================== GROTTO =================
    // ===========================================

    m.def("grotto_get_key_data_size",           //
          &pyFastFss::grotto_get_key_data_size, //
          py::arg("bitWidthIn"),                //
          py::arg("elementSize"),               //
          py::arg("elementNum"),                //
          R"(
Get Grotto key data size.

Args:
    bitWidthIn (int):      bit width of input data
    elementSize (int):     element size of input data
    elementNum (int):      element number of input data

Returns:
    int: key data size

Raises:
    ValueError:   If the input argument is invalid.
        )");

    m.def("grotto_key_gen",           //
          &pyFastFss::grotto_key_gen, //
          py::arg("keyOut"),          //
          py::arg("alpha"),           //
          py::arg("seed0"),           //
          py::arg("seed1"),           //
          py::arg("bitWidthIn"),      //
          py::arg("elementNum"),      //
          R"(
Generate Grotto key.

Args:
    keyOut (torch.Tensor):  Grotto key tensor
    alpha (torch.Tensor):   alpha tensor
    seed0 (torch.Tensor):   seed0 tensor
    seed1 (torch.Tensor):   seed1 tensor
    bitWidthIn (int):       bit width of input data
    elementNum (int):       element number of input data
    
Returns:
    torch.Tensor: keyOut

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_grottoKeyGen or FastFss_cuda_grottoKeyGen fail.
        )");

    m.def("grotto_eval",           //
          &pyFastFss::grotto_eval, //
          py::arg("sharedOut"),    //
          py::arg("maskedX"),      //
          py::arg("key"),          //
          py::arg("seed"),         //
          py::arg("equalBound"),   //
          py::arg("partyId"),      //
          py::arg("bitWidthIn"),   //
          py::arg("elementNum"),   //
          R"(
Evaluate Grotto.

Args:
    sharedOut (torch.Tensor):       shared tensor
    maskedX (torch.Tensor):         masked tensor
    key (torch.Tensor):             key tensor
    seed (torch.Tensor):            seed tensor
    equalBound (bool):              equalBound
    partyId (int):                  party id
    bitWidthIn (int):               bit width of input data
    elementNum (int):               element number of input data
    
Returns:
    torch.Tensor: sharedOut

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_grottoEval or FastFss_cuda_grottoEval fail.
)");

    m.def("grotto_eval_eq",           //
          &pyFastFss::grotto_eval_eq, //
          py::arg("sharedOut"),       //
          py::arg("maskedX"),         //
          py::arg("key"),             //
          py::arg("seed"),            //
          py::arg("partyId"),         //
          py::arg("bitWidthIn"),      //
          py::arg("elementNum"),      //
          R"(
Evaluate Grotto.

Args:
    sharedOut (torch.Tensor):       shared tensor
    maskedX (torch.Tensor):         masked tensor
    key (torch.Tensor):             key tensor
    seed (torch.Tensor):            seed tensor
    partyId (int):                  party id
    bitWidthIn (int):               bit width of input data
    elementNum (int):               element number of input data

Returns:
    torch.Tensor: sharedOut

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_grottoEvalEq or FastFss_cuda_grottoEvalEq fail.
)");

    m.def("grotto_mic_eval",           //
          &pyFastFss::grotto_mic_eval, //
          py::arg("sharedOut"),        //
          py::arg("maskedX"),          //
          py::arg("key"),              //
          py::arg("seed"),             //
          py::arg("partyId"),          //
          py::arg("leftBoundary"),     //
          py::arg("rightBoundary"),    //
          py::arg("bitWidthIn"),       //
          py::arg("elementNum"),       //
          R"(
Evaluate Grotto MIC.

Args:
    sharedOut (torch.Tensor):       shared tensor
    maskedX (torch.Tensor):         masked tensor
    key(torch.Tensor):              key tensor
    seed(torch.Tensor):             seed tensor
    partyId(int):                   party id
    leftBoundary(torch.Tensor):     leftBoundary tensor
    rightBoundary(torch.Tensor):    rightBoundary tensor
    bitWidthIn(int):                bit width of input data
    elementNum(int):                element number of input data

Returns:
    torch.Tensor: sharedOut

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_grottoMICEval or FastFss_cuda_grottoMICEval fail.
)");

    m.def("grotto_interval_lut_eval",
          &pyFastFss::grotto_interval_lut_eval, //
          py::arg("sharedOutE"),                //
          py::arg("sharedOutT"),                //
          py::arg("maskedX"),                   //
          py::arg("key"),                       //
          py::arg("seed"),                      //
          py::arg("partyId"),                   //
          py::arg("leftBoundary"),              //
          py::arg("rightBoundary"),             //
          py::arg("lookUpTable"),               //
          py::arg("bitWidthIn"),                //
          py::arg("bitWidthOut"),               //
          py::arg("elementNum"),                //
          R"(
Evaluate Grotto MIC.

Args:
    sharedOutE (torch.Tensor):      shared tensor
    sharedOutT (torch.Tensor):      shared tensor
    maskedX (torch.Tensor):         masked tensor
    key(torch.Tensor):              key tensor
    seed(torch.Tensor):             seed tensor
    partyId(int):                   party id
    leftBoundary(torch.Tensor):     leftBoundary tensor
    rightBoundary(torch.Tensor):    rightBoundary tensor
    lookUpTable(torch.Tensor):      lookUpTable tensor
    bitWidthIn(int):                bit width of input data
    bitWidthOut(int):               bit width of output data
    elementNum(int):                element number of input data

Returns:
    tuple[torch.Tensor, torch.Tensor]: sharedOutE, sharedOutT

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_grottoIntervalLutEval or FastFss_cuda_grottoIntervalLutEval fail.
)");

    // ===========================================
    // ================== DCF MIC ================
    // ===========================================

    m.def("dcf_mic_get_key_data_size",           //
          &pyFastFss::dcf_mic_get_key_data_size, //
          py::arg("bitWidthIn"),                 //
          py::arg("bitWidthOut"),                //
          py::arg("elementSize"),                //
          py::arg("elementNum"),                 //
          R"(
Get DCF MIC key data size.

Args:
    bitWidthIn (int):      bit width of input data
    bitWidthOut (int):     bit width of output data
    elementSize (int):     element size of input data
    elementNum (int):      element number of input data

Returns:
    int: key data size

Raises:
    ValueError:   If the input argument is invalid.
        )");

    m.def("dcf_mic_key_gen",           //
          &pyFastFss::dcf_mic_key_gen, //
          py::arg("keyOut"),           //
          py::arg("zOut"),             //
          py::arg("alpha"),            //
          py::arg("seed0"),            //
          py::arg("seed1"),            //
          py::arg("leftBoundary"),     //
          py::arg("rightBoundary"),    //
          py::arg("bitWidthIn"),       //
          py::arg("bitWidthOut"),      //
          py::arg("elementNum"),       //
          R"(
Generate DCF MIC key.

Args:
    keyOut (torch.Tensor):              key tensor
    zOut (torch.Tensor):                zipped key tensor
    alpha (torch.Tensor):               alpha tensor
    seed0 (torch.Tensor):               seed0 tensor
    seed1 (torch.Tensor):               seed1 tensor
    leftBoundary (torch.Tensor):        left boundary tensor
    rightBoundary(torch.Tensor):        right boundary tensor
    bitWidthIn (int):                   bit width of input data
    bitWidthOut (int):                  bit width of output data
    elementNum (int):                   element number of input data

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_dcfMicKeyGen or FastFss_cuda_dcfMicKeyGen fail.
        )");

    m.def("dcf_mic_eval",           //
          &pyFastFss::dcf_mic_eval, //
          py::arg("sharedOut"),     //
          py::arg("maskedX"),       //
          py::arg("key"),           //
          py::arg("sharedZ"),       //
          py::arg("seed"),          //
          py::arg("partyId"),       //
          py::arg("leftBoundary"),  //
          py::arg("rightBoundary"), //
          py::arg("bitWidthIn"),    //
          py::arg("bitWidthOut"),   //
          py::arg("elementNum"),    //
          R"(
Evaluate DCF MIC.

Args:
    sharedOut (torch.Tensor):               shared output tensor
    maskedX (torch.Tensor):                 masked input tensor
    key (torch.Tensor):                     key tensor
    sharedZ (torch.Tensor):                 shared zipped key tensor
    seed (torch.Tensor):                    seed tensor
    partyId (int):                          party id
    leftBoundary (torch.Tensor):            left boundary tensor
    rightBoundary(torch.Tensor):            right boundary tensor
    bitWidthIn (int):                       bit width of input data
    bitWidthOut (int):                      bit width of output data
    elementNum (int):                       element number of input data
    
Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_dcfMicEval or FastFss_cuda_dcfMicEval fail.
        )");

    // ===========================================
    // ==================== ONEHOT ===============
    // ===========================================

    m.def("onehot_get_key_data_size",           //
          &pyFastFss::onehot_get_key_data_size, //
          py::arg("bitWidthIn"),                //
          py::arg("elementNum"),                //
          R"(
      )");

    m.def("onehot_key_gen",           //
          &pyFastFss::onehot_key_gen, //
          py::arg("keyInOut"),        //
          py::arg("alpha"),           //
          py::arg("bitWidthIn"),      //
          py::arg("elementNum"),      //
          R"(
    )");

    m.def("onehot_lut_eval",           //
          &pyFastFss::onehot_lut_eval, //
          py::arg("sharedOutE"),       //
          py::arg("sharedOutT"),       //
          py::arg("maskedX"),          //
          py::arg("key"),              //
          py::arg("partyId"),          //
          py::arg("lookUpTable"),      //
          py::arg("bitWidthIn"),       //
          py::arg("bitWidthOut"),      //
          py::arg("elementNum"),       //
          R"(
  )");

    // ===========================================
    // ==================== PRNG =================
    // ===========================================

    auto PrngClass = py::class_<pyFastFss::Prng>(m, "Prng");

    PrngClass.def(py::init<torch::Device>(),                //
                  py::arg("device") = torch::Device("cpu"), //
                  R"(
Create a random number generator.

Args:
    device (torch.device): device

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_prngInit or FastFss_cuda_prngInit fail.
)");

    PrngClass.def("device",                 //
                  &pyFastFss::Prng::device, //
                  R"(
Get the device.

Returns:
    torch.device: device
)");

    PrngClass.def("get_current_seed",                 //
                  &pyFastFss::Prng::get_current_seed, //
                  R"(
Get the current seed.

Returns:
    tuple[bytes, bytes]: current seed

Raises:
    RuntimeError: If the FastFss_cpu_prngGetCurrentSeed or FastFss_cuda_prngGetCurrentSeed fail.
)");

    PrngClass.def("set_current_seed",                 //
                  &pyFastFss::Prng::set_current_seed, //
                  py::arg("seed128bit"),              //
                  py::arg("counter128bit"),           //
                  R"(
Set the current seed.

Args:
    seed128bit (bytes):     seed128bit
    counter128bit (bytes):  counter128bit

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_prngSetCurrentSeed or FastFss_cuda_prngSetCurrentSeed

)");

    PrngClass.def("to_",
                  &pyFastFss::Prng::to_, //
                  py::arg("device"),     //
                  R"(
Move the random number generator to the specified device.

Args:
    device (torch.device): device

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_prngTo or FastFss_cuda_prngTofail.
)");

    PrngClass.def("rand_",                 //
                  &pyFastFss::Prng::rand_, //
                  py::arg("out"),          //
                  py::arg("bitWidth"),     //
                  R"(
Generate random numbers.

Args:
    out (torch.Tensor): output tensor
    bitWidth (int):     bit width

Returns:
    torch.Tensor:       output tensor

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the FastFss_cpu_prngRand or FastFss_cuda_prngRand fail.
)");
}