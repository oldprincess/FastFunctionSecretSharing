#ifndef FAST_FSS_CUDA_CONFIG
#define FAST_FSS_CUDA_CONFIG

#ifdef __cplusplus
extern "C" {
#endif

int FastFss_cuda_setGridDim(int dim);

int FastFss_cuda_getGridDim();

#ifdef __cplusplus
}
#endif

#endif