#ifndef FAST_FSS_CUDA_CONFIG
#define FAST_FSS_CUDA_CONFIG

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief           set the number of cuda grid dim
 * @param[in]   dim the number of cuda grid dim
 * @return          error code
 * @retval          0   success
 * @retval          -1  runtime error
 * @retval          -2  invalid parameter
 */
int FastFss_cuda_setGridDim(int dim);

/**
 * @brief   get the number of cuda grid dim
 * @return  number of cuda grid dim or error code
 * @retval  >0  number of cuda grid dim
 * @retval  -1  runtime error
 */
int FastFss_cuda_getGridDim();

#ifdef __cplusplus
}
#endif

#endif