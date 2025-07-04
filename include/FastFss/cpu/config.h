#ifndef FAST_FSS_CPU_CONFIG
#define FAST_FSS_CPU_CONFIG

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief           set the number of threads
 * @param[in]   num the number of threads
 * @return          error code
 * @retval          0   success
 * @retval          -1  runtime error
 * @retval          -2  invalid parameter
 */
int FastFss_cpu_setNumThreads(int num);

/**
 * @brief   get the number of threads
 * @return  number of threads or error code
 * @retval  >0  number of threads
 * @retval  -1  runtime error
 */
int FastFss_cpu_getNumThreads();

#ifdef __cplusplus
}
#endif

#endif