#ifndef FAST_FSS_PP_CONFIG_H
#define FAST_FSS_PP_CONFIG_H

#include <FastFss/cpu/config.h>
#include <FastFss/cuda/config.h>
#include <FastFss/errors.h>

#include <stdexcept>
#include <string>

namespace FastFss::config {

namespace cpu {

/**
 * @brief           set the number of threads
 * @param[in]   num the number of threads
 */
inline void setNumThreads(int num)
{
    int ret = FastFss_cpu_setNumThreads(num);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cpu_setNumThreads failed. error code: " + std::to_string(ret));
    }
}

/**
 * @brief   get the number of threads
 * @return  number of threads
 */
inline int getNumThreads()
{
    int n = FastFss_cpu_getNumThreads();
    if (n < 0)
    {
        throw std::runtime_error("FastFss_cpu_getNumThreads failed. error code: " + std::to_string(n));
    }
    return n;
}

} // namespace cpu

namespace cuda {

/**
 * @brief           set the number of cuda grid dim
 * @param[in]   dim the number of cuda grid dim
 */
inline void setGridDim(int dim)
{
    int ret = FastFss_cuda_setGridDim(dim);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_setGridDim failed. error code: " + std::to_string(ret));
    }
}

/**
 * @brief   get the number of cuda grid dim
 * @return  number of cuda grid dim
 */
inline int getGridDim()
{
    int n = FastFss_cuda_getGridDim();
    if (n < 0)
    {
        throw std::runtime_error("FastFss_cuda_getGridDim failed. error code: " + std::to_string(n));
    }
    return n;
}

inline void setFineGrainParallelGridDimThreshold(int dim)
{
    int ret = FastFss_cuda_setFineGrainParallelGridDimThreshold(dim);
    if (ret != FAST_FSS_SUCCESS)
    {
        throw std::runtime_error("FastFss_cuda_setFineGrainParallelGridDimThreshold failed. error code: " +
                                 std::to_string(ret));
    }
}

inline int getFineGrainParallelGridDimThreshold()
{
    int n = FastFss_cuda_getFineGrainParallelGridDimThreshold();
    if (n < 0)
    {
        throw std::runtime_error("FastFss_cuda_getFineGrainParallelGridDimThreshold failed. error code: " +
                                 std::to_string(n));
    }
    return n;
}

} // namespace cuda

} // namespace FastFss::config

#endif
