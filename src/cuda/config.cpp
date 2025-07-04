#include <FastFss/cuda/config.h>

#include <mutex>
#include <thread>

static int        gNumDims = 32;
static std::mutex gMutex;

int FastFss_cuda_setGridDim(int dim)
{
    try
    {
        std::lock_guard<std::mutex> lock(gMutex);
        if (dim <= 0)
        {
            return -2;
        }
        gNumDims = dim;
    }
    catch (std::exception& e)
    {
        return -1;
    }
    return 0;
}

int FastFss_cuda_getGridDim()
{
    try
    {
        std::lock_guard<std::mutex> lock(gMutex);
        return gNumDims;
    }
    catch (std::exception& e)
    {
        return -1;
    }
}
