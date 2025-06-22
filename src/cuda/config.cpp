#include <FastFss/cuda/config.h>

#include <mutex>
#include <thread>

static int        gNumDims = 32;
static std::mutex gMutex;

int FastFss_cuda_setGridDim(int dim)
{
    std::lock_guard<std::mutex> lock(gMutex);
    if (dim <= 0)
    {
        return -1;
    }
    gNumDims = dim;
    return 0;
}

int FastFss_cuda_getGridDim()
{
    std::lock_guard<std::mutex> lock(gMutex);
    return gNumDims;
}
