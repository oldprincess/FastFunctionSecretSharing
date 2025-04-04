#include <FastFss/cpu/config.h>

#include <mutex>
#include <thread>

static int        gNumThreads = 1;
static std::mutex gMutex;

int FastFss_cpu_setNumThreads(int num)
{
    std::lock_guard<std::mutex> lock(gMutex);
    if (num <= 0)
    {
        return -1;
    }
    gNumThreads = num;
    return 0;
}

int FastFss_cpu_getNumThreads()
{
    std::lock_guard<std::mutex> lock(gMutex);
    return gNumThreads;
}
