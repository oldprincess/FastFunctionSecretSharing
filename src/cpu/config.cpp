#include <FastFss/cpu/config.h>

#include <mutex>
#include <thread>

static int        gNumThreads = std::thread::hardware_concurrency();
static std::mutex gMutex;

int FastFss_cpu_setNumThreads(int num)
{
    try
    {
        std::lock_guard<std::mutex> lock(gMutex);
        if (num <= 0)
        {
            return -2;
        }
        gNumThreads = num;
    }
    catch (...)
    {
        return -1;
    }
    return 0;
}

int FastFss_cpu_getNumThreads()
{
    try
    {
        std::lock_guard<std::mutex> lock(gMutex);
        return gNumThreads;
    }
    catch (...)
    {
        return -1;
    }
}
