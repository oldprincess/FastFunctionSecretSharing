#include <FastFss/cuda/config.h>
#include <FastFss/cuda/dpf.h>
#include <FastFss/errors.h>
#include <cuda_runtime.h>

#include "../impl/dpf.h"
#include "../kernel/dpf.h"
#include "../kernel/parallel_execute.h"

using namespace FastFss;

int FastFss_cuda_dpfKeyGen(void       *key,
                           size_t      keyDataSize,
                           const void *alpha,
                           size_t      alphaDataSize,
                           const void *beta,
                           size_t      betaDataSize,
                           const void *seed0,
                           size_t      seedDataSize0,
                           const void *seed1,
                           size_t      seedDataSize1,
                           size_t      bitWidthIn,
                           size_t      bitWidthOut,
                           size_t      groupSize,
                           size_t      elementSize,
                           size_t      elementNum,
                           void       *cudaStreamPtr)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfKeyGenTask<scalar_t> task{};
            task.key           = key;
            task.keyDataSize   = keyDataSize;
            task.alpha         = alpha;
            task.alphaDataSize = alphaDataSize;
            task.beta          = beta;
            task.betaDataSize  = betaDataSize;
            task.seed0         = seed0;
            task.seedDataSize0 = seedDataSize0;
            task.seed1         = seed1;
            task.seedDataSize1 = seedDataSize1;
            task.bitWidthIn    = bitWidthIn;
            task.bitWidthOut   = bitWidthOut;
            task.groupSize     = groupSize;
            task.elementSize   = elementSize;
            task.elementNum    = elementNum;
            task.cudaStreamPtr = cudaStreamPtr;

            return (int)kernel::parallel_execute(task);
        });
}

int FastFss_cuda_dpfEval(void       *sharedOut,
                         size_t      sharedOutDataSize,
                         const void *maskedX,
                         size_t      maskedXDataSize,
                         const void *key,
                         size_t      keyDataSize,
                         const void *seed,
                         size_t      seedDataSize,
                         int         partyId,
                         size_t      bitWidthIn,
                         size_t      bitWidthOut,
                         size_t      groupSize,
                         size_t      elementSize,
                         size_t      elementNum,
                         void       *cache,
                         size_t      cacheDataSize,
                         void       *cudaStreamPtr)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfEvalTask<scalar_t> task{};
            task.sharedOut         = sharedOut;
            task.sharedOutDataSize = sharedOutDataSize;
            task.maskedX           = maskedX;
            task.maskedXDataSize   = maskedXDataSize;
            task.key               = key;
            task.keyDataSize       = keyDataSize;
            task.seed              = seed;
            task.seedDataSize      = seedDataSize;
            task.partyId           = partyId;
            task.bitWidthIn        = bitWidthIn;
            task.bitWidthOut       = bitWidthOut;
            task.groupSize         = groupSize;
            task.elementSize       = elementSize;
            task.elementNum        = elementNum;
            task.cache             = cache;
            task.cacheDataSize     = cacheDataSize;
            task.cudaStreamPtr     = cudaStreamPtr;

            return (int)kernel::parallel_execute(task);
        });
}

int FastFss_cuda_dpfEvalAll(void       *sharedOut,
                            size_t      sharedOutDataSize,
                            const void *maskedX,
                            size_t      maskedXDataSize,
                            const void *key,
                            size_t      keyDataSize,
                            const void *seed,
                            size_t      seedDataSize,
                            int         partyId,
                            size_t      bitWidthIn,
                            size_t      bitWidthOut,
                            size_t      groupSize,
                            size_t      elementSize,
                            size_t      elementNum,
                            void       *cache,
                            size_t      cacheDataSize,
                            void       *cudaStreamPtr)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfEvalAllTask<scalar_t> task{};
            task.sharedOut         = sharedOut;
            task.sharedOutDataSize = sharedOutDataSize;
            task.maskedX           = maskedX;
            task.maskedXDataSize   = maskedXDataSize;
            task.key               = key;
            task.keyDataSize       = keyDataSize;
            task.seed              = seed;
            task.seedDataSize      = seedDataSize;
            task.partyId           = partyId;
            task.bitWidthIn        = bitWidthIn;
            task.bitWidthOut       = bitWidthOut;
            task.groupSize         = groupSize;
            task.elementSize       = elementSize;
            task.elementNum        = elementNum;
            task.cache             = cache;
            task.cacheDataSize     = cacheDataSize;
            task.cudaStreamPtr     = cudaStreamPtr;

            return (int)kernel::parallel_execute(task);
        });
}

int FastFss_cuda_dpfEvalMulti(void       *sharedOut,
                              size_t      sharedOutDataSize,
                              const void *maskedX,
                              size_t      maskedXDataSize,
                              const void *key,
                              size_t      keyDataSize,
                              const void *seed,
                              size_t      seedDataSize,
                              int         partyId,
                              const void *point,
                              size_t      pointDataSize,
                              size_t      bitWidthIn,
                              size_t      bitWidthOut,
                              size_t      groupSize,
                              size_t      elementSize,
                              size_t      elementNum,
                              void       *cache,
                              size_t      cacheDataSize,
                              void       *cudaStreamPtr)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DpfEvalMultiTask<scalar_t> task{};
            task.sharedOut         = sharedOut;
            task.sharedOutDataSize = sharedOutDataSize;
            task.maskedX           = maskedX;
            task.maskedXDataSize   = maskedXDataSize;
            task.key               = key;
            task.keyDataSize       = keyDataSize;
            task.seed              = seed;
            task.seedDataSize      = seedDataSize;
            task.partyId           = partyId;
            task.point             = point;
            task.pointDataSize     = pointDataSize;
            task.bitWidthIn        = bitWidthIn;
            task.bitWidthOut       = bitWidthOut;
            task.groupSize         = groupSize;
            task.elementSize       = elementSize;
            task.elementNum        = elementNum;
            task.cache             = cache;
            task.cacheDataSize     = cacheDataSize;
            task.cudaStreamPtr     = cudaStreamPtr;

            return kernel::parallel_execute(task);
        });
}
