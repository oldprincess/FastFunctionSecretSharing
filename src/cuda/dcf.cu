#include <FastFss/cuda/dcf.h>

#include "../impl/dcf.h"
#include "../kernel/dcf.h"
#include "../kernel/parallel_execute.h"

using namespace FastFss;

int FastFss_cuda_dcfKeyGen(void       *key,
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
            kernel::DcfKeyGenTask<scalar_t> task{};
            task.key = key;
            task.keyDataSize = keyDataSize;
            task.alpha = alpha;
            task.alphaDataSize = alphaDataSize;
            task.beta = beta;
            task.betaDataSize = betaDataSize;
            task.seed0 = seed0;
            task.seedDataSize0 = seedDataSize0;
            task.seed1 = seed1;
            task.seedDataSize1 = seedDataSize1;
            task.bitWidthIn = bitWidthIn;
            task.bitWidthOut = bitWidthOut;
            task.groupSize = groupSize;
            task.elementSize = elementSize;
            task.elementNum = elementNum;
            task.cudaStreamPtr = cudaStreamPtr;

            return kernel::parallel_execute(task);
        });
}

int FastFss_cuda_dcfEval(void       *sharedOut,
                         size_t      sharedOutSize,
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
            kernel::DcfEvalTask<scalar_t> task{};
            task.sharedOut = sharedOut;
            task.sharedOutDataSize = sharedOutSize;
            task.maskedX = maskedX;
            task.maskedXDataSize = maskedXDataSize;
            task.key = key;
            task.keyDataSize = keyDataSize;
            task.seed = seed;
            task.seedDataSize = seedDataSize;
            task.partyId = partyId;
            task.bitWidthIn = bitWidthIn;
            task.bitWidthOut = bitWidthOut;
            task.groupSize = groupSize;
            task.elementSize = elementSize;
            task.elementNum = elementNum;
            task.cache = cache;
            task.cacheDataSize = cacheDataSize;
            task.cudaStreamPtr = cudaStreamPtr;

            return kernel::parallel_execute(task);
        });
}
