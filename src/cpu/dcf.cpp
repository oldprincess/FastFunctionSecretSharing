#include <FastFss/cpu/dcf.h>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

#include "../impl/dcf.h"
#include "../kernel/dcf.h"
#include "../kernel/parallel_execute.h"

using namespace FastFss;

int FastFss_cpu_dcfKeyGen(void       *key,
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
                          size_t      elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DcfKeyGenTask<scalar_t> task{
                .key           = key,
                .keyDataSize   = keyDataSize,
                .alpha         = alpha,
                .alphaDataSize = alphaDataSize,
                .beta          = beta,
                .betaDataSize  = betaDataSize,
                .seed0         = seed0,
                .seedDataSize0 = seedDataSize0,
                .seed1         = seed1,
                .seedDataSize1 = seedDataSize1,
                .bitWidthIn    = bitWidthIn,
                .bitWidthOut   = bitWidthOut,
                .groupSize     = groupSize,
                .elementSize   = elementSize,
                .elementNum    = elementNum,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_dcfEval(void       *sharedOut,
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
                        size_t      cacheDataSize)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DcfEvalTask<scalar_t> task{
                .sharedOut         = sharedOut,
                .sharedOutDataSize = sharedOutSize,
                .maskedX           = maskedX,
                .maskedXDataSize   = maskedXDataSize,
                .key               = key,
                .keyDataSize       = keyDataSize,
                .seed              = seed,
                .seedDataSize      = seedDataSize,
                .partyId           = partyId,
                .bitWidthIn        = bitWidthIn,
                .bitWidthOut       = bitWidthOut,
                .groupSize         = groupSize,
                .elementSize       = elementSize,
                .elementNum        = elementNum,
                .cache             = cache,
                .cacheDataSize     = cacheDataSize,
            };
            return kernel::parallel_execute(task);
        });
}
