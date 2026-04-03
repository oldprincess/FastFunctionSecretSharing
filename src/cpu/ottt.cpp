#include "../impl/ottt.h"

#include <FastFss/cpu/ottt.h>

#include "../kernel/ottt.h"
#include "../kernel/parallel_execute.h"

using namespace FastFss;

int FastFss_cpu_otttKeyGen(void       *key,
                           size_t      keyDataSize,
                           const void *alpha,
                           size_t      alphaDataSize,
                           size_t      bitWidthIn,
                           size_t      elementSize,
                           size_t      elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::OtttKeyGenTask<scalar_t> task{
                .key           = key,
                .keyDataSize   = keyDataSize,
                .alpha         = alpha,
                .alphaDataSize = alphaDataSize,
                .bitWidthIn    = bitWidthIn,
                .elementSize   = elementSize,
                .elementNum    = elementNum,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cpu_otttLutEval(void       *sharedOutE,
                            size_t      sharedOutEDataSize,
                            void       *sharedOutT,
                            size_t      sharedOutTDataSize,
                            const void *maskedX,
                            size_t      maskedXDataSize,
                            const void *key,
                            size_t      keyDataSize,
                            int         partyId,
                            const void *lookUpTable,
                            size_t      lookUpTableDataSize,
                            size_t      bitWidthIn,
                            size_t      elementSize,
                            size_t      elementNum)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::OtttLutEvalTask<scalar_t> task{
                .sharedOutE          = sharedOutE,
                .sharedOutEDataSize  = sharedOutEDataSize,
                .sharedOutT          = sharedOutT,
                .sharedOutTDataSize  = sharedOutTDataSize,
                .maskedX             = maskedX,
                .maskedXDataSize     = maskedXDataSize,
                .key                 = key,
                .keyDataSize         = keyDataSize,
                .partyId             = partyId,
                .lookUpTable         = lookUpTable,
                .lookUpTableDataSize = lookUpTableDataSize,
                .bitWidthIn          = bitWidthIn,
                .elementSize         = elementSize,
                .elementNum          = elementNum,
            };
            return kernel::parallel_execute(task);
        });
}
