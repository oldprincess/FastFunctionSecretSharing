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
            kernel::OtttKeyGenTask<scalar_t> task{};
            task.key           = key;
            task.keyDataSize   = keyDataSize;
            task.alpha         = alpha;
            task.alphaDataSize = alphaDataSize;
            task.bitWidthIn    = bitWidthIn;
            task.elementSize   = elementSize;
            task.elementNum    = elementNum;

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
            kernel::OtttLutEvalTask<scalar_t> task{};
            task.sharedOutE          = sharedOutE;
            task.sharedOutEDataSize  = sharedOutEDataSize;
            task.sharedOutT          = sharedOutT;
            task.sharedOutTDataSize  = sharedOutTDataSize;
            task.maskedX             = maskedX;
            task.maskedXDataSize     = maskedXDataSize;
            task.key                 = key;
            task.keyDataSize         = keyDataSize;
            task.partyId             = partyId;
            task.lookUpTable         = lookUpTable;
            task.lookUpTableDataSize = lookUpTableDataSize;
            task.bitWidthIn          = bitWidthIn;
            task.elementSize         = elementSize;
            task.elementNum          = elementNum;

            return kernel::parallel_execute(task);
        });
}
