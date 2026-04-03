#include <FastFss/cuda/ottt.h>

#include "../impl/ottt.h"
#include "../kernel/ottt.h"
#include "../kernel/parallel_execute.h"

using namespace FastFss;

int FastFss_cuda_otttKeyGen(void       *key,
                            size_t      keyDataSize,
                            const void *alpha,
                            size_t      alphaDataSize,
                            size_t      bitWidthIn,
                            size_t      elementSize,
                            size_t      elementNum,
                            void       *cudaStreamPtr)
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
                .cudaStreamPtr = cudaStreamPtr,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cuda_otttLutEval(void       *sharedOutE,
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
                             size_t      elementNum,
                             void       *cudaStreamPtr)
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
                .cudaStreamPtr       = cudaStreamPtr,
            };
            return kernel::parallel_execute(task);
        });
}

int FastFss_cuda_otttGetKeyDataSize(size_t *keyDataSize, size_t bitWidthIn, size_t elementNum)
{
    if (!(3 <= bitWidthIn))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    *keyDataSize = impl::otttGetKeyDataSize(bitWidthIn, elementNum);
    return FAST_FSS_SUCCESS;
}
