#include "impl/ottt.h"

#include <FastFss/errors.h>
#include <FastFss/ottt.h>

#include "impl/def.h"

using namespace FastFss;

int FastFss_otttGetKeyDataSize(size_t *keyDataSize,
                               size_t  bitWidthIn,
                               size_t  elementNum)
{
    if (!(3 <= bitWidthIn))
    {
        return FAST_FSS_INVALID_BITWIDTH_ERROR;
    }
    *keyDataSize = impl::otttGetKeyDataSize(bitWidthIn, elementNum);
    return FAST_FSS_SUCCESS;
}
