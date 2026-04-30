#include <FastFss/cuda/config.h>
#include <FastFss/cuda/spline.h>

#include "../kernel/parallel_execute.h"
#include "../kernel/spline.h"

using namespace FastFss;

int FastFss_cuda_dcfSplineKeyGen(void       *key,
                                 size_t      keyDataSize,
                                 void       *e,
                                 size_t      eDataSize,
                                 void       *beta,
                                 size_t      betaDataSize,
                                 const void *alpha,
                                 size_t      alphaDataSize,
                                 const void *seed0,
                                 size_t      seedDataSize0,
                                 const void *seed1,
                                 size_t      seedDataSize1,
                                 const void *coefficients,
                                 size_t      coefficientsDataSize,
                                 size_t      degree,
                                 const void *leftEndpoints,
                                 size_t      leftEndpointsDataSize,
                                 const void *rightEndpoints,
                                 size_t      rightEndpointsDataSize,
                                 size_t      intervalNum,
                                 size_t      bitWidthIn,
                                 size_t      bitWidthOut,
                                 size_t      elementSize,
                                 size_t      elementNum,
                                 void       *cudaStreamPtr)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DcfSplineKeyGenTask<scalar_t> task{};
            task.key                    = key;
            task.keyDataSize            = keyDataSize;
            task.e                      = e;
            task.eDataSize              = eDataSize;
            task.beta                   = beta;
            task.betaDataSize           = betaDataSize;
            task.alpha                  = alpha;
            task.alphaDataSize          = alphaDataSize;
            task.seed0                  = seed0;
            task.seedDataSize0          = seedDataSize0;
            task.seed1                  = seed1;
            task.seedDataSize1          = seedDataSize1;
            task.coefficients           = coefficients;
            task.coefficientsDataSize   = coefficientsDataSize;
            task.degree                 = degree;
            task.leftEndpoints          = leftEndpoints;
            task.leftEndpointsDataSize  = leftEndpointsDataSize;
            task.rightEndpoints         = rightEndpoints;
            task.rightEndpointsDataSize = rightEndpointsDataSize;
            task.intervalNum            = intervalNum;
            task.bitWidthIn             = bitWidthIn;
            task.bitWidthOut            = bitWidthOut;
            task.elementSize            = elementSize;
            task.elementNum             = elementNum;
            task.cudaStreamPtr          = cudaStreamPtr;
            return kernel::parallel_execute(task);
        });
}

int FastFss_cuda_dcfSplineEval(void       *sharedOut,
                               size_t      sharedOutDataSize,
                               const void *maskedX,
                               size_t      maskedXDataSize,
                               const void *key,
                               size_t      keyDataSize,
                               const void *sharedE,
                               size_t      sharedEDataSize,
                               const void *sharedBeta,
                               size_t      sharedBetaDataSize,
                               const void *seed,
                               size_t      seedDataSize,
                               int         partyId,
                               const void *leftEndpoints,
                               size_t      leftEndpointsDataSize,
                               const void *rightEndpoints,
                               size_t      rightEndpointsDataSize,
                               size_t      intervalNum,
                               size_t      degree,
                               size_t      bitWidthIn,
                               size_t      bitWidthOut,
                               size_t      elementSize,
                               size_t      elementNum,
                               void       *cache,
                               size_t      cacheDataSize,
                               void       *cudaStreamPtr)
{
    return FAST_FSS_DISPATCH_INTEGRAL_TYPES(
        elementSize, { return (int)FAST_FSS_INVALID_ELEMENT_SIZE_ERROR; },
        [&] {
            kernel::DcfSplineEvalTask<scalar_t> task{};
            task.sharedOut              = sharedOut;
            task.sharedOutDataSize      = sharedOutDataSize;
            task.maskedX                = maskedX;
            task.maskedXDataSize        = maskedXDataSize;
            task.key                    = key;
            task.keyDataSize            = keyDataSize;
            task.sharedE                = sharedE;
            task.sharedEDataSize        = sharedEDataSize;
            task.sharedBeta             = sharedBeta;
            task.sharedBetaDataSize     = sharedBetaDataSize;
            task.seed                   = seed;
            task.seedDataSize           = seedDataSize;
            task.partyId                = partyId;
            task.leftEndpoints          = leftEndpoints;
            task.leftEndpointsDataSize  = leftEndpointsDataSize;
            task.rightEndpoints         = rightEndpoints;
            task.rightEndpointsDataSize = rightEndpointsDataSize;
            task.intervalNum            = intervalNum;
            task.degree                 = degree;
            task.bitWidthIn             = bitWidthIn;
            task.bitWidthOut            = bitWidthOut;
            task.elementSize            = elementSize;
            task.elementNum             = elementNum;
            task.cache                  = cache;
            task.cacheDataSize          = cacheDataSize;
            task.cudaStreamPtr          = cudaStreamPtr;
            return kernel::parallel_execute(task);
        });
}
