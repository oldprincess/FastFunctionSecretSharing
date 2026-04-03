#include <FastFss/cpu/prng.h>
#include <FastFss/errors.h>

#include <cstring>

#if !defined(AES_IMPL)
#include "../impl/aesni.h"
#define AES_IMPL
#endif

namespace FastFss::cpu {

struct Prng
{
public:
    std::uint8_t seed[16];
    std::uint8_t counter[16];
};

static void aes128_ctr(const void *seed, void *counter, void *dst, size_t bytesNum) noexcept
{
    impl::AES128 aes128ctx;
    aes128ctx.set_enc_key(seed);

    std::size_t    blockNum   = bytesNum / 16;
    std::size_t    bytesRem   = bytesNum % 16;
    std::uint64_t *counterPtr = (std::uint64_t *)counter;
    std::uint8_t  *dstPtr     = (std::uint8_t *)dst;

    std::size_t i = 0;

    constexpr int batchSize = 16;
    for (; (i + batchSize) <= blockNum; i += 1 * batchSize)
    {
        std::size_t dstOffest = i * 16;

        std::uint64_t curCounter[batchSize][2];
        for (int j = 0; j < batchSize; j++)
        {
            curCounter[j][0] = counterPtr[0] + i + j;
            std::uint64_t CF = (curCounter[j][0] < counterPtr[0]);
            curCounter[j][1] = counterPtr[1] + CF;
        }
        aes128ctx.enc_n_block<batchSize>(dstPtr + dstOffest, curCounter);
    }

    for (; i < blockNum; i++)
    {
        std::size_t dstOffest = i * 16;

        std::uint64_t curCounter[2];
        curCounter[0] = counterPtr[0] + i;
        curCounter[1] = counterPtr[1] + (curCounter[0] < i);

        aes128ctx.enc_n_block<1>(dstPtr + dstOffest, curCounter);
    }

    if (bytesRem != 0)
    {
        std::uint8_t  outputBufer[16];
        std::uint64_t curCounter[2];
        curCounter[0] = counterPtr[0] + blockNum;
        curCounter[1] = counterPtr[1] + (curCounter[0] < blockNum);

        std::size_t dstOffest = blockNum * 16;

        aes128ctx.enc_n_block<1>(outputBufer, curCounter);
        for (std::size_t i = 0; i < bytesRem; i++)
        {
            dstPtr[dstOffest + i] = outputBufer[i];
        }
        blockNum += 1;
    }
    counterPtr[0] = counterPtr[0] + blockNum;
    counterPtr[1] = counterPtr[1] + (counterPtr[0] < blockNum);
}

} // namespace FastFss::cpu

void *FastFss_cpu_prngInit()
{
    void *prng = nullptr;
    try
    {
        prng = (void *)new FastFss::cpu::Prng();
    }
    catch (...)
    {
    }
    return prng;
}

void FastFss_cpu_prngRelease(void *prng)
{
    if (prng == nullptr)
    {
        return;
    }
    delete (FastFss::cpu::Prng *)prng;
}

int FastFss_cpu_prngSetCurrentSeed(void *prng, const void *seed128bit, const void *counter128bit)
{
    if (prng == nullptr || seed128bit == nullptr)
    {
        return FAST_FSS_PRNG_INPUT_INVALID_ARGUMENT;
    }
    FastFss::cpu::Prng *prngObj = (FastFss::cpu::Prng *)prng;
    std::memcpy(prngObj->seed, seed128bit, 16);
    if (counter128bit != nullptr)
    {
        std::memcpy(prngObj->counter, counter128bit, 16);
    }
    else
    {
        std::memset(prngObj->counter, 0, 16);
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_cpu_prngGetCurrentSeed(const void *prng, void *seed128bit, void *counter128bit)
{
    if (prng == nullptr || seed128bit == nullptr)
    {
        return FAST_FSS_PRNG_INPUT_INVALID_ARGUMENT;
    }
    std::memcpy(seed128bit, ((FastFss::cpu::Prng *)prng)->seed, 16);
    if (counter128bit != nullptr)
    {
        std::memcpy(counter128bit, ((FastFss::cpu::Prng *)prng)->counter, 16);
    }
    return FAST_FSS_SUCCESS;
}

int FastFss_cpu_prngGen(void *prng, void *dst, size_t bitWidth, size_t elementSize, size_t elementNum)
{
    if (prng == nullptr || dst == nullptr)
    {
        return FAST_FSS_PRNG_INPUT_INVALID_ARGUMENT;
    }
    if (bitWidth == 0 || bitWidth > elementSize * 8)
    {
        return FAST_FSS_PRNG_INVALID_BIT_WIDTH;
    }
    FastFss::cpu::Prng *prngObj = (FastFss::cpu::Prng *)prng;
    FastFss::cpu::aes128_ctr(prngObj->seed, prngObj->counter, dst, elementSize * elementNum);
    return FAST_FSS_SUCCESS;
}