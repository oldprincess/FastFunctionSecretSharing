#ifndef FAST_FSS_PP_PRNG_H
#define FAST_FSS_PP_PRNG_H

#include <FastFss/cpu/prng.h>
#include <FastFss/cuda/prng.h>
#include <FastFss/errors.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace FastFss::prng {

namespace cpu {

class Prng
{
public:
    /** Init native PRNG and set random 128-bit seed/counter on host. */
    Prng()
    {
        void *const raw = FastFss_cpu_prngInit();
        if (raw == nullptr)
        {
            throw std::runtime_error("FastFss_cpu_prngInit failed");
        }
        ctx_.reset(raw, releaseCtx);
        std::array<std::uint8_t, 16> seed{};
        std::array<std::uint8_t, 16> counter{};
        fillBytesRandomDevice(seed);
        fillBytesRandomDevice(counter);
        setSeed(std::span<const std::uint8_t, 16>{seed}, std::span<const std::uint8_t, 16>{counter});
    }

    /**
     * @brief Write host 128-bit seed/counter into native PRNG state.
     * @param seed    16-byte AES-128 key (host).
     * @param counter 16-byte CTR initial block (host).
     */
    void setSeed(std::span<const std::uint8_t, 16> seed, std::span<const std::uint8_t, 16> counter)
    {
        throwIfError(FastFss_cpu_prngSetCurrentSeed(ctx_.get(), seed.data(), counter.data()),
                     "FastFss_cpu_prngSetCurrentSeed");
    }

    /** @brief Host-side snapshot of PRNG seed and counter. */
    struct SeedState
    {
        std::array<std::uint8_t, 16> seed{};    ///< 16-byte AES-128 key.
        std::array<std::uint8_t, 16> counter{}; ///< 16-byte CTR counter.
    };

    /** @brief Read current seed/counter from native context into host. */
    SeedState getSeed() const
    {
        SeedState s{};
        throwIfError(FastFss_cpu_prngGetCurrentSeed(ctx_.get(), s.seed.data(), s.counter.data()),
                     "FastFss_cpu_prngGetCurrentSeed");
        return s;
    }

    /**
     * @brief Fill host buffer with random elements (same contract as @c FastFss_cpu_prngGen).
     * @param dst          Writable buffer; total size @c elementSize * elementCount bytes.
     * @param bitWidth     Effective bits per element; must be @c <= elementSize * 8.
     * @param elementSize  Byte size of one element.
     * @param elementCount Number of elements to write.
     */
    void generate(void *dst, std::size_t bitWidth, std::size_t elementSize, std::size_t elementCount)
    {
        throwIfError(FastFss_cpu_prngGen(ctx_.get(), dst, bitWidth, elementSize, elementCount), "FastFss_cpu_prngGen");
    }

    /**
     * @brief Like generate(void*, ...) with @c elementSize = sizeof(T).
     * @tparam T           Element type (not @c void).
     * @param dst          Writable buffer of @a elementCount elements.
     * @param bitWidth     Effective bits per element; must be @c <= sizeof(T) * 8.
     * @param elementCount Number of @c T elements.
     */
    template <typename T>
    void generate(T *dst, std::size_t bitWidth, std::size_t elementCount)
    {
        static_assert(!std::is_void_v<T>, "T must not be void");
        generate(static_cast<void *>(dst), bitWidth, sizeof(T), elementCount);
    }

    /** @brief Opaque native context pointer for C API interop (shared with all copies). */
    [[nodiscard]] void *nativeHandle() noexcept
    {
        return ctx_.get();
    }

    /** @copydoc nativeHandle() */
    [[nodiscard]] const void *nativeHandle() const noexcept
    {
        return ctx_.get();
    }

private:
    static void releaseCtx(void *p) noexcept
    {
        if (p != nullptr)
        {
            FastFss_cpu_prngRelease(p);
        }
    }

    static void fillBytesRandomDevice(std::array<std::uint8_t, 16> &out)
    {
        std::random_device                      rd;
        std::uniform_int_distribution<unsigned> dist(0, 255);
        for (auto &b : out)
        {
            b = static_cast<std::uint8_t>(dist(rd));
        }
    }

    static void throwIfError(int code, const char *api)
    {
        if (code == FAST_FSS_SUCCESS)
        {
            return;
        }
        throw std::runtime_error(std::string(api) + " failed (code " + std::to_string(code) + ")");
    }

    std::shared_ptr<void> ctx_;
};

} // namespace cpu

namespace cuda {

class Prng
{
public:
    /** Allocate device PRNG state and set random 128-bit seed/counter from host. */
    Prng()
    {
        void *const raw = FastFss_cuda_prngInit();
        if (raw == nullptr)
        {
            throw std::runtime_error("FastFss_cuda_prngInit failed");
        }
        ctx_.reset(raw, releaseCtx);
        std::array<std::uint8_t, 16> seed{};
        std::array<std::uint8_t, 16> counter{};
        fillBytesRandomDevice(seed);
        fillBytesRandomDevice(counter);
        setSeed(std::span<const std::uint8_t, 16>{seed}, std::span<const std::uint8_t, 16>{counter});
    }

    /**
     * @brief Copy host 128-bit seed/counter into device PRNG state.
     * @param seed    16-byte AES-128 key (host).
     * @param counter 16-byte CTR initial block (host).
     */
    void setSeed(std::span<const std::uint8_t, 16> seed, std::span<const std::uint8_t, 16> counter)
    {
        throwIfError(FastFss_cuda_prngSetCurrentSeed(ctx_.get(), seed.data(), counter.data()),
                     "FastFss_cuda_prngSetCurrentSeed");
    }

    /** @brief Host snapshot after copying device seed/counter. */
    struct SeedState
    {
        std::array<std::uint8_t, 16> seed{};    ///< 16-byte AES-128 key (from device).
        std::array<std::uint8_t, 16> counter{}; ///< 16-byte CTR counter (from device).
    };

    /** @brief Copy device seed/counter into host @ref SeedState. */
    SeedState getSeed() const
    {
        SeedState s{};
        throwIfError(FastFss_cuda_prngGetCurrentSeed(ctx_.get(), s.seed.data(), s.counter.data()),
                     "FastFss_cuda_prngGetCurrentSeed");
        return s;
    }

    /**
     * @brief Fill device buffer (same contract as @c FastFss_cuda_prngGen).
     * @param deviceDst     Device-accessible buffer; size @c elementSize * elementCount bytes.
     * @param bitWidth      Effective bits per element; must be @c <= elementSize * 8.
     * @param elementSize   Byte size of one element.
     * @param elementCount  Number of elements to write.
     * @param cudaStreamPtr CUDA stream handle as @c void* (library convention); @c nullptr for default stream.
     */
    void generate(void       *deviceDst,
                  std::size_t bitWidth,
                  std::size_t elementSize,
                  std::size_t elementCount,
                  void       *cudaStreamPtr = nullptr)
    {
        throwIfError(FastFss_cuda_prngGen(ctx_.get(), deviceDst, bitWidth, elementSize, elementCount, cudaStreamPtr),
                     "FastFss_cuda_prngGen");
    }

    /**
     * @brief Like generate(void*, ...) with @c elementSize = sizeof(T).
     * @tparam T            Element type (not @c void); must be trivially device-writable layout.
     * @param devicePtr     Device pointer to @a elementCount elements.
     * @param bitWidth      Effective bits per element; must be @c <= sizeof(T) * 8.
     * @param elementCount  Number of @c T elements.
     * @param cudaStreamPtr Same as non-template @ref generate(void*,std::size_t,std::size_t,std::size_t,void*).
     */
    template <typename T>
    void generate(T *devicePtr, std::size_t bitWidth, std::size_t elementCount, void *cudaStreamPtr = nullptr)
    {
        static_assert(!std::is_void_v<T>, "T must not be void");
        generate(static_cast<void *>(devicePtr), bitWidth, sizeof(T), elementCount, cudaStreamPtr);
    }

    /** @brief Opaque device PRNG state pointer for C API interop. */
    [[nodiscard]] void *nativeHandle() noexcept
    {
        return ctx_.get();
    }

    /** @copydoc nativeHandle() */
    [[nodiscard]] const void *nativeHandle() const noexcept
    {
        return ctx_.get();
    }

private:
    static void releaseCtx(void *p) noexcept
    {
        if (p != nullptr)
        {
            FastFss_cuda_prngRelease(p);
        }
    }

    static void fillBytesRandomDevice(std::array<std::uint8_t, 16> &out)
    {
        std::random_device                      rd;
        std::uniform_int_distribution<unsigned> dist(0, 255);
        for (auto &b : out)
        {
            b = static_cast<std::uint8_t>(dist(rd));
        }
    }

    static void throwIfError(int code, const char *api)
    {
        if (code == FAST_FSS_SUCCESS)
        {
            return;
        }
        throw std::runtime_error(std::string(api) + " failed (code " + std::to_string(code) + ")");
    }

    std::shared_ptr<void> ctx_;
};

} // namespace cuda

} // namespace FastFss::prng

#endif