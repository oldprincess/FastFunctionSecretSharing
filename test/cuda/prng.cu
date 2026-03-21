// clang-format off
// nvcc -I include -I third_party/googletest/googletest/include -I third_party/googletest/googletest src/cuda/prng.cu test/cuda/prng.cu third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cuda_prng.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <FastFss/cuda/prng.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "utils.cuh"

namespace {

class CudaPrngTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        prng = FastFss_cuda_prngInit();
        ASSERT_NE(prng, nullptr);
    }

    void TearDown() override
    {
        if (prng != nullptr)
        {
            FastFss_cuda_prngRelease(prng);
        }
    }

    void *prng = nullptr;
};

TEST_F(CudaPrngTest, ReseedingReproducesSameBytes)
{
    const std::array<std::uint8_t, 16> seed{};
    const std::array<std::uint8_t, 16> counter = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    };
    std::vector<std::uint8_t> hostBuffer1(163);
    std::vector<std::uint8_t> hostBuffer2(163);
    auto deviceBuffer1 = FastFss::cuda::make_unique_gpu_ptr(hostBuffer1.size());
    auto deviceBuffer2 = FastFss::cuda::make_unique_gpu_ptr(hostBuffer2.size());

    ASSERT_EQ(
        FastFss_cuda_prngSetCurrentSeed(prng, seed.data(), counter.data()), 0);
    ASSERT_EQ(FastFss_cuda_prngGen(prng, deviceBuffer1.get(), 8, 1,
                                   hostBuffer1.size(), nullptr),
              0);

    ASSERT_EQ(
        FastFss_cuda_prngSetCurrentSeed(prng, seed.data(), counter.data()), 0);
    ASSERT_EQ(FastFss_cuda_prngGen(prng, deviceBuffer2.get(), 8, 1,
                                   hostBuffer2.size(), nullptr),
              0);

    FastFss::cuda::memcpy_gpu2cpu(hostBuffer1.data(), deviceBuffer1.get(),
                                  hostBuffer1.size());
    FastFss::cuda::memcpy_gpu2cpu(hostBuffer2.data(), deviceBuffer2.get(),
                                  hostBuffer2.size());
    EXPECT_EQ(hostBuffer1, hostBuffer2);
}

TEST_F(CudaPrngTest, GetCurrentSeedReturnsLastConfiguredSeed)
{
    const std::array<std::uint8_t, 16> seed = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    const std::array<std::uint8_t, 16> counter = {
        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    };
    std::array<std::uint8_t, 16> currentSeed{};
    std::array<std::uint8_t, 16> currentCounter{};

    ASSERT_EQ(
        FastFss_cuda_prngSetCurrentSeed(prng, seed.data(), counter.data()), 0);
    ASSERT_EQ(FastFss_cuda_prngGetCurrentSeed(prng, currentSeed.data(),
                                              currentCounter.data()),
              0);

    EXPECT_EQ(currentSeed, seed);
    EXPECT_EQ(currentCounter, counter);
}

} // namespace
