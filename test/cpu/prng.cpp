// clang-format off
// g++ -I include -I third_party/googletest/googletest/include -I third_party/googletest/googletest src/cpu/prng.cpp test/cpu/prng.cpp third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cpu_prng.exe -std=c++17 -maes
// clang-format on
#include <FastFss/cpu/prng.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

class CpuPrngTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        prng = FastFss_cpu_prngInit();
        ASSERT_NE(prng, nullptr);
    }

    void TearDown() override
    {
        if (prng != nullptr)
        {
            FastFss_cpu_prngRelease(prng);
        }
    }

    void *prng = nullptr;
};

TEST_F(CpuPrngTest, ReseedingReproducesSameBytes)
{
    const std::array<std::uint8_t, 16> seed{};
    const std::array<std::uint8_t, 16> counter = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    };
    std::vector<std::uint8_t> buffer1(163);
    std::vector<std::uint8_t> buffer2(163);

    ASSERT_EQ(FastFss_cpu_prngSetCurrentSeed(prng, seed.data(), counter.data()),
              0);
    ASSERT_EQ(FastFss_cpu_prngGen(prng, buffer1.data(), 8, 1, buffer1.size()),
              0);

    ASSERT_EQ(FastFss_cpu_prngSetCurrentSeed(prng, seed.data(), counter.data()),
              0);
    ASSERT_EQ(FastFss_cpu_prngGen(prng, buffer2.data(), 8, 1, buffer2.size()),
              0);

    EXPECT_EQ(buffer1, buffer2);
}

TEST_F(CpuPrngTest, GetCurrentSeedReturnsLastConfiguredSeed)
{
    const std::array<std::uint8_t, 16> seed = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    const std::array<std::uint8_t, 16> counter = {
        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    };
    std::array<std::uint8_t, 16> currentSeed{};
    std::array<std::uint8_t, 16> currentCounter{};

    ASSERT_EQ(FastFss_cpu_prngSetCurrentSeed(prng, seed.data(), counter.data()),
              0);
    ASSERT_EQ(FastFss_cpu_prngGetCurrentSeed(prng, currentSeed.data(),
                                             currentCounter.data()),
              0);

    EXPECT_EQ(currentSeed, seed);
    EXPECT_EQ(currentCounter, counter);
}

} // namespace
