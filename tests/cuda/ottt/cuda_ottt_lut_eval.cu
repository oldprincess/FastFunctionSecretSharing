#include <FastFss/cuda/ottt.h>
#include <FastFss/errors.h>
#include <FastFss/ottt.h>
#include <gtest/gtest.h>

#include <vector>

#include "common/cuda_test_utils.cuh"
#include "common/test_types.hpp"
#include "common/test_utils.hpp"

namespace FastFss::tests {
namespace {

template <typename T>
class CudaOtttLutEvalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE(CudaOtttLutEvalTest, EvalElementTypes);

TYPED_TEST(CudaOtttLutEvalTest, LutEvalReconstructsLookupValue)
{
    using T                          = TypeParam;
    constexpr std::size_t bitWidthIn = 7;

    for (const auto elementNum : element_counts())
    {
        std::size_t keySize = 0;
        ASSERT_EQ(FastFss_otttGetKeyDataSize(&keySize, bitWidthIn, elementNum), FAST_FSS_SUCCESS);

        auto           alpha = random_vector<T>(elementNum, bitWidthIn);
        auto           x     = random_vector<T>(elementNum, bitWidthIn);
        std::vector<T> maskedX(elementNum);
        for (std::size_t i = 0; i < elementNum; ++i)
        {
            maskedX[i] = mod_bits<T>(alpha[i] + x[i], bitWidthIn);
        }

        std::vector<T> lut(1ULL << bitWidthIn);
        for (std::size_t i = 0; i < lut.size(); ++i)
        {
            lut[i] = mod_bits<T>(T(i) * T(3) + T(1), bit_size_v<T>);
        }

        std::vector<std::uint8_t> key0(keySize);
        std::vector<std::uint8_t> key1(keySize);
        for (std::size_t i = 0; i < key0.size(); ++i)
        {
            key0[i] = static_cast<std::uint8_t>((7 * i + 3) & 0xffU);
            key1[i] = key0[i];
        }

        ::FastFss::tests::cuda::StreamPair streams;
        auto                               dKey0 = ::FastFss::tests::cuda::make_gpu_buffer(key0.data(), key0.size());
        auto                               dKey1 = ::FastFss::tests::cuda::make_gpu_buffer(key1.data(), key1.size());
        auto dAlpha   = ::FastFss::tests::cuda::make_gpu_buffer(alpha.data(), alpha.size() * sizeof(T));
        auto dMaskedX = ::FastFss::tests::cuda::make_gpu_buffer(maskedX.data(), maskedX.size() * sizeof(T));
        auto dLut     = ::FastFss::tests::cuda::make_gpu_buffer(lut.data(), lut.size() * sizeof(T));
        auto dE0      = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
        auto dE1      = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
        auto dT0      = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));
        auto dT1      = ::FastFss::tests::cuda::make_gpu_buffer(elementNum * sizeof(T));

        ASSERT_EQ(FastFss_cuda_otttKeyGen(dKey1.get(), keySize, dAlpha.get(), alpha.size() * sizeof(T), bitWidthIn,
                                          sizeof(T), elementNum, streams.party_stream(1)),
                  FAST_FSS_SUCCESS);
        ASSERT_EQ(FastFss_cuda_otttLutEval(dE0.get(), elementNum * sizeof(T), dT0.get(), elementNum * sizeof(T),
                                           dMaskedX.get(), maskedX.size() * sizeof(T), dKey0.get(), keySize, 0,
                                           dLut.get(), lut.size() * sizeof(T), bitWidthIn, sizeof(T), elementNum,
                                           streams.party_stream(0)),
                  FAST_FSS_SUCCESS);
        ASSERT_EQ(FastFss_cuda_otttLutEval(dE1.get(), elementNum * sizeof(T), dT1.get(), elementNum * sizeof(T),
                                           dMaskedX.get(), maskedX.size() * sizeof(T), dKey1.get(), keySize, 1,
                                           dLut.get(), lut.size() * sizeof(T), bitWidthIn, sizeof(T), elementNum,
                                           streams.party_stream(1)),
                  FAST_FSS_SUCCESS);

        std::vector<T> shareE0(elementNum), shareE1(elementNum);
        std::vector<T> shareT0(elementNum), shareT1(elementNum);
        ::FastFss::tests::cuda::memcpy_gpu_to_cpu(shareE0.data(), dE0.get(), shareE0.size() * sizeof(T));
        ::FastFss::tests::cuda::memcpy_gpu_to_cpu(shareE1.data(), dE1.get(), shareE1.size() * sizeof(T));
        ::FastFss::tests::cuda::memcpy_gpu_to_cpu(shareT0.data(), dT0.get(), shareT0.size() * sizeof(T));
        ::FastFss::tests::cuda::memcpy_gpu_to_cpu(shareT1.data(), dT1.get(), shareT1.size() * sizeof(T));

        for (std::size_t i = 0; i < elementNum; ++i)
        {
            const T e   = mod_bits<T>(shareE0[i] + shareE1[i], 1);
            const T t   = mod_bits<T>(shareT0[i] + shareT1[i], bit_size_v<T>);
            const T out = mod_bits<T>(e == 0 ? t : (T(0) - t), bit_size_v<T>);
            EXPECT_EQ(out, lut[static_cast<std::size_t>(x[i])]);
        }
    }
}

} // namespace
} // namespace FastFss::tests
