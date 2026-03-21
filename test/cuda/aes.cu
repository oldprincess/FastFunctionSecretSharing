// clang-format off
// nvcc -I include -I third_party/googletest/googletest/include -I third_party/googletest/googletest test/cuda/aes.cu third_party/googletest/googletest/src/gtest-all.cc third_party/googletest/googletest/src/gtest_main.cc -o cuda_aes.exe -std=c++17 --expt-relaxed-constexpr
// clang-format on
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "../../src/cuda/aes.cuh"

namespace {

constexpr std::array<std::uint8_t, 32> kUserKey = {
    0x04, 0xb5, 0xf0, 0x47, 0x03, 0xe2, 0x02, 0x5f, 0x5d, 0x08, 0x46,
    0xc8, 0x0a, 0x68, 0x19, 0xa0, 0x04, 0xb5, 0xf0, 0x47, 0x03, 0xe2,
    0x02, 0x5f, 0x5d, 0x08, 0x46, 0xc8, 0x0a, 0x68, 0x19, 0xa0,
};

constexpr std::array<std::uint8_t, 144> kPlaintext = {
    0x13, 0xf1, 0xdb, 0xd4, 0x4e, 0x92, 0x3a, 0x83, 0xd0, 0x23, 0x29, 0x7d,
    0xb0, 0x72, 0x59, 0x22, 0x12, 0x9f, 0x6f, 0xff, 0xc1, 0x85, 0x11, 0xde,
    0x69, 0xcf, 0xca, 0x6f, 0x9a, 0xf5, 0xdc, 0x42, 0xcf, 0x85, 0x56, 0x0a,
    0x54, 0x42, 0xdd, 0x96, 0x36, 0x6b, 0x80, 0x22, 0x6d, 0x23, 0x19, 0x36,
    0xa4, 0x88, 0x3d, 0xc9, 0x7a, 0x46, 0x88, 0x4d, 0x6f, 0x2a, 0xd6, 0x83,
    0xc9, 0xbd, 0xa5, 0x29, 0x02, 0x20, 0xaa, 0xb3, 0x8c, 0xaa, 0xa2, 0x71,
    0x45, 0x0b, 0x23, 0x1b, 0x6b, 0x26, 0x7d, 0x8a, 0x37, 0x7d, 0x77, 0xb0,
    0xfd, 0xbb, 0x90, 0xaf, 0x98, 0xc5, 0xda, 0xdd, 0x18, 0x63, 0x0d, 0x02,
    0xe8, 0x7a, 0x07, 0x58, 0xe4, 0xda, 0x47, 0x7c, 0x24, 0xe2, 0x9a, 0x99,
    0xfd, 0xd8, 0x20, 0x4e, 0x8a, 0x77, 0xeb, 0x77, 0xbf, 0x47, 0x0b, 0x23,
    0xaf, 0xd1, 0x20, 0x4c, 0xc8, 0xfa, 0x8b, 0x71,
};

__device__ std::uint8_t dUserKey[32];
__device__ std::uint8_t dPlaintext[144];

__global__ void aesTestKernel(int *ret)
{
    FastFss::impl::AES128GlobalContext aesCtx;
    FastFss::impl::AES128              aes;
    std::uint8_t                       outputByCount[kPlaintext.size()];
    std::uint8_t                       outputByTemplate[kPlaintext.size()];

    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }

    aes.set_enc_key(dUserKey, &aesCtx);
    aes.enc_blocks(outputByCount, dPlaintext, kPlaintext.size() / 16, &aesCtx);
    aes.enc_n_block<9>(outputByTemplate, dPlaintext, &aesCtx);

    *ret = 0;
    for (int i = 0; i < static_cast<int>(kPlaintext.size()); ++i)
    {
        if (outputByCount[i] != outputByTemplate[i])
        {
            *ret = -1;
            break;
        }
    }
}

#ifdef __CUDACC__
#pragma nv_diag_suppress 177
#endif
TEST(CudaAesTest, BlockApisProduceSameCiphertext)
{
    int *deviceRet = nullptr;
    int  hostRet   = -1;

    ASSERT_EQ(cudaMemcpyToSymbol(dUserKey, kUserKey.data(), kUserKey.size()),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyToSymbol(dPlaintext, kPlaintext.data(), kPlaintext.size()),
        cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&deviceRet), sizeof(int)),
              cudaSuccess);

    aesTestKernel<<<1, 256>>>(deviceRet);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(&hostRet, deviceRet, sizeof(int), cudaMemcpyDeviceToHost),
        cudaSuccess);
    ASSERT_EQ(cudaFree(deviceRet), cudaSuccess);

    EXPECT_EQ(hostRet, 0);
}
#ifdef __CUDACC__
#pragma nv_diag_default 177
#endif

} // namespace
