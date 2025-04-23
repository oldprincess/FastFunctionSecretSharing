#pragma once
#ifndef SRC_IMPL_AESNI_H
#define SRC_IMPL_AESNI_H

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace FastFss::impl {

namespace internal {

typedef struct Aes128AesniCTX
{
    std::uint8_t round_key[11][16];
} Aes128AesniCTX;

/**
 * MIT License. Copyright (c) 2023 Jubal Mordecai Velasco,
 * https://github.com/mrdcvlsc/AES/blob/main/AES.hpp
 */
static inline __m128i AES_128_ASSIST(__m128i temp1, __m128i temp2)
{
    __m128i temp3;
    temp2 = _mm_shuffle_epi32(temp2, 0xff);
    temp3 = _mm_slli_si128(temp1, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp3 = _mm_slli_si128(temp3, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp3 = _mm_slli_si128(temp3, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp1 = _mm_xor_si128(temp1, temp2);
    return temp1;
}

/**
 * MIT License. Copyright (c) 2023 Jubal Mordecai Velasco,
 * https://github.com/mrdcvlsc/AES/blob/main/AES.hpp
 */
inline void AES_128_Key_Expansion(const unsigned char* userkey,
                                  unsigned char*       key)
{
    __m128i  temp1, temp2;
    __m128i* Key_Schedule = (__m128i*)key;

    temp1            = _mm_loadu_si128((__m128i*)userkey);
    Key_Schedule[0]  = temp1;
    temp2            = _mm_aeskeygenassist_si128(temp1, 0x1);
    temp1            = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[1]  = temp1;
    temp2            = _mm_aeskeygenassist_si128(temp1, 0x2);
    temp1            = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[2]  = temp1;
    temp2            = _mm_aeskeygenassist_si128(temp1, 0x4);
    temp1            = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[3]  = temp1;
    temp2            = _mm_aeskeygenassist_si128(temp1, 0x8);
    temp1            = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[4]  = temp1;
    temp2            = _mm_aeskeygenassist_si128(temp1, 0x10);
    temp1            = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[5]  = temp1;
    temp2            = _mm_aeskeygenassist_si128(temp1, 0x20);
    temp1            = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[6]  = temp1;
    temp2            = _mm_aeskeygenassist_si128(temp1, 0x40);
    temp1            = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[7]  = temp1;
    temp2            = _mm_aeskeygenassist_si128(temp1, 0x80);
    temp1            = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[8]  = temp1;
    temp2            = _mm_aeskeygenassist_si128(temp1, 0x1b);
    temp1            = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[9]  = temp1;
    temp2            = _mm_aeskeygenassist_si128(temp1, 0x36);
    temp1            = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[10] = temp1;
}

/**
 * Ending here, to the previous similar comment declaration.
 * the code is
 * "derived from Jubal Mordecai Velasco,
 * https://github.com/mrdcvlsc/AES/blob/main/AES.hpp"
 */

// ****************************************
// ************* AES 128 ******************
// ****************************************

inline void aes128_aesni_enc_key_init(std::uint8_t  round_key[11][16],
                                      const uint8_t user_key[16])
{
    __m128i rk[11];
    AES_128_Key_Expansion(user_key, (unsigned char*)rk);
    _mm_storeu_si128((__m128i*)(round_key[0]), rk[0]);
    _mm_storeu_si128((__m128i*)(round_key[1]), rk[1]);
    _mm_storeu_si128((__m128i*)(round_key[2]), rk[2]);
    _mm_storeu_si128((__m128i*)(round_key[3]), rk[3]);
    _mm_storeu_si128((__m128i*)(round_key[4]), rk[4]);
    _mm_storeu_si128((__m128i*)(round_key[5]), rk[5]);
    _mm_storeu_si128((__m128i*)(round_key[6]), rk[6]);
    _mm_storeu_si128((__m128i*)(round_key[7]), rk[7]);
    _mm_storeu_si128((__m128i*)(round_key[8]), rk[8]);
    _mm_storeu_si128((__m128i*)(round_key[9]), rk[9]);
    _mm_storeu_si128((__m128i*)(round_key[10]), rk[10]);
}

template <int N>
inline void aes128_aesni_compute_n_block(const std::uint8_t round_key[11][16],
                                         void*              ciphertext,
                                         const void*        plaintext)
{
    const __m128i* rk = (__m128i*)(round_key);
    __m128i        state[N];
    __m128i        rki;

    for (int j = 0; j < N; j++)
    {
        state[j] = _mm_loadu_si128((const __m128i*)plaintext + j);
    }

    rki = _mm_loadu_si128(rk + 0);
    for (int j = 0; j < N; j++)
    {
        state[j] = _mm_xor_si128(state[j], rki);
    }
    for (int i = 1; i < 10; i++)
    {
        rki = _mm_loadu_si128(rk + i);
        for (int j = 0; j < N; j++)
        {
            state[j] = _mm_aesenc_si128(state[j], rki);
        }
    }
    rki = _mm_loadu_si128(rk + 10);
    for (int j = 0; j < N; j++)
    {
        state[j] = _mm_aesenclast_si128(state[j], rki);
    }

    for (int j = 0; j < N; j++)
    {
        _mm_storeu_si128((__m128i*)ciphertext + j, state[j]);
    }
}
inline void aes128_aesni_enc_blocks(const std::uint8_t round_key[11][16],
                                    uint8_t*           ciphertext,
                                    const uint8_t*     plaintext,
                                    size_t             block_num)
{
    while (block_num >= 4)
    {
        aes128_aesni_compute_n_block<4>(round_key, ciphertext, plaintext);
        ciphertext += 16 * 4;
        plaintext += 16 * 4;
        block_num -= 1 * 4;
    }

    while (block_num)
    {
        aes128_aesni_compute_n_block<1>(round_key, ciphertext, plaintext);
        ciphertext += 16;
        plaintext += 16;
        block_num -= 1;
    }
}

}; // namespace internal

class AES128
{
public:
    static inline void aes128_enc1_block(void*       ciphertext,
                                         const void* plaintext,
                                         const void* user_key) noexcept
    {
        std::uint8_t rk[11][16];
        internal::aes128_aesni_enc_key_init(rk, (const std::uint8_t*)user_key);
        internal::aes128_aesni_compute_n_block<1>(rk, ciphertext, plaintext);
    }

    static inline void aes128_enc2_block(void*       ciphertext,
                                         const void* plaintext,
                                         const void* user_key) noexcept
    {
        std::uint8_t rk[11][16];
        internal::aes128_aesni_enc_key_init(rk, (const std::uint8_t*)user_key);
        internal::aes128_aesni_compute_n_block<2>(rk, ciphertext, plaintext);
    }

    static inline void aes128_enc4_block(void*       ciphertext,
                                         const void* plaintext,
                                         const void* user_key) noexcept
    {
        std::uint8_t rk[11][16];
        internal::aes128_aesni_enc_key_init(rk, (const std::uint8_t*)user_key);
        internal::aes128_aesni_compute_n_block<4>(rk, ciphertext, plaintext);
    }

public:
    inline void set_enc_key(const void* user_key) noexcept
    {
        internal::aes128_aesni_enc_key_init(rk_, (const std::uint8_t*)user_key);
    }

    template <int N>
    inline void enc_n_block(void* out, const void* in) noexcept
    {
        internal::aes128_aesni_compute_n_block<N>(rk_, out, in);
    }

    inline void enc_blocks(void*       out,
                           const void* in,
                           std::size_t block_num) noexcept
    {
        internal::aes128_aesni_enc_blocks(rk_, (std::uint8_t*)out,
                                          (const std::uint8_t*)in, block_num);
    }

private:
    std::uint8_t rk_[11][16];
};

} // namespace FastFss::impl

#endif