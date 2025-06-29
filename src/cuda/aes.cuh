#ifndef SRC_CUDA_AES_CUH
#define SRC_CUDA_AES_CUH

#include <cstddef>
#include <cstdint>

namespace FastFss::impl {

#define MEM_LOAD32BE(src)                                  \
    (((std::uint32_t)(((std::uint8_t *)(src))[0]) << 24) | \
     ((std::uint32_t)(((std::uint8_t *)(src))[1]) << 16) | \
     ((std::uint32_t)(((std::uint8_t *)(src))[2]) << 8) |  \
     ((std::uint32_t)(((std::uint8_t *)(src))[3]) << 0))

#define MEM_STORE32BE(dst, a)                                        \
    (((std::uint8_t *)(dst))[0] = ((std::uint32_t)(a) >> 24) & 0xFF, \
     ((std::uint8_t *)(dst))[1] = ((std::uint32_t)(a) >> 16) & 0xFF, \
     ((std::uint8_t *)(dst))[2] = ((std::uint32_t)(a) >> 8) & 0xFF,  \
     ((std::uint8_t *)(dst))[3] = ((std::uint32_t)(a) >> 0) & 0xFF)

namespace internal {

__device__ static const std::uint32_t TE0[256] = {
    0xc66363a5, 0xf87c7c84, 0xee777799, 0xf67b7b8d, 0xfff2f20d, 0xd66b6bbd,
    0xde6f6fb1, 0x91c5c554, 0x60303050, 0x02010103, 0xce6767a9, 0x562b2b7d,
    0xe7fefe19, 0xb5d7d762, 0x4dababe6, 0xec76769a, 0x8fcaca45, 0x1f82829d,
    0x89c9c940, 0xfa7d7d87, 0xeffafa15, 0xb25959eb, 0x8e4747c9, 0xfbf0f00b,
    0x41adadec, 0xb3d4d467, 0x5fa2a2fd, 0x45afafea, 0x239c9cbf, 0x53a4a4f7,
    0xe4727296, 0x9bc0c05b, 0x75b7b7c2, 0xe1fdfd1c, 0x3d9393ae, 0x4c26266a,
    0x6c36365a, 0x7e3f3f41, 0xf5f7f702, 0x83cccc4f, 0x6834345c, 0x51a5a5f4,
    0xd1e5e534, 0xf9f1f108, 0xe2717193, 0xabd8d873, 0x62313153, 0x2a15153f,
    0x0804040c, 0x95c7c752, 0x46232365, 0x9dc3c35e, 0x30181828, 0x379696a1,
    0x0a05050f, 0x2f9a9ab5, 0x0e070709, 0x24121236, 0x1b80809b, 0xdfe2e23d,
    0xcdebeb26, 0x4e272769, 0x7fb2b2cd, 0xea75759f, 0x1209091b, 0x1d83839e,
    0x582c2c74, 0x341a1a2e, 0x361b1b2d, 0xdc6e6eb2, 0xb45a5aee, 0x5ba0a0fb,
    0xa45252f6, 0x763b3b4d, 0xb7d6d661, 0x7db3b3ce, 0x5229297b, 0xdde3e33e,
    0x5e2f2f71, 0x13848497, 0xa65353f5, 0xb9d1d168, 0x00000000, 0xc1eded2c,
    0x40202060, 0xe3fcfc1f, 0x79b1b1c8, 0xb65b5bed, 0xd46a6abe, 0x8dcbcb46,
    0x67bebed9, 0x7239394b, 0x944a4ade, 0x984c4cd4, 0xb05858e8, 0x85cfcf4a,
    0xbbd0d06b, 0xc5efef2a, 0x4faaaae5, 0xedfbfb16, 0x864343c5, 0x9a4d4dd7,
    0x66333355, 0x11858594, 0x8a4545cf, 0xe9f9f910, 0x04020206, 0xfe7f7f81,
    0xa05050f0, 0x783c3c44, 0x259f9fba, 0x4ba8a8e3, 0xa25151f3, 0x5da3a3fe,
    0x804040c0, 0x058f8f8a, 0x3f9292ad, 0x219d9dbc, 0x70383848, 0xf1f5f504,
    0x63bcbcdf, 0x77b6b6c1, 0xafdada75, 0x42212163, 0x20101030, 0xe5ffff1a,
    0xfdf3f30e, 0xbfd2d26d, 0x81cdcd4c, 0x180c0c14, 0x26131335, 0xc3ecec2f,
    0xbe5f5fe1, 0x359797a2, 0x884444cc, 0x2e171739, 0x93c4c457, 0x55a7a7f2,
    0xfc7e7e82, 0x7a3d3d47, 0xc86464ac, 0xba5d5de7, 0x3219192b, 0xe6737395,
    0xc06060a0, 0x19818198, 0x9e4f4fd1, 0xa3dcdc7f, 0x44222266, 0x542a2a7e,
    0x3b9090ab, 0x0b888883, 0x8c4646ca, 0xc7eeee29, 0x6bb8b8d3, 0x2814143c,
    0xa7dede79, 0xbc5e5ee2, 0x160b0b1d, 0xaddbdb76, 0xdbe0e03b, 0x64323256,
    0x743a3a4e, 0x140a0a1e, 0x924949db, 0x0c06060a, 0x4824246c, 0xb85c5ce4,
    0x9fc2c25d, 0xbdd3d36e, 0x43acacef, 0xc46262a6, 0x399191a8, 0x319595a4,
    0xd3e4e437, 0xf279798b, 0xd5e7e732, 0x8bc8c843, 0x6e373759, 0xda6d6db7,
    0x018d8d8c, 0xb1d5d564, 0x9c4e4ed2, 0x49a9a9e0, 0xd86c6cb4, 0xac5656fa,
    0xf3f4f407, 0xcfeaea25, 0xca6565af, 0xf47a7a8e, 0x47aeaee9, 0x10080818,
    0x6fbabad5, 0xf0787888, 0x4a25256f, 0x5c2e2e72, 0x381c1c24, 0x57a6a6f1,
    0x73b4b4c7, 0x97c6c651, 0xcbe8e823, 0xa1dddd7c, 0xe874749c, 0x3e1f1f21,
    0x964b4bdd, 0x61bdbddc, 0x0d8b8b86, 0x0f8a8a85, 0xe0707090, 0x7c3e3e42,
    0x71b5b5c4, 0xcc6666aa, 0x904848d8, 0x06030305, 0xf7f6f601, 0x1c0e0e12,
    0xc26161a3, 0x6a35355f, 0xae5757f9, 0x69b9b9d0, 0x17868691, 0x99c1c158,
    0x3a1d1d27, 0x279e9eb9, 0xd9e1e138, 0xebf8f813, 0x2b9898b3, 0x22111133,
    0xd26969bb, 0xa9d9d970, 0x078e8e89, 0x339494a7, 0x2d9b9bb6, 0x3c1e1e22,
    0x15878792, 0xc9e9e920, 0x87cece49, 0xaa5555ff, 0x50282878, 0xa5dfdf7a,
    0x038c8c8f, 0x59a1a1f8, 0x09898980, 0x1a0d0d17, 0x65bfbfda, 0xd7e6e631,
    0x844242c6, 0xd06868b8, 0x824141c3, 0x299999b0, 0x5a2d2d77, 0x1e0f0f11,
    0x7bb0b0cb, 0xa85454fc, 0x6dbbbbd6, 0x2c16163a,
};

}; // namespace internal

__device__ static inline void aes_load_s_box(std::uint32_t (**sram_te0_ptr)[32])
{
    constexpr int NUM_SHARED_MEM_BANKS = 32;
    __shared__ std::uint32_t sram_te0[256][NUM_SHARED_MEM_BANKS];
    if (threadIdx.x < 256)
    {
        for (int bank = 0; bank < NUM_SHARED_MEM_BANKS; ++bank)
        {
            sram_te0[threadIdx.x][bank] = internal::TE0[threadIdx.x];
        }
    }
    __syncthreads();
    *sram_te0_ptr = sram_te0;
}

namespace internal {

__device__ __forceinline__ static std::uint32_t get0(std::uint32_t s)
{
    return s & 0xFF;
}
__device__ __forceinline__ static std::uint32_t get1(std::uint32_t s)
{
    return __byte_perm(s, 0, 0x4441);
}
__device__ __forceinline__ static std::uint32_t get2(std::uint32_t s)
{
    return __byte_perm(s, 0, 0x4442);
}
__device__ __forceinline__ static std::uint32_t get3(std::uint32_t s)
{
    return __byte_perm(s, 0, 0x4443);
}
__device__ __forceinline__ static std::uint32_t rotl1(std::uint32_t s)
{
    return __byte_perm(s, s, 0x4321);
}
__device__ __forceinline__ static std::uint32_t rotl2(std::uint32_t s)
{
    return __byte_perm(s, s, 0x5432);
}
__device__ __forceinline__ static std::uint32_t rotl3(std::uint32_t s)
{
    return __byte_perm(s, s, 0x6543);
}

/**
 * @brief               AES key schedule (encryption)
 * @param round_key     44-dword round key (11-round)
 * @param user_key      16-byte secret key
 */
__device__ static inline void aes128_enc_key_init(
    std::uint32_t      round_key[44],
    const std::uint8_t user_key[16],
    const std::uint32_t (*sram_te0_ptr)[32]) noexcept
{
    int wtid = threadIdx.x % 32;

    static const std::uint32_t Rcon[10] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36,
    };
    constexpr int  nr = 10; // Nr
    constexpr int  nk = 4;  // Nk
    std::uint32_t *w  = round_key;
    //--------------Load as BigEndian---------------
    for (int i = 0; i < nk; i++)
    {
        w[i] = MEM_LOAD32BE(user_key + 4 * i);
    }
    //------------KeyExpand-----------------
    for (int i = nk; i < 4 * (nr + 1); i += 4)
    {
        auto tmp = w[i - 1];
        // tmp = SubWord(RotWord(w[i-1]))
        auto tmp1 = tmp;
        tmp       = get1(sram_te0_ptr[get3(tmp1)][wtid]);
        tmp ^= get1(sram_te0_ptr[get0(tmp1)][wtid]) << 8;
        tmp ^= get1(sram_te0_ptr[get1(tmp1)][wtid]) << 16;
        tmp ^= (get1(sram_te0_ptr[get2(tmp1)][wtid]) ^ Rcon[i / nk - 1]) << 24;

        w[i]     = w[i - nk] ^ tmp;
        w[i + 1] = w[i + 1 - nk] ^ w[i + 1 - 1];
        w[i + 2] = w[i + 2 - nk] ^ w[i + 2 - 1];
        w[i + 3] = w[i + 3 - nk] ^ w[i + 3 - 1];
    }
}

/**
 * @brief       AES Round Function (encryption)
 * @param s     4-dword state (128-bit)
 * @param rk    4-dword round key (128-bit)
 */
__device__ static inline void aes_enc_round(
    std::uint32_t       s[4],
    const std::uint32_t rk[4],
    const std::uint32_t (*sram_te0_ptr)[32]) noexcept
{
    int wtid = threadIdx.x % 32;

    std::uint32_t t[4];
    //-------ShiftRow + SubByte + MixCol-------------
    // t0
    t[0] = sram_te0_ptr[get3(s[0])][wtid] ^
           rotl1(sram_te0_ptr[get2(s[1])][wtid]) ^
           rotl2(sram_te0_ptr[get1(s[2])][wtid]) ^
           rotl3(sram_te0_ptr[get0(s[3])][wtid]) ^ rk[0];

    // t1
    t[1] = sram_te0_ptr[get3(s[1])][wtid] ^
           rotl1(sram_te0_ptr[get2(s[2])][wtid]) ^
           rotl2(sram_te0_ptr[get1(s[3])][wtid]) ^
           rotl3(sram_te0_ptr[get0(s[0])][wtid]) ^ rk[1];
    // t2
    t[2] = sram_te0_ptr[get3(s[2])][wtid] ^
           rotl1(sram_te0_ptr[get2(s[3])][wtid]) ^
           rotl2(sram_te0_ptr[get1(s[0])][wtid]) ^
           rotl3(sram_te0_ptr[get0(s[1])][wtid]) ^ rk[2];
    // t3
    t[3] = sram_te0_ptr[get3(s[3])][wtid] ^
           rotl1(sram_te0_ptr[get2(s[0])][wtid]) ^
           rotl2(sram_te0_ptr[get1(s[1])][wtid]) ^
           rotl3(sram_te0_ptr[get0(s[2])][wtid]) ^ rk[3];

    s[0] = t[0];
    s[1] = t[1];
    s[2] = t[2];
    s[3] = t[3];
}

/**
 * @brief       AES Last Round Function (encryption)
 * @param s     4-dword state (128-bit)
 * @param rk    4-dword round key (128-bit)
 */
__device__ static inline void aes_enc_last(
    std::uint32_t       s[4],
    const std::uint32_t rk[4],
    const std::uint32_t (*sram_te0_ptr)[32]) noexcept
{
    int wtid = threadIdx.x % 32;

    std::uint32_t t[4];
    //------------ShiftRow + SubByte-----------
    // t0
    t[0] = get1(sram_te0_ptr[(s[0] >> 24) & 0xFF][wtid]) << 24;
    t[0] ^= get1(sram_te0_ptr[(s[1] >> 16) & 0xFF][wtid]) << 16;
    t[0] ^= get1(sram_te0_ptr[(s[2] >> 8) & 0xFF][wtid]) << 8;
    t[0] ^= get1(sram_te0_ptr[(s[3] >> 0) & 0xFF][wtid]) << 0;
    // t1
    t[1] = get1(sram_te0_ptr[(s[1] >> 24) & 0xFF][wtid]) << 24;
    t[1] ^= get1(sram_te0_ptr[(s[2] >> 16) & 0xFF][wtid]) << 16;
    t[1] ^= get1(sram_te0_ptr[(s[3] >> 8) & 0xFF][wtid]) << 8;
    t[1] ^= get1(sram_te0_ptr[(s[0] >> 0) & 0xFF][wtid]) << 0;
    // t2
    t[2] = get1(sram_te0_ptr[(s[2] >> 24) & 0xFF][wtid]) << 24;
    t[2] ^= get1(sram_te0_ptr[(s[3] >> 16) & 0xFF][wtid]) << 16;
    t[2] ^= get1(sram_te0_ptr[(s[0] >> 8) & 0xFF][wtid]) << 8;
    t[2] ^= get1(sram_te0_ptr[(s[1] >> 0) & 0xFF][wtid]) << 0;
    // t3
    t[3] = get1(sram_te0_ptr[(s[3] >> 24) & 0xFF][wtid]) << 24;
    t[3] ^= get1(sram_te0_ptr[(s[0] >> 16) & 0xFF][wtid]) << 16;
    t[3] ^= get1(sram_te0_ptr[(s[1] >> 8) & 0xFF][wtid]) << 8;
    t[3] ^= get1(sram_te0_ptr[(s[2] >> 0) & 0xFF][wtid]) << 0;

    //------------AddRoundKey-------------
    s[0] = t[0] ^ rk[0];
    s[1] = t[1] ^ rk[1];
    s[2] = t[2] ^ rk[2];
    s[3] = t[3] ^ rk[3];
}

/**
 * @brief               AES encrypt
 * @param round_key     44-dword round key (11-round)
 * @param ciphertext    16-byte ciphertext
 * @param plaintext     16-byte plaintext
 */
__device__ inline void aes128_compute_block(
    const std::uint32_t round_key[44],
    void               *ciphertext,
    const void         *plaintext,
    const std::uint32_t (*sram_te0_ptr)[32]) noexcept
{
    int           nr = 10;
    std::uint32_t state[4];
    //----------------AddRoundKey----------------
    state[0] = round_key[0] ^ MEM_LOAD32BE((const char *)plaintext);
    state[1] = round_key[1] ^ MEM_LOAD32BE((const char *)plaintext + 4);
    state[2] = round_key[2] ^ MEM_LOAD32BE((const char *)plaintext + 8);
    state[3] = round_key[3] ^ MEM_LOAD32BE((const char *)plaintext + 12);
    // ---------- round -------------
    for (int i = 1; i < nr; i++)
    {
        aes_enc_round(state, round_key + 4 * i, sram_te0_ptr);
    }
    aes_enc_last(state, round_key + 4 * nr, sram_te0_ptr);
    //-----------Store as BigEndian--------------
    MEM_STORE32BE((char *)ciphertext, state[0]);
    MEM_STORE32BE((char *)ciphertext + 4, state[1]);
    MEM_STORE32BE((char *)ciphertext + 8, state[2]);
    MEM_STORE32BE((char *)ciphertext + 12, state[3]);
}

template <int N>
__device__ inline void aes128_compute_n_block(
    const std::uint32_t round_key[44],
    void               *ciphertext,
    const void         *plaintext,
    const std::uint32_t (*sram_te0_ptr)[32]) noexcept
{
    constexpr int nr = 10;
    std::uint32_t state[N][4];
    //----------------AddRoundKey----------------
    for (int j = 0; j < N; j++)
    {
        const char *ptr = (const char *)plaintext + 16 * j;

        state[j][0] = round_key[0] ^ MEM_LOAD32BE((const char *)ptr);
        state[j][1] = round_key[1] ^ MEM_LOAD32BE((const char *)ptr + 4);
        state[j][2] = round_key[2] ^ MEM_LOAD32BE((const char *)ptr + 8);
        state[j][3] = round_key[3] ^ MEM_LOAD32BE((const char *)ptr + 12);
    }
    // ---------- round -------------
    for (int i = 1; i < nr; i++)
    {
        for (int j = 0; j < N; j++)
        {
            aes_enc_round(state[j], round_key + 4 * i, sram_te0_ptr);
        }
    }
    for (int j = 0; j < N; j++)
    {
        aes_enc_last(state[j], round_key + 4 * nr, sram_te0_ptr);
    }
    //-----------Store as BigEndian--------------
    for (int j = 0; j < N; j++)
    {
        char *ptr = (char *)ciphertext + 16 * j;

        MEM_STORE32BE((char *)ptr, state[j][0]);
        MEM_STORE32BE((char *)ptr + 4, state[j][1]);
        MEM_STORE32BE((char *)ptr + 8, state[j][2]);
        MEM_STORE32BE((char *)ptr + 12, state[j][3]);
    }
}

#undef MEM_STORE32BE
#undef MEM_LOAD32BE

}; // namespace internal

class AES128GlobalContext
{
public:
    std::uint32_t (*sram_te0_ptr)[32] = nullptr;

    __device__ AES128GlobalContext()
    {
        aes_load_s_box(&sram_te0_ptr);
    }
};

class AES128
{
public:
    __device__ static void aes128_enc1_block(
        void                      *ciphertext,
        const void                *plaintext,
        const void                *user_key,
        const AES128GlobalContext *ctx) noexcept
    {
        std::uint32_t rk[44];
        internal::aes128_enc_key_init(rk, (const std::uint8_t *)user_key,
                                      ctx->sram_te0_ptr);
        internal::aes128_compute_block(rk, ciphertext, plaintext,
                                       ctx->sram_te0_ptr);
    }

    __device__ static void aes128_enc2_block(
        void                      *ciphertext,
        const void                *plaintext,
        const void                *user_key,
        const AES128GlobalContext *ctx) noexcept
    {
        std::uint32_t rk[44];
        internal::aes128_enc_key_init(rk, (const std::uint8_t *)user_key,
                                      ctx->sram_te0_ptr);
        internal::aes128_compute_n_block<2>(rk, ciphertext, plaintext,
                                            ctx->sram_te0_ptr);
    }

    __device__ static void aes128_enc4_block(
        void                      *ciphertext,
        const void                *plaintext,
        const void                *user_key,
        const AES128GlobalContext *ctx) noexcept
    {
        std::uint32_t rk[44];
        internal::aes128_enc_key_init(rk, (const std::uint8_t *)user_key,
                                      ctx->sram_te0_ptr);
        internal::aes128_compute_n_block<4>(rk, ciphertext, plaintext,
                                            ctx->sram_te0_ptr);
    }

public:
    __device__ void set_enc_key(const void                *user_key,
                                const AES128GlobalContext *ctx) noexcept
    {
        internal::aes128_enc_key_init(rk_, (const std::uint8_t *)user_key,
                                      ctx->sram_te0_ptr);
    }

    __device__ void enc_block(void                      *out,
                              const void                *in,
                              const AES128GlobalContext *ctx) noexcept
    {
        internal::aes128_compute_block(rk_, out, in, ctx->sram_te0_ptr);
    }

    __device__ void enc4_block(void                      *out,
                               const void                *in,
                               const AES128GlobalContext *ctx) noexcept
    {
        internal::aes128_compute_n_block<4>(rk_, out, in, ctx->sram_te0_ptr);
    }

    template <int N>
    __device__ void enc_n_block(void                      *out,
                                const void                *in,
                                const AES128GlobalContext *ctx) noexcept
    {
        internal::aes128_compute_n_block<N>(rk_, out, in, ctx->sram_te0_ptr);
    }

    __device__ void enc_blocks(void                      *out,
                               const void                *in,
                               std::size_t                block_num,
                               const AES128GlobalContext *ctx) noexcept
    {
        std::uint8_t       *out_u8 = (std::uint8_t *)out;
        const std::uint8_t *in_u8  = (const std::uint8_t *)in;
        while (block_num)
        {
            internal::aes128_compute_block(rk_, out_u8, in_u8,
                                           ctx->sram_te0_ptr);
            out_u8 += 16, in_u8 += 16;
            block_num -= 1;
        }
    }

private:
    std::uint32_t rk_[44];
};

}; // namespace FastFss::impl

#endif