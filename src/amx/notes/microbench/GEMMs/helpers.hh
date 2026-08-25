#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdfloat>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

// Request OS permission to use AMX tile instructions (for linux)
inline bool
enable_amx()
{
    constexpr int ARCH_REQ_XCOMP_PERM = 0x1023;
    constexpr int XFEATURE_XTILEDATA = 18;
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}

// 64-byte AMX tile connfig struct
{
    std::uint8_t palette_id;
    std::uint8_t start_row;
    std::uint8_t reserved[14];
    std::uint16_t colsb[16];
    std::uint8_t rows[16];
};

// Generic AMX BF16 initialization
inline void
init_amx_bf16(int num_accumulators = 4, int num_other_tiles = 4, int rows = 16)
{
    TileConfig config = {};
    config.palette_id = 1;

    // initialize the acccumalator tiles 16 x 16 float numbers
    for (int t = 0; t < num_accumulators && t < 8; ++t) {
        config.rows[t] = rows;
        config.colsb[t] = 16 * sizeof(float);
    }

    // Input tiles: 16 rows x 32 BF16 numbers
    for (int t = num_accumulators; t < (num_accumulators + num_other_tiles) && t < 8; ++t) {
        config.rows[t] = rows;
        config.colsb[t] = 32 * sizeof(std::bfloat16_t);
    }

    _tile_loadconfig(&config);
}

// Pack row-major B[K][N] into VNNI layout B[K/2][N][2] for TDPBF16PS.
inline std::vector<std::bfloat16_t>
pack_vnni_b(const std::bfloat16_t *B, int K, int N)
{
    std::vector<std::bfloat16_t> packed_B(K * N);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            packed_B[(k / 2) * (2 * N) + 2 * n + (k % 2)] = B[k * N + n];
        }
    }
    return packed_B;
}
