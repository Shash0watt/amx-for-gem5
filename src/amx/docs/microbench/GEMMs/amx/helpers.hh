#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdfloat>
#include <sys/syscall.h>
#include <unistd.h>

// Request OS permission to use AMX tile instructions (for linux)
inline bool
enable_amx()
{
    constexpr int ARCH_REQ_XCOMP_PERM = 0x1023;
    constexpr int XFEATURE_XTILEDATA = 18;
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}

// 64-byte AMX tile config struct
struct TileConfig
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

    // initialize the accumulator tiles 16 x 16 float numbers
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

// Pack row-major B[K][N] into VNNI layout dst[K/2][2*N] for TDPBF16PS.
inline void
pack_vnni_b(const std::bfloat16_t *B, std::bfloat16_t *dst, int K, int N)
{
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            dst[(k / 2) * (2 * N) + 2 * n + (k % 2)] = B[k * N + n];
        }
    }
}

// Reorganizes row-major matrix B[K][N] into Continuous Tile Layout dst for AMX GEMM.
inline void
pack_continuous_b(const std::bfloat16_t *B, std::bfloat16_t *dst, int K, int N)
{
    std::size_t out_idx = 0;
    constexpr int TN = 16;
    constexpr int TK = 32;

    // Traverse matrix B in 32-column vertical panels (two 16-column tiles: left & right)
    for (int n = 0; n < N; n += 2 * TN) {
        // Step down the panel along K in chunks of 32 rows
        for (int k = 0; k < K; k += TK) {
            // Pack Tile 0 (Left tile: columns n to n+16, rows k to k+32)
            for (int r = 0; r < 16; ++r) {
                int k_even = k + 2 * r;
                int k_odd  = k + 2 * r + 1;
                for (int c = 0; c < 16; ++c) {
                    dst[out_idx++] = B[k_even * N + (n + c)];
                    dst[out_idx++] = B[k_odd  * N + (n + c)];
                }
            }

            // Pack Tile 1 (Right tile: columns n+16 to n+32, rows k to k+32)
            for (int r = 0; r < 16; ++r) {
                int k_even = k + 2 * r;
                int k_odd  = k + 2 * r + 1;
                for (int c = 0; c < 16; ++c) {
                    dst[out_idx++] = B[k_even * N + (n + 16 + c)];
                    dst[out_idx++] = B[k_odd  * N + (n + 16 + c)];
                }
            }
        }
    }
}

// Reorganizes row-major matrix A[M][K] into Continuous Tile Layout dst for AMX GEMM.
inline void
pack_continuous_a(const std::bfloat16_t *A, std::bfloat16_t *dst, int M, int K)
{
    std::size_t out_idx = 0;
    constexpr int TM = 16;
    constexpr int TK = 32;

    for (int m = 0; m < M; m += TM) {
        for (int k = 0; k < K; k += TK) {
            for (int r = 0; r < TM; ++r) {
                for (int c = 0; c < TK; ++c) {
                    dst[out_idx++] = A[(m + r) * K + (k + c)];
                }
            }
        }
    }
}

