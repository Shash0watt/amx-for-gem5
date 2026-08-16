#pragma once

#include <cstdint>
#include <immintrin.h>
#include <stdfloat>
// * Compile with C++23, AMX-TILE, and AMX-BF16 enabled.

// Requirements:
// * A is row-major M x K.
// * B is a VNNI-packed K x N matrix. For each pair of K rows, store:
//     B[k][0], B[k+1][0], B[k][1], B[k+1][1], ...
// * C is row-major M x N and is overwritten.
// * M and N are multiples of 32; K is a multiple of 32.
// * The caller has requested XTILEDATA permission.
// No accumulator blocking or load/dot-product software pipelining.
inline void
bf16_gemm_none(const std::bfloat16_t *A, const std::bfloat16_t *B, float *C,
               int M, int N, int K)
{
    constexpr int tile_m = 16;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;

    struct alignas(64) TileConfig
    {
        std::uint8_t palette_id;
        std::uint8_t start_row;
        std::uint8_t reserved[14];
        std::uint16_t colsb[16];
        std::uint8_t rows[16];
    };

    static_assert(sizeof(TileConfig) == 64);

    TileConfig config = {};
    config.palette_id = 1;

    // TMM0: 16 x 16 FP32 accumulator tile.
    config.rows[0] = tile_m;
    config.colsb[0] = tile_n * sizeof(float);

    // TMM1: 16 x 32 BF16 A tile.
    config.rows[1] = tile_m;
    config.colsb[1] = tile_k * sizeof(std::bfloat16_t);

    // TMM2: packed 32 x 16 BF16 B tile.
    config.rows[2] = tile_k / 2;
    config.colsb[2] = tile_n * 2 * sizeof(std::bfloat16_t);

    _tile_loadconfig(&config);

    const int stride_a = K * sizeof(std::bfloat16_t);
    const int stride_b = 2 * N * sizeof(std::bfloat16_t);
    const int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += tile_m) {
        for (int n = 0; n < N; n += tile_n) {
            _tile_zero(0);

            for (int k = 0; k < K; k += tile_k) {
                const std::bfloat16_t *a = A + m * K + k;
                const std::bfloat16_t *b =
                    B + (k / 2) * (2 * N) + 2 * n;

                _tile_loadd(1, a, stride_a);
                _tile_loadd(2, b, stride_b);
                _tile_dpbf16ps(0, 1, 2);
            }

            _tile_stored(0, C + m * N + n, stride_c);
        }
    }

    _tile_release();
}

// Optimisations: 2D accumalators, minimal sw pipelining
inline void
bf16_gemm_med(const std::bfloat16_t *A, const std::bfloat16_t *B, float *C,
              int M, int N, int K)
{
    constexpr int tile_m = 16;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;

    struct alignas(64) TileConfig
    {
        std::uint8_t palette_id;
        std::uint8_t start_row;
        std::uint8_t reserved[14];
        std::uint16_t colsb[16];
        std::uint8_t rows[16];
    };

    static_assert(sizeof(TileConfig) == 64);

    TileConfig config = {};
    config.palette_id = 1;

    // TMM0 to TMM3: four 16 x 16 FP32 accumulator tiles.
    for (int tile = 0; tile < 4; ++tile) {
        config.rows[tile] = tile_m;
        config.colsb[tile] = tile_n * sizeof(float);
    }

    // TMM4 & TMM5: two 16 x 32 BF16 A tiles.
    for (int tile = 4; tile < 6; ++tile) {
        config.rows[tile] = tile_m;
        config.colsb[tile] = tile_k * sizeof(std::bfloat16_t);
    }

    // TMM6 & TMM7: two packed 32 x 16 BF16 B tiles.
    for (int tile = 6; tile < 8; ++tile) {
        config.rows[tile] = tile_k / 2;
        config.colsb[tile] = tile_n * 2 * sizeof(std::bfloat16_t);
    }

    _tile_loadconfig(&config);

    const int stride_a = K * sizeof(std::bfloat16_t);
    const int stride_b = 2 * N * sizeof(std::bfloat16_t);
    const int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += 2 * tile_m) {
        for (int n = 0; n < N; n += 2 * tile_n) {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            for (int k = 0; k < K; k += tile_k) {
                const std::bfloat16_t *a_top = A + m * K + k;
                const std::bfloat16_t *a_bottom = A + (m + tile_m) * K + k;
                const std::bfloat16_t *b_left = B + (k / 2) * (2 * N) + 2 * n;
                const std::bfloat16_t *b_right = b_left + 2 * tile_n;

                _tile_loadd(4, a_top, stride_a);  // TMM4 (A[m][k])
                _tile_loadd(6, b_left, stride_b); // TMM6 (B[k][n])

                // TMM0: C[m][n] += TMM4 * TMM6
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(5, a_bottom, stride_a); // TMM5 (A[m+tile_m][k])

                // TMM2: C[m+tile_m][n] += TMM5 * TMM6
                _tile_dpbf16ps(2, 5, 6);
                _tile_loadd(7, b_right, stride_b); // TMM7 (B[k][n+tile_n])

                // TMM1: C[m][n+tile_n] += TMM4 * TMM7
                _tile_dpbf16ps(1, 4, 7);

                // TMM3: C[m+tile_m][n+tile_n] += TMM5 * TMM7
                _tile_dpbf16ps(3, 5, 7);
            }

            _tile_stored(0, C + m * N + n, stride_c);
            _tile_stored(1, C + m * N + n + tile_n, stride_c);
            _tile_stored(2, C + (m + tile_m) * N + n, stride_c);
            _tile_stored(3, C + (m + tile_m) * N + n + tile_n, stride_c);
        }
    }

    _tile_release();
}

// Version without load/dot-product software pipelining.
// It has the same requirements and tile mapping as bf16gemm_mid.
inline void
bf16_gemm_low(const std::bfloat16_t *A, const std::bfloat16_t *B, float *C,
             int M, int N, int K)
{
    constexpr int tile_m = 16;
    constexpr int tile_n = 16;
    constexpr int tile_k = 32;

    struct alignas(64) TileConfig
    {
        std::uint8_t palette_id;
        std::uint8_t start_row;
        std::uint8_t reserved[14];
        std::uint16_t colsb[16];
        std::uint8_t rows[16];
    };

    static_assert(sizeof(TileConfig) == 64);

    TileConfig config = {};
    config.palette_id = 1;

    for (int tile = 0; tile < 4; ++tile) {
        config.rows[tile] = tile_m;
        config.colsb[tile] = tile_n * sizeof(float);
    }

    for (int tile = 4; tile < 6; ++tile) {
        config.rows[tile] = tile_m;
        config.colsb[tile] = tile_k * sizeof(std::bfloat16_t);
    }

    for (int tile = 6; tile < 8; ++tile) {
        config.rows[tile] = tile_k / 2;
        config.colsb[tile] = tile_n * 2 * sizeof(std::bfloat16_t);
    }

    _tile_loadconfig(&config);

    const int stride_a = K * sizeof(std::bfloat16_t);
    const int stride_b = 2 * N * sizeof(std::bfloat16_t);
    const int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += 2 * tile_m) {
        for (int n = 0; n < N; n += 2 * tile_n) {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            for (int k = 0; k < K; k += tile_k) {
                const std::bfloat16_t *a_top = A + m * K + k;
                const std::bfloat16_t *a_bottom = A + (m + tile_m) * K + k;
                const std::bfloat16_t *b_left = B + (k / 2) * (2 * N) + 2 * n;
                const std::bfloat16_t *b_right = b_left + 2 * tile_n;

                _tile_loadd(4, a_top, stride_a);
                _tile_loadd(5, a_bottom, stride_a);
                _tile_loadd(6, b_left, stride_b);
                _tile_loadd(7, b_right, stride_b);

                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }

            _tile_stored(0, C + m * N + n, stride_c);
            _tile_stored(1, C + m * N + n + tile_n, stride_c);
            _tile_stored(2, C + (m + tile_m) * N + n, stride_c);
            _tile_stored(3, C + (m + tile_m) * N + n + tile_n, stride_c);
        }
    }

    _tile_release();
}
