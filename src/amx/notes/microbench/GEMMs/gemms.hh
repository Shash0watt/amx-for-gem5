#pragma once

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <stdfloat>

#include "helpers.hh"

// Baseline naive gemm
// load into 2 tiles, accumalate, repeat
// literally no optimisations
inline void
bf16_gemm_none(const std::bfloat16_t *A, const std::bfloat16_t *B, float *C,
               int M, int N, int K)
{
    init_amx_bf16(1, 2);

    constexpr int TM = 16;
    constexpr int TN = 16;
    constexpr int TK = 32;

    const int stride_a = K * sizeof(std::bfloat16_t);
    const int stride_b = 2 * N * sizeof(std::bfloat16_t);
    const int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += TM) {
        for (int n = 0; n < N; n += TN) {
            _tile_zero(0);

            for (int k = 0; k < K; k += TK) {
                const std::bfloat16_t *a = A + m * K + k;
                const std::bfloat16_t *b = B + (k / 2) * (2 * N) + 2 * n;

                _tile_loadd(1, a, stride_a);
                _tile_loadd(2, b, stride_b);
                _tile_dpbf16ps(0, 1, 2);
            }

            _tile_stored(0, C + m * N + n, stride_c);
        }
    }

    _tile_release();
}

// 2x2 tile gemm
// 4 accumulators (32x32 C block)
// loads all 4 input tiles upfront, then computes 4 dot products sequentially
inline void
bf16_gemm_low(const std::bfloat16_t *A, const std::bfloat16_t *B, float *C,
              int M, int N, int K)
{
    init_amx_bf16(4, 4);

    constexpr int TM = 16;
    constexpr int TN = 16;
    constexpr int TK = 32;

    const int stride_a = K * sizeof(std::bfloat16_t);
    const int stride_b = 2 * N * sizeof(std::bfloat16_t);
    const int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += 2 * TM) {
        for (int n = 0; n < N; n += 2 * TN) {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            for (int k = 0; k < K; k += TK) {
                const std::bfloat16_t *a_top = A + m * K + k;
                const std::bfloat16_t *a_bot = A + (m + TM) * K + k;
                const std::bfloat16_t *b_left = B + (k / 2) * (2 * N) + 2 * n;
                const std::bfloat16_t *b_right = b_left + 2 * TN;

                // Load all tiles first
                _tile_loadd(4, a_top, stride_a);
                _tile_loadd(5, a_bot, stride_a);
                _tile_loadd(6, b_left, stride_b);
                _tile_loadd(7, b_right, stride_b);

                // Compute dot products
                _tile_dpbf16ps(0, 4, 6); // C00 += A_top * B_left
                _tile_dpbf16ps(1, 4, 7); // C01 += A_top * B_right
                _tile_dpbf16ps(2, 5, 6); // C10 += A_bot * B_left
                _tile_dpbf16ps(3, 5, 7); // C11 += A_bot * B_right
            }

            _tile_stored(0, C + m * N + n, stride_c);
            _tile_stored(1, C + m * N + n + TN, stride_c);
            _tile_stored(2, C + (m + TM) * N + n, stride_c);
            _tile_stored(3, C + (m + TM) * N + n + TN, stride_c);
        }
    }

    _tile_release();
}

// 2x2 tile gemm
// interleaves tile loads and compute to hide memory latency
inline void
bf16_gemm_med(const std::bfloat16_t *A, const std::bfloat16_t *B, float *C,
              int M, int N, int K)
{
    init_amx_bf16(4, 4);

    constexpr int TM = 16;
    constexpr int TN = 16;
    constexpr int TK = 32;

    const int stride_a = K * sizeof(std::bfloat16_t);
    const int stride_b = 2 * N * sizeof(std::bfloat16_t);
    const int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += 2 * TM) {
        for (int n = 0; n < N; n += 2 * TN) {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            for (int k = 0; k < K; k += TK) {
                const std::bfloat16_t *a_top = A + m * K + k;
                const std::bfloat16_t *a_bot = A + (m + TM) * K + k;
                const std::bfloat16_t *b_left = B + (k / 2) * (2 * N) + 2 * n;
                const std::bfloat16_t *b_right = b_left + 2 * TN;

                _tile_loadd(4, a_top, stride_a);
                _tile_loadd(6, b_left, stride_b);
                _tile_dpbf16ps(0, 4, 6); // C00 += A_top * B_left

                _tile_loadd(5, a_bot, stride_a);
                _tile_dpbf16ps(2, 5, 6); // C10 += A_bot * B_left

                _tile_loadd(7, b_right, stride_b);
                _tile_dpbf16ps(1, 4, 7); // C01 += A_top * B_right
                _tile_dpbf16ps(3, 5, 7); // C11 += A_bot * B_right
            }

            _tile_stored(0, C + m * N + n, stride_c);
            _tile_stored(1, C + m * N + n + TN, stride_c);
            _tile_stored(2, C + (m + TM) * N + n, stride_c);
            _tile_stored(3, C + (m + TM) * N + n + TN, stride_c);
        }
    }

    _tile_release();
}

// 2x2 tile gemm 
// 256x256 cache blocking for L2 locality + software prefetching for next tiles
inline void
bf16_gemm_opt(const std::bfloat16_t *A, const std::bfloat16_t *B, float *C,
              int M, int N, int K)
{
    init_amx_bf16(4, 4);

    constexpr int TM = 16;
    constexpr int TN = 16;
    constexpr int TK = 32;
    constexpr int MC_BLOCK = 256;
    constexpr int NC_BLOCK = 256;

    const int stride_a = K * sizeof(std::bfloat16_t);
    const int stride_b = 2 * N * sizeof(std::bfloat16_t);
    const int stride_c = N * sizeof(float);

    for (int nc = 0; nc < N; nc += NC_BLOCK) {
        const int nc_end = std::min(nc + NC_BLOCK, N);
        for (int mc = 0; mc < M; mc += MC_BLOCK) {
            const int mc_end = std::min(mc + MC_BLOCK, M);

            for (int m = mc; m < mc_end; m += 2 * TM) {
                for (int n = nc; n < nc_end; n += 2 * TN) {
                    _tile_zero(0);
                    _tile_zero(1);
                    _tile_zero(2);
                    _tile_zero(3);

                    for (int k = 0; k < K; k += TK) {
                        const std::bfloat16_t *a_top = A + m * K + k;
                        const std::bfloat16_t *a_bot = A + (m + TM) * K + k;
                        const std::bfloat16_t *b_left = B + (k / 2) * (2 * N) + 2 * n;
                        const std::bfloat16_t *b_right = b_left + 2 * TN;

                        if (k + TK < K) {
                            _mm_prefetch(reinterpret_cast<const char*>(a_top + TK), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(b_left + TK * N), _MM_HINT_T0);
                        }

                        _tile_loadd(4, a_top, stride_a);
                        _tile_loadd(6, b_left, stride_b);
                        _tile_dpbf16ps(0, 4, 6);

                        _tile_loadd(5, a_bot, stride_a);
                        _tile_dpbf16ps(2, 5, 6);

                        _tile_loadd(7, b_right, stride_b);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_dpbf16ps(3, 5, 7);
                    }

                    _tile_stored(0, C + m * N + n, stride_c);
                    _tile_stored(1, C + m * N + n + TN, stride_c);
                    _tile_stored(2, C + (m + TM) * N + n, stride_c);
                    _tile_stored(3, C + (m + TM) * N + n + TN, stride_c);
                }
            }
        }
    }

    _tile_release();
}
