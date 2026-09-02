#pragma once

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <stdfloat>

#include "helpers.hh"

// Hardware tile dimensions for AMX BF16 matrix multiplication
// TMxTK * TKxTN -> TMxTN (16x16 output tile)
constexpr int TM = 16;
constexpr int TN = 16;
constexpr int TK = 32;

// 1x1 tile AMX GEMM baseline with unaligned loads

inline void
bf16_gemm_none(const std::bfloat16_t *A, const std::bfloat16_t *B, float *C,
               int M, int N, int K)
{
    init_amx_bf16(1, 2);

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

// 1x1 tile AMX GEMM baseline with 64-byte aligned memory loads
inline void
bf16_gemm_aligned(const std::bfloat16_t *A, const std::bfloat16_t *B, float *C,
                  int M, int N, int K)
{
    init_amx_bf16(1, 2);

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

// 1x1 tile AMX GEMM with 64-byte aligned memory loads and Continuous Tile Layout for Matrix B
inline void
bf16_gemm_aligned_cont_b(const std::bfloat16_t *A, const std::bfloat16_t *B,
                         float *C, int M, int N, int K)
{
    init_amx_bf16(1, 2);

    const int stride_a = K * sizeof(std::bfloat16_t);
    constexpr int stride_b = 32 * sizeof(std::bfloat16_t); 
    const int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += TM) {
        for (int n = 0; n < N; n += 2 * TN) {
            for (int sub_n = 0; sub_n < 2; ++sub_n) {
                _tile_zero(0);

                const std::bfloat16_t *a = A + m * K;
                const std::bfloat16_t *b = B + n * K + sub_n * 512;

                for (int k = 0; k < K; k += TK) {
                    _tile_loadd(1, a, stride_a);
                    _tile_loadd(2, b, stride_b);
                    _tile_dpbf16ps(0, 1, 2);

                    a += TK;
                    b += 1024;
                }

                _tile_stored(0, C + m * N + n + sub_n * TN, stride_c);
            }
        }
    }

    _tile_release();
}

// 1x1 tile AMX GEMM with 64-byte aligned memory loads and Continuous Tile Layout for Matrix A & Matrix B
inline void
bf16_gemm_aligned_cont_ab(const std::bfloat16_t *A, const std::bfloat16_t *B,
                          float *C, int M, int N, int K)
{
    init_amx_bf16(1, 2);

    constexpr int stride_a = 32 * sizeof(std::bfloat16_t); // 64 bytes
    constexpr int stride_b = 32 * sizeof(std::bfloat16_t); // 64 bytes
    const int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += TM) {
        for (int n = 0; n < N; n += 2 * TN) {
            for (int sub_n = 0; sub_n < 2; ++sub_n) {
                _tile_zero(0);

                const std::bfloat16_t *a = A + m * K;
                const std::bfloat16_t *b = B + n * K + sub_n * 512;

                for (int k = 0; k < K; k += TK) {
                    _tile_loadd(1, a, stride_a);
                    _tile_loadd(2, b, stride_b);
                    _tile_dpbf16ps(0, 1, 2);

                    a += TM * TK; // 512 elements = 1024 bytes
                    b += 1024;
                }

                _tile_stored(0, C + m * N + n + sub_n * TN, stride_c);
            }
        }
    }

    _tile_release();
}

// 2x2 tile AMX GEMM with Continuous Tile Layout for both Matrix A and Matrix B
inline void
intel_manual(const std::bfloat16_t *A, const std::bfloat16_t *B,
             float *C, int M, int N, int K)
{
    init_amx_bf16(4, 4);

    constexpr int stride_a = 32 * sizeof(std::bfloat16_t); 
    constexpr int stride_b = 32 * sizeof(std::bfloat16_t); 
    const int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += 2 * TM) {
        for (int n = 0; n < N; n += 2 * TN) {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            const std::bfloat16_t *a_top = A + m * K;
            const std::bfloat16_t *a_bot = A + (m + TM) * K;
            const std::bfloat16_t *b_ptr = B + n * K;

            for (int k = 0; k < K; k += TK) {
                const std::bfloat16_t *b_left = b_ptr;
                const std::bfloat16_t *b_right = b_ptr + 512; // 512 elements = 1024 bytes

                _tile_loadd(4, a_top, stride_a);
                _tile_loadd(6, b_left, stride_b);
                _tile_dpbf16ps(0, 4, 6); // C00 += A_top * B_left

                _tile_loadd(5, a_bot, stride_a);
                _tile_dpbf16ps(2, 5, 6); // C10 += A_bot * B_left

                _tile_loadd(7, b_right, stride_b);
                _tile_dpbf16ps(1, 4, 7); // C01 += A_top * B_right
                _tile_dpbf16ps(3, 5, 7); // C11 += A_bot * B_right

                a_top += TM * TK; // 512 elements = 1024 bytes
                a_bot += TM * TK; // 512 elements = 1024 bytes
                b_ptr += 1024;    // 1024 elements = 2048 bytes
            }

            _tile_stored(0, C + m * N + n, stride_c);
            _tile_stored(1, C + m * N + n + TN, stride_c);
            _tile_stored(2, C + (m + TM) * N + n, stride_c);
            _tile_stored(3, C + (m + TM) * N + n + TN, stride_c);
        }
    }

    _tile_release();
}

// 3x2 tile AMX GEMM with Continuous Tile Layout for both Matrix A and Matrix B

inline void
bf16_gemm_6acc(const std::bfloat16_t *A,
               const std::bfloat16_t *B, float *C, int M, int N,
               int K)
{
    init_amx_bf16(6, 2);

    constexpr int stride_a = 32 * sizeof(std::bfloat16_t); 
    constexpr int stride_b = 32 * sizeof(std::bfloat16_t); 
    const int stride_c = N * sizeof(float);

    for (int m = 0; m < M;) {
        if (m + 3 * TM <= M) {
            for (int n = 0; n < N; n += 2 * TN) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
                _tile_zero(4);
                _tile_zero(5);

                const std::bfloat16_t *a0 = A + (m + 0 * TM) * K;
                const std::bfloat16_t *a1 = A + (m + 1 * TM) * K;
                const std::bfloat16_t *a2 = A + (m + 2 * TM) * K;

                const std::bfloat16_t *b_ptr = B + n * K;

                for (int k = 0; k < K; k += TK) {
                    const std::bfloat16_t *b_left = b_ptr;
                    const std::bfloat16_t *b_right = b_ptr + 512; // 512 elements = 1024 bytes

                    _tile_loadd(6, b_left, stride_b);

                    _tile_loadd(7, a0, stride_a);
                    _tile_dpbf16ps(0, 7, 6); // C00 += A0 * B_left

                    _tile_loadd(7, a1, stride_a);
                    _tile_dpbf16ps(2, 7, 6); // C10 += A1 * B_left

                    _tile_loadd(7, a2, stride_a);
                    _tile_dpbf16ps(4, 7, 6); // C20 += A2 * B_left

                    _tile_loadd(6, b_right, stride_b);
                    _tile_dpbf16ps(5, 7, 6); // C21 += A2 * B_right

                    _tile_loadd(7, a1, stride_a);
                    _tile_dpbf16ps(3, 7, 6); // C11 += A1 * B_right

                    _tile_loadd(7, a0, stride_a);
                    _tile_dpbf16ps(1, 7, 6); // C01 += A0 * B_right

                    a0 += TM * TK;
                    a1 += TM * TK;
                    a2 += TM * TK;
                    b_ptr += 1024;
                }

                _tile_stored(0, C + (m + 0 * TM) * N + n, stride_c);
                _tile_stored(1, C + (m + 0 * TM) * N + n + TN, stride_c);
                _tile_stored(2, C + (m + 1 * TM) * N + n, stride_c);
                _tile_stored(3, C + (m + 1 * TM) * N + n + TN, stride_c);
                _tile_stored(4, C + (m + 2 * TM) * N + n, stride_c);
                _tile_stored(5, C + (m + 2 * TM) * N + n + TN, stride_c);
            }
            m += 3 * TM;
        } else if (m + 2 * TM <= M) {
            for (int n = 0; n < N; n += 2 * TN) {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);

                const std::bfloat16_t *a0 = A + (m + 0 * TM) * K;
                const std::bfloat16_t *a1 = A + (m + 1 * TM) * K;
                const std::bfloat16_t *b_ptr = B + n * K;

                for (int k = 0; k < K; k += TK) {
                    const std::bfloat16_t *b_left = b_ptr;
                    const std::bfloat16_t *b_right = b_ptr + 512;

                    _tile_loadd(6, b_left, stride_b);
                    _tile_loadd(7, a0, stride_a);
                    _tile_dpbf16ps(0, 7, 6);

                    _tile_loadd(7, a1, stride_a);
                    _tile_dpbf16ps(2, 7, 6);

                    _tile_loadd(6, b_right, stride_b);
                    _tile_dpbf16ps(3, 7, 6);

                    _tile_loadd(7, a0, stride_a);
                    _tile_dpbf16ps(1, 7, 6);

                    a0 += TM * TK;
                    a1 += TM * TK;
                    b_ptr += 1024;
                }

                _tile_stored(0, C + (m + 0 * TM) * N + n, stride_c);
                _tile_stored(1, C + (m + 0 * TM) * N + n + TN, stride_c);
                _tile_stored(2, C + (m + 1 * TM) * N + n, stride_c);
                _tile_stored(3, C + (m + 1 * TM) * N + n + TN, stride_c);
            }
            m += 2 * TM;
        } else {
            for (int n = 0; n < N; n += 2 * TN) {
                _tile_zero(0);
                _tile_zero(1);

                const std::bfloat16_t *a0 = A + m * K;
                const std::bfloat16_t *b_ptr = B + n * K;

                for (int k = 0; k < K; k += TK) {
                    const std::bfloat16_t *b_left = b_ptr;
                    const std::bfloat16_t *b_right = b_ptr + 512;

                    _tile_loadd(6, b_left, stride_b);
                    _tile_loadd(7, a0, stride_a);
                    _tile_dpbf16ps(0, 7, 6);

                    _tile_loadd(6, b_right, stride_b);
                    _tile_dpbf16ps(1, 7, 6);

                    a0 += TM * TK;
                    b_ptr += 1024;
                }

                _tile_stored(0, C + m * N + n, stride_c);
                _tile_stored(1, C + m * N + n + TN, stride_c);
            }
            m += TM;
        }
    }

    _tile_release();
}

