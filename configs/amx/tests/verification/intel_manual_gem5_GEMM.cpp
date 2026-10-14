// intel_manual 2x2-tile BF16 GEMM (src/amx/docs/microbench/GEMMs/amx/gemms.hh)
// ported to the amx_* m5ops, run on inputs mixing zeros, integers and decimals.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <gem5/m5ops.h>

// The 64-byte memory layout consumed by AMX_TILE_LOADCONFIG.
struct TileConfig
{
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved0[14];
    uint16_t columnBytes[8];
    uint8_t reserved1[16];
    uint8_t rows[8];
    uint8_t reserved2[8];
};

constexpr int TM = 16;
constexpr int TN = 16;
constexpr int TK = 32;

// A is 16 x 2048, B is 2048 x 512, and C is 16 x 512.
constexpr int M = 16;
constexpr int N = 512;
constexpr int K = 2048;

// The intel_manual kernel: C = A * B with a 2x2 block of accumulator tiles.
void
gemm(const __bf16 *packed_a, const __bf16 *packed_b, float *C)
{
    constexpr int stride_a = 32 * sizeof(__bf16);
    constexpr int stride_b = 32 * sizeof(__bf16);
    constexpr int stride_c = N * sizeof(float);

    for (int m = 0; m < M; m += 2 * TM) {
        const bool has_bottom = m + TM < M;
        for (int n = 0; n < N; n += 2 * TN) {
            amx_tile_zero(0);
            amx_tile_zero(1);
            amx_tile_zero(2);
            amx_tile_zero(3);

            const __bf16 *a_top = packed_a + m * K;
            const __bf16 *a_bot = packed_a + (m + TM) * K;
            const __bf16 *b_ptr = packed_b + n * K;

            for (int k = 0; k < K; k += TK) {
                const __bf16 *b_left = b_ptr;
                const __bf16 *b_right = b_ptr + 512; // 512 elements = 1024 bytes

                amx_tile_loadd(4, a_top, stride_a);
                amx_tile_loadd(6, b_left, stride_b);
                amx_tile_dpbf16ps(0, 4, 6); // C00 += A_top * B_left

                if (has_bottom) {
                    amx_tile_loadd(5, a_bot, stride_a);
                    amx_tile_dpbf16ps(2, 5, 6); // C10 += A_bot * B_left
                }

                amx_tile_loadd(7, b_right, stride_b);
                amx_tile_dpbf16ps(1, 4, 7); // C01 += A_top * B_right
                if (has_bottom) {
                    amx_tile_dpbf16ps(3, 5, 7); // C11 += A_bot * B_right
                }

                a_top += TM * TK; // 512 elements = 1024 bytes
                if (has_bottom) {
                    a_bot += TM * TK;
                }
                b_ptr += 1024; // 1024 elements = 2048 bytes
            }

            amx_tile_stored(0, C + m * N + n, stride_c);
            amx_tile_stored(1, C + m * N + n + TN, stride_c);
            if (has_bottom) {
                amx_tile_stored(2, C + (m + TM) * N + n, stride_c);
                amx_tile_stored(3, C + (m + TM) * N + n + TN, stride_c);
            }
        }
    }
}

int
main()
{
    std::fprintf(stderr, "GEMM: initializing A (%d x %d) and B (%d x %d)\n",
                 M, K, K, N);
    // Inputs cycle through zeros, integers and decimals.
    const float values[] = {0.0F, 1.0F, -2.0F, 3.0F, 0.5F, -1.25F, 0.1F, 2.75F};
    static __bf16 A[M][K], B[K][N];
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            A[m][k] = static_cast<__bf16>(values[(m + k) % 8]);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            B[k][n] = static_cast<__bf16>(values[(3 * k + n) % 8]);
        }
    }

    std::fprintf(stderr, "GEMM: inputs initialized; packing A\n");
    // Pack A into Continuous Tile Layout
    alignas(64) static __bf16 packed_a[M * K];
    int out = 0;
    for (int m = 0; m < M; m += TM) {
        for (int k = 0; k < K; k += TK) {
            for (int r = 0; r < TM; ++r) {
                for (int c = 0; c < TK; ++c) {
                    packed_a[out++] = A[m + r][k + c];
                }
            }
        }
    }

    std::fprintf(stderr, "GEMM: A packed; packing B\n");
    // Pack B into Continuous Tile Layout (as needed by intel amx)
    alignas(64) static __bf16 packed_b[K * N];
    out = 0;
    for (int n = 0; n < N; n += 2 * TN) {
        for (int k = 0; k < K; k += TK) {
            for (int half = 0; half < 2; ++half) {
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < 16; ++c) {
                        packed_b[out++] = B[k + 2 * r][n + half * TN + c];
                        packed_b[out++] = B[k + 2 * r + 1][n + half * TN + c];
                    }
                }
            }
        }
    }

    alignas(64) static float C[M * N];

    alignas(64) TileConfig config = {};
    config.paletteId = 1;
    for (int tile = 0; tile < 8; ++tile) {
        config.rows[tile] = 16;
        config.columnBytes[tile] = 64; // 32 BF16 values
    }

    amx_tile_loadconfig(&config);

    // Warm-up run brings A, B and C into the caches.
    gemm(packed_a, packed_b, C);
    amx_wait();

    m5_work_begin(0, 0);
    gemm(packed_a, packed_b, C);
    amx_wait();
    m5_work_end(0, 0);

    alignas(64) TileConfig release = {};
    amx_tile_loadconfig(&release);

    // Print the raw FP32 bits of each element of C, one matrix row per line.
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            uint32_t bits;
            std::memcpy(&bits, &C[m * N + n], sizeof(bits));
            std::printf("%08x ", bits);
        }
        std::printf("\n");
    }

    return 0;
}
