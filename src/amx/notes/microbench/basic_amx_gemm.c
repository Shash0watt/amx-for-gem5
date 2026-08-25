#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

// Compile from the repository root:
// gcc -O3 -mamx-tile -mamx-bf16 basic_amx_gemm.c -o basic_gemm

enum
{
    M = 16,
    N = 16,
    K = 32,
    WARMUP_ITERATIONS = 100,
    MEASURED_ITERATIONS = 1000,
    A_ROW_BYTES = K * sizeof(uint16_t),
    PACKED_B_ROW_BYTES = N * 2 * sizeof(uint16_t),
    C_ROW_BYTES = N * sizeof(float)
};

// 64-byte AMX Tile Configuration Register structure
typedef struct
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

// Request OS permission to use Intel AMX tile hardware
static bool set_tiledata_use(void)
{
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
                   XFEATURE_XTILEDATA) == 0;
}

static inline void
run_gemm(const __tilecfg *config, const uint16_t *matrix_a,
         const uint16_t *packed_b, float *matrix_c)
{
    _tile_loadconfig(config);
    _tile_zero(0);
    _tile_loadd(1, matrix_a, A_ROW_BYTES);
    _tile_loadd(2, packed_b, PACKED_B_ROW_BYTES);
    _tile_dpbf16ps(0, 1, 2);
    _tile_stored(0, matrix_c, C_ROW_BYTES);
}

int
main(void)
{
    // 1. Request OS permission for AMX tile data.
    if (!set_tiledata_use()) {
        fprintf(stderr, "Error: AMX tile permission (XFEATURE_XTILEDATA) failed.\n");
        return -1;
    }

    // 2. Set up A[16][32] and B[32][16]. BF16 values are stored as raw bits:
    // 0x4000 is 2.0 and 0x4080 is 4.0.
    alignas(64) uint16_t matrix_a[M][K];
    alignas(64) uint16_t matrix_b[K][N];
    alignas(64) uint16_t packed_b[K / 2][2 * N];
    alignas(64) float matrix_c[M][N] = {0};

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            matrix_a[m][k] = 0x4000;
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            matrix_b[k][n] = 0x4080;
        }
    }

    // TDPBF16PS expects adjacent K values from B next to each other.
    for (int k = 0; k < K; k += 2) {
        for (int n = 0; n < N; ++n) {
            packed_b[k / 2][2 * n] = matrix_b[k][n];
            packed_b[k / 2][2 * n + 1] = matrix_b[k + 1][n];
        }
    }

    // 3. Tile configuration setup
    alignas(64) __tilecfg config = {0};
    config.palette_id = 1;
    config.start_row = 0;

    // TMM0 holds the 16x16 FP32 result.
    config.rows[0] = M;
    config.colsb[0] = C_ROW_BYTES;

    // TMM1 holds A as 16 rows of 32 BF16 values.
    config.rows[1] = M;
    config.colsb[1] = A_ROW_BYTES;

    // TMM2 holds the packed 32x16 BF16 matrix B as 16 physical rows.
    config.rows[2] = K / 2;
    config.colsb[2] = PACKED_B_ROW_BYTES;

    // 4. Warm up AMX and bring the data into cache.
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        run_gemm(&config, &matrix_a[0][0], &packed_b[0][0],
                 &matrix_c[0][0]);
    }

    // 5. Measure 1000 GEMMs as one batch.
    _mm_lfence();
    uint64_t start_cycles = __rdtsc();
    _mm_lfence();

    for (int i = 0; i < MEASURED_ITERATIONS; ++i) {
        run_gemm(&config, &matrix_a[0][0], &packed_b[0][0],
                 &matrix_c[0][0]);
    }

    unsigned int tsc_aux;
    uint64_t end_cycles = __rdtscp(&tsc_aux);
    _mm_lfence();

    double average_cycles =
        (double)(end_cycles - start_cycles) / MEASURED_ITERATIONS;

    // 6. Release the tile state and print the result.
    _tile_release();

    // Every output is the sum of 32 products: 32 * 2.0 * 4.0 = 256.0.
    printf("C[0][0]: %.1f (expected: 256.0)\n", matrix_c[0][0]);
    printf("Average TSC cycles over %d iterations: %.2f\n",
           MEASURED_ITERATIONS, average_cycles);

    return 0;
}
