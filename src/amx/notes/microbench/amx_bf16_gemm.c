#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

// compile using: gcc -O3 -mamx-tile -mamx-bf16 src/amx/notes/microbench/amx_bf16_gemm.c -o amx_bf16_gemm

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
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
    {
        return false;
    }
    return true;
}

// Pack matrix B into VNNI format for AMX TDPBF16PS
static uint16_t *pack_matB_bf16(const uint16_t *matrix_b, int total_k, int total_n)
{
    uint16_t *packed_b = (uint16_t *)malloc(total_k * total_n * sizeof(uint16_t));
    if (!packed_b) return NULL;

    for (int k = 0; k < total_k; k += 2)
    {
        for (int n = 0; n < total_n; n++)
        {
            packed_b[(k / 2) * (total_n * 2) + (n * 2) + 0] = matrix_b[(k + 0) * total_n + n];
            packed_b[(k / 2) * (total_n * 2) + (n * 2) + 1] = matrix_b[(k + 1) * total_n + n];
        }
    }
    return packed_b;
}

// Helper to create a matrix filled with a specific uint16_t (BF16) value
static uint16_t *make_matrix_bf16(int rows, int cols, uint16_t val)
{
    int total = rows * cols;
    uint16_t *mat = (uint16_t *)malloc(total * sizeof(uint16_t));
    if (!mat) return NULL;
    for (int i = 0; i < total; i++) {
        mat[i] = val;
    }
    return mat;
}

// Helper to create a float matrix filled with a specific value
static float *make_matrix_fp32(int rows, int cols, float val)
{
    int total = rows * cols;
    float *mat = (float *)malloc(total * sizeof(float));
    if (!mat) return NULL;
    for (int i = 0; i < total; i++) {
        mat[i] = val;
    }
    return mat;
}

int main(void)
{
    // 1. Request OS permission for AMX tile data
    if (!set_tiledata_use()) {
        fprintf(stderr, "Error: AMX tile permission (XFEATURE_XTILEDATA) failed.\n");
        return -1;
    }

    // 2. Input matrix setup:
    int total_rows = 64;
    int total_cols = 64;

    // Matrix A: 64x64 filled with 0x4000 (2.0 in BF16)
    // Matrix B: 64x64 filled with 0x4080 (4.0 in BF16)
    uint16_t *matA = make_matrix_bf16(total_rows, total_cols, 0x4000);
    uint16_t *matB = make_matrix_bf16(total_cols, total_cols, 0x4080);
    uint16_t *packed_matB = pack_matB_bf16(matB, total_cols, total_cols);
    float *matrix_c = make_matrix_fp32(total_rows, total_cols, 0.0F);

    if (!matA || !matB || !packed_matB || !matrix_c) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return -1;
    }

    int tile_m = 16;                           // 16 rows
    int tile_k_elements = 32;                  // 32 BF16 elements (64 bytes)
    int tile_n = 16;                           // 16 FP32 columns (64 bytes)

    int stride_a = total_cols * sizeof(uint16_t);
    int stride_b = total_cols * sizeof(uint16_t);
    int stride_c = total_cols * sizeof(float);

    // 3. Tile configuration setup
    alignas(64) __tilecfg config = {0};
    config.palette_id = 1;
    config.start_row = 0;

    // Tile 0 (C - FP32 destination): 16 rows, 64 bytes per row
    config.rows[0] = tile_m;
    config.colsb[0] = tile_n * sizeof(float);

    // Tile 1 (A - BF16 input): 16 rows, 64 bytes per row
    config.rows[1] = tile_m;
    config.colsb[1] = tile_k_elements * sizeof(uint16_t);

    // Tile 2 (B - BF16 VNNI input pairs): 16 rows, 64 bytes per row
    config.rows[2] = tile_k_elements / 2;
    config.colsb[2] = tile_n * sizeof(float);

    // 4. Warmup loop (Wakes up AMX hardware TMUL, handles kernel XSAVE allocation, warms caches)
    const int warmup_iterations = 1000;
    for (int i = 0; i < warmup_iterations; i++) {
        _tile_loadconfig(&config);
        for (int m = 0; m < total_rows; m += tile_m) {
            for (int n = 0; n < total_cols; n += tile_n) {
                _tile_zero(0);
                for (int k = 0; k < total_cols; k += tile_k_elements) {
                    const uint16_t *ptr_a = &matA[m * total_cols + k];
                    const uint16_t *ptr_b = &packed_matB[(k / 2) * (total_cols * 2) + (n * 2)];

                    _tile_loadd(1, ptr_a, stride_a);
                    _tile_loadd(2, ptr_b, stride_b);
                    _tile_dpbf16ps(0, 1, 2);
                }
                float *ptr_c = &matrix_c[m * total_cols + n];
                _tile_stored(0, ptr_c, stride_c);
            }
        }
    }

    // 5. Timed warm execution measurement using __rdtscp
    unsigned int dummy;
    uint64_t start_cycles = __rdtscp(&dummy);

    _tile_loadconfig(&config);
    for (int m = 0; m < total_rows; m += tile_m) {
        for (int n = 0; n < total_cols; n += tile_n) {
            _tile_zero(0);
            for (int k = 0; k < total_cols; k += tile_k_elements) {
                const uint16_t *ptr_a = &matA[m * total_cols + k];
                const uint16_t *ptr_b = &packed_matB[(k / 2) * (total_cols * 2) + (n * 2)];

                _tile_loadd(1, ptr_a, stride_a);
                _tile_loadd(2, ptr_b, stride_b);
                _tile_dpbf16ps(0, 1, 2);
            }
            float *ptr_c = &matrix_c[m * total_cols + n];
            _tile_stored(0, ptr_c, stride_c);
        }
    }

    uint64_t end_cycles = __rdtscp(&dummy);
    uint64_t elapsed_cycles = end_cycles - start_cycles;

    // 6. Release tile configuration
    _tile_release();

    // 7. Verify and print result & cycle count
    float expected_val = 2.0f * 4.0f * (float)total_cols;
    printf("AMX TDPBF16PS Matrix Multiplication Result C[0][0]: %.1f (Expected: %.1f)\n", matrix_c[0], expected_val);
    printf("Warm Execution CPU Cycles (loadconfig to stored): %lu cycles\n", (unsigned long)elapsed_cycles);

    free(matA);
    free(matB);
    free(packed_matB);
    free(matrix_c);

    return 0;
}
