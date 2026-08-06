#include <immintrin.h>
#include "optimized_gemm.h"

// gcc -O3 -mamx-tile -mamx-bf16 amx_gemm.c -o amx_matmul (if not inamx machine)
// gcc -O3 -march=native main.c -o amx_matmul (if in amx machine)

//--- AMX Benchmark Results (Lighthouse)m---
// Total Time for 100 iterations: 0.000443 seconds
// Average Time per iteration : 0.000004 seconds (0.00 ms)
// Performance                : 946.28 GOPS

// --- AMX Benchmark Results (Emulation) ---
// Total Time for 100 iterations: 0.042673 seconds
// Average Time per iteration : 0.000427 seconds (0.43 ms)
// Performance                : 9.83 GOPS

int main()
{   
    // syscall to use intel AMX
    if (!set_tiledata_use())
        exit(-1);

    // get hardware information
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(0x1d, 1, eax, ebx, ecx, edx);

    // save tile specific hardware information
    int max_bytes_per_row = ebx & 0xffff;
    int max_rows = ecx & 0xffff;

    // create a test matrix..
    int rows = 128;
    int cols = 128;

    int total_rows = ((rows + max_rows - 1) / max_rows) * max_rows;
    int total_cols = ((cols + max_bytes_per_row - 1) / max_bytes_per_row) * max_bytes_per_row;

    int8_t *matA = make_random_row_matrix(rows, cols);
    int8_t *matB = make_random_row_matrix(rows, cols);

    int8_t *padded_matA = pad_matrix_int8(matA, rows, cols, total_rows, total_cols);
    int8_t *padded_matB = pad_matrix_int8(matB, rows, cols, total_rows, total_cols);

    int8_t *packed_matB = pack_matB(padded_matB, total_cols, total_cols);
    int32_t *matrix_c = make_matrix32(total_rows, total_cols, 0);

    int tile_m = max_rows;
    int tile_k = max_bytes_per_row;
    int tile_n = max_bytes_per_row / 4;

    int stride_a = total_cols * sizeof(int8_t);
    int stride_b = total_cols * 4;
    int stride_c = total_cols * sizeof(int32_t);

    configure_tiles(tile_m, tile_n, tile_k);

    // warmup here
    int warmups = 10;

    for (int i = 0; i < warmups; i++)
    {   
        // --- start warmup matrix multiplication loop ---
        for (int m = 0; m < total_rows; m += tile_m)
        {
            for (int n = 0; n < total_cols; n += tile_n)
            {
                _tile_zero(0);

                for (int k = 0; k < total_cols; k += tile_k)
                {
                    int8_t *ptr_a = &padded_matA[m * total_cols + k];
                    int8_t *ptr_b = &packed_matB[(k / 4) * (total_cols * 4) + (n * 4)];

                    _tile_loadd(1, ptr_a, stride_a);
                    _tile_loadd(2, ptr_b, stride_b);
                    _tile_dpbf16ps(0, 1, 2);  // C= A*B+c
                }

                int32_t *ptr_c = &matrix_c[m * total_cols + n];
                _tile_stored(0, ptr_c, stride_c);
            }
        }
        // --- end warmup matrix multiplication ---
    }

    // test here
    double freq_before = get_current_cpu_freq_mhz();
    int iterations = 100;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++)
    {
        // --- start actual matrix multiplication loop ---
        for (int m = 0; m < total_rows; m += tile_m)
        {
            for (int n = 0; n < total_cols; n += tile_n)
            {
                _tile_zero(0);

                for (int k = 0; k < total_cols; k += tile_k)
                {
                    int8_t *ptr_a = &padded_matA[m * total_cols + k];
                    int8_t *ptr_b = &packed_matB[(k / 4) * (total_cols * 4) + (n * 4)];

                    _tile_loadd(1, ptr_a, stride_a);
                    _tile_loadd(2, ptr_b, stride_b);
                    _tile_dpbssd(0, 1, 2);
                }

                int32_t *ptr_c = &matrix_c[m * total_cols + n];
                _tile_stored(0, ptr_c, stride_c);
            }
        }
        // --- end warmup matrix multiplication ---
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double freq_after = get_current_cpu_freq_mhz();
    // benchmark end
    
    // --- calculate metrics ---
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double avg_time = time_taken / iterations;
    double total_ops = 2.0 * total_rows * total_cols * total_cols; 
    double gops = (total_ops / avg_time) / 1e9;

    printf("--- AMX Benchmark & Frequency Results ---\n");
    printf("Total Time for %d iterations: %.6f seconds\n", iterations, time_taken);
    printf("Average Time per iteration : %.6f seconds (%.2f ms)\n", avg_time, avg_time * 1000.0);
    printf("Performance                : %.2f GOPS\n", gops);
    printf("\n");
    printf("CPU Frequency Before Loop  : %.2f MHz (%.2f GHz)\n", freq_before, freq_before / 1000.0);
    printf("CPU Frequency After Loop   : %.2f MHz (%.2f GHz)\n", freq_after, freq_after / 1000.0);
    printf("Frequency Change            : %.2f MHz\n", freq_after - freq_before);

    _tile_release();
    free(matA);
    free(matB);
    free(padded_matA);
    free(padded_matB);
    free(packed_matB);
    free(matrix_c);

    return 0;
}
