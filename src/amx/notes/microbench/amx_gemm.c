#define _GNU_SOURCE
#include <sched.h>   // Explicitly include this right away
#include <cpuid.h>
#include <time.h>
#include <stdio.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

// gcc -O3 -mamx-tile -mamx-int8 amx_gemm.c -o amx_matmul (if not inamx machine)
// gcc -O3 -march=native main.c -o amx_matmul (if in amx machine)

//--- AMX Benchmark Results (Lighthouse)m---
// Total Time for 100 iterations: 0.000443 seconds
// Average Time per iteration : 0.000004 seconds (0.00 ms)
// Performance                : 946.28 GOPS

// --- AMX Benchmark Results (Emulation) ---
// Total Time for 100 iterations: 0.042673 seconds
// Average Time per iteration : 0.000427 seconds (0.43 ms)
// Performance                : 9.83 GOPS

// helper function to get the current core's frequency in MHz
static double get_current_cpu_freq_mhz()
{
    // Method 1: Try sysfs (Best for bare-metal Linux)
    int cpu = sched_getcpu(); 
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    
    FILE *f = fopen(path, "r");
    if (f != NULL) {
        unsigned long freq_khz = 0;
        if (fscanf(f, "%lu", &freq_khz) == 1) {
            fclose(f);
            return (double)freq_khz / 1000.0; // Convert kHz to MHz
        }
        fclose(f);
    }
    
    // Method 2 Fallback: Parse /proc/cpuinfo (Best for VMs, WSL, and Containers)
    f = fopen("/proc/cpuinfo", "r");
    if (f != NULL) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            // Look for the "cpu MHz" line provided by the kernel
            if (strncmp(line, "cpu MHz", 7) == 0) {
                char *colon = strchr(line, ':');
                if (colon) {
                    double freq = atof(colon + 1);
                    fclose(f);
                    return freq; // Already in MHz
                }
            }
        }
        fclose(f);
    }
    
    return 0.0; // If all else fails
}

typedef struct
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

static bool set_tiledata_use()
{
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
    {
        return false;
    }
    return true;
}

static void configure_tiles(int tile_m, int tile_n, int tile_k)
{
    __tilecfg tcfg = {0};

    tcfg.palette_id = 1;
    tcfg.start_row = 0;

    tcfg.rows[0] = tile_m;
    tcfg.colsb[0] = tile_n * sizeof(int32_t);

    tcfg.rows[1] = tile_m;
    tcfg.colsb[1] = tile_k * sizeof(int8_t);

    tcfg.rows[2] = tile_k / 4;
    tcfg.colsb[2] = tile_n * 4;

    _tile_loadconfig(&tcfg);
}

static int8_t *pack_matB(int8_t *matrix_b, int total_k, int total_n)
{
    int total_elements = total_k * total_n;
    int8_t *packed_b = (int8_t *)malloc(total_elements * sizeof(int8_t));

    for (int k = 0; k < total_k; k += 4)
    {
        for (int n = 0; n < total_n; n++)
        {
            for (int i = 0; i < 4; i++)
            {
                packed_b[(k / 4) * (total_n * 4) + (n * 4) + i] = matrix_b[(k + i) * total_n + n];
            }
        }
    }
    return packed_b;
}

static int32_t *make_matrix32(int rows, int cols, int32_t value)
{
    int total_elements = rows * cols;
    int32_t *matrix = (int32_t *)malloc(total_elements * sizeof(int32_t));
    if (matrix == NULL)
        return NULL;
    for (int i = 0; i < total_elements; i++)
        matrix[i] = value;
    return matrix;
}

static int8_t *make_random_row_matrix(int rows, int cols)
{
    int total_elements = rows * cols;
    int8_t *matrix = (int8_t *)malloc(total_elements * sizeof(int8_t));

    if (matrix == NULL)
        return NULL;

    for (int r = 0; r < rows; r++)
    {
        int8_t num = (int8_t)((rand() % 9) + 1);
        for (int c = 0; c < cols; c++)
        {
            matrix[r * cols + c] = num;
        }
    }
    return matrix;
}

static int8_t *pad_matrix_int8(int8_t *orig_matrix, int orig_rows, int orig_cols, int padded_rows, int padded_cols)
{
    int total_padded_elements = padded_rows * padded_cols;
    int8_t *padded = (int8_t *)calloc(total_padded_elements, sizeof(int8_t));

    if (padded == NULL)
        return NULL;

    for (int r = 0; r < orig_rows; r++)
    {
        for (int c = 0; c < orig_cols; c++)
        {
            padded[r * padded_cols + c] = orig_matrix[r * orig_cols + c];
        }
    }
    return padded;
}

int main()
{
    if (!set_tiledata_use())
        exit(-1);

    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(0x1d, 1, eax, ebx, ecx, edx);

    int max_bytes_per_row = ebx & 0xffff;
    int max_rows = ecx & 0xffff;

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
                    _tile_dpbssd(0, 1, 2);  // C= A*B+c
                }

                int32_t *ptr_c = &matrix_c[m * total_cols + n];
                _tile_stored(0, ptr_c, stride_c);
            }
        }
    }

    // test here
    double freq_before = get_current_cpu_freq_mhz();
    int iterations = 100;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++)
    {
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
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double freq_after = get_current_cpu_freq_mhz();
    // benchmark end

    // calculate metrics
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