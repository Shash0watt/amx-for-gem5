#ifndef OPTIMIZED_GEMM_H
#define OPTIMIZED_GEMM_H

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

typedef struct
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

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

    // tile C - results
    tcfg.rows[0] = tile_m;
    tcfg.colsb[0] = tile_n * sizeof(int32_t);

    // tile A
    tcfg.rows[1] = tile_m;
    tcfg.colsb[1] = tile_k * sizeof(int8_t);

    // tile B - vnni packed
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

#endif // OPTIMIZED_GEMM_H
