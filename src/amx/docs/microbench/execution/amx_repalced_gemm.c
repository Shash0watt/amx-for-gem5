#define _GNU_SOURCE
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
#include <sched.h>

#define MAX_ROWS 16
#define MAX_COLS_BYTES 64

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

// Fake tile structure for emulation
typedef struct
{
    uint16_t rows;
    uint16_t colbytes;
    int8_t data[MAX_ROWS][MAX_COLS_BYTES];
} tile_t;

tile_t TMM[8]; 

// --- EMULATION FUNCTIONS ---

void tile_zero(int tile)
{
    for (int i = 0; i < 8; i++)
    {
        memset(TMM[i].data, 0, sizeof(TMM[i].data));
    }
}

void tile_loadconfig(__tilecfg *tcfg)
{
    for (int i = 0; i < 8; i++)
    {
        TMM[i].rows = tcfg->rows[i];
        TMM[i].colbytes = tcfg->colsb[i];
        memset(TMM[i].data, 0, sizeof(TMM[i].data));
    }
}

void tile_release()
{
    memset(TMM, 0, sizeof(TMM));
}

void tile_loadd(int dest, const void *arr, int stride)
{
    tile_t *tile = &TMM[dest];
    const int8_t *src = (const int8_t *)arr;

    for (int i = 0; i < tile->rows; i++)
    {
        memcpy(tile->data[i], src + (i * stride), tile->colbytes);
    }
}

void tile_stored(int src, void *arr, int stride)
{
    tile_t *tile = &TMM[src];
    int8_t *dst = (int8_t *)arr;

    for (int i = 0; i < tile->rows; i++)
    {
        memcpy(dst + (i * stride), tile->data[i], tile->colbytes);
    }
}

// use avx 512 maybe
void tile_dpbssd(int t_dst, int t_src1, int t_src2)
{
    tile_t *C = &TMM[t_dst];
    tile_t *A = &TMM[t_src1];
    tile_t *B = &TMM[t_src2];

    int M = A->rows;
    int K = B->colbytes / 4;
    int N = B->rows;

    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            int32_t *c_val = (int32_t *)&C->data[m][k * 4];

            for (int n = 0; n < N; n++)
            {
                // Load the 4 bytes into the SSE registers
                __m128i a_vals = _mm_cvtsi32_si128(*(int32_t *)&A->data[m][n * 4]);
                __m128i b_vals = _mm_cvtsi32_si128(*(int32_t *)&B->data[n][k * 4]);

                // Sign extend 8-bit to 16-bit via unpacking and shifting
                __m128i a_ext = _mm_unpacklo_epi8(a_vals, a_vals);
                __m128i b_ext = _mm_unpacklo_epi8(b_vals, b_vals);
                a_ext = _mm_srai_epi16(a_ext, 8);
                b_ext = _mm_srai_epi16(b_ext, 8);

                // Multiply and Add Adjacent
                __m128i madd = _mm_madd_epi16(a_ext, b_ext);

                // Horizontal add
                __m128i sum = _mm_add_epi32(madd, _mm_srli_si128(madd, 4));

                // Accumulate into C matrix tile
                *c_val += _mm_cvtsi128_si32(sum);
            }
        }
    }
}

// --- HELPER FUNCTIONS ---

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
    // Bypassed for emulation mode so it runs on non-AMX targets
    return true; 
    
    /* Original Hardware Call:
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
    {
        return false;
    }
    return true;
    */
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

    tile_loadconfig(&tcfg); // Swapped to emulation function
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

    // Default to the emulation limits instead of calling hardware cpuid
    int max_bytes_per_row = MAX_COLS_BYTES;
    int max_rows = MAX_ROWS;

    /* Original Hardware CPUID Call:
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(0x1d, 1, eax, ebx, ecx, edx);
    int max_bytes_per_row = ebx & 0xffff;
    int max_rows = ecx & 0xffff;
    */

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

    // Warmup Loop
    int warmups = 10;
    for (int i = 0; i < warmups; i++)
    {
        for (int m = 0; m < total_rows; m += tile_m)
        {
            for (int n = 0; n < total_cols; n += tile_n)
            {
                tile_zero(0); // Emulated

                for (int k = 0; k < total_cols; k += tile_k)
                {
                    int8_t *ptr_a = &padded_matA[m * total_cols + k];
                    int8_t *ptr_b = &packed_matB[(k / 4) * (total_cols * 4) + (n * 4)];

                    tile_loadd(1, ptr_a, stride_a);  // Emulated
                    tile_loadd(2, ptr_b, stride_b);  // Emulated
                    tile_dpbssd(0, 1, 2);            // Emulated
                }

                int32_t *ptr_c = &matrix_c[m * total_cols + n];
                tile_stored(0, ptr_c, stride_c);     // Emulated
            }
        }
    }

    // Benchmark Loop
    double freq_before = get_current_cpu_freq_mhz(); // Fixed structural typo 's'
    int iterations = 100;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++)
    {
        for (int m = 0; m < total_rows; m += tile_m)
        {
            for (int n = 0; n < total_cols; n += tile_n)
            {
                tile_zero(0); // Emulated

                for (int k = 0; k < total_cols; k += tile_k)
                {
                    int8_t *ptr_a = &padded_matA[m * total_cols + k];
                    int8_t *ptr_b = &packed_matB[(k / 4) * (total_cols * 4) + (n * 4)];

                    tile_loadd(1, ptr_a, stride_a);  // Emulated
                    tile_loadd(2, ptr_b, stride_b);  // Emulated
                    tile_dpbssd(0, 1, 2);            // Emulated
                }

                int32_t *ptr_c = &matrix_c[m * total_cols + n];
                tile_stored(0, ptr_c, stride_c);     // Emulated
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double freq_after = get_current_cpu_freq_mhz();

    // Metrics compilation
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

    tile_release(); // Emulated
    free(matA);
    free(matB);
    free(padded_matA);
    free(padded_matB);
    free(packed_matB);
    free(matrix_c);

    return 0;
}