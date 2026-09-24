#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdalign.h>
#include <immintrin.h>
#include <unistd.h>
#include <x86intrin.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

// Intel AMX setup 
alignas(64) int8_t mat_A[16 * 64];

typedef struct
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} tilecfg_t;

bool set_tiledata_use()
{
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        return false;
    }
    return true;
}

void init_amx()
{
    alignas(64) tilecfg_t cfg = {0};
    cfg.palette_id = 1;
    cfg.start_row = 0;

    // Configure TMM0 - TMM7 to be 16 rows by 64 bytes (max size)
    for (int i = 0; i < 8; i++) {
        cfg.rows[i] = 16;
        cfg.colsb[i] = 64;
    }

    _tile_loadconfig(&cfg);
}

int perf_fd = -1;

void init_perf_counter()
{
    // --- set up the counter ---
    struct perf_event_attr pe = {0};
    pe.size = sizeof(struct perf_event_attr);
    pe.disabled = 0; // enable perf_event

    pe.type = PERF_TYPE_HARDWARE; 
    pe.config = PERF_COUNT_HW_CPU_CYCLES; // count unhalted cpu cycles
    pe.exclude_kernel = 1; // don't count OS cycles
    pe.exclude_hv = 1;  // don't count VM cycles

    perf_fd = syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);

    // --- enable/open the counter ---
    if (perf_fd == -1) {
        perror("perf_event_open failed");
        exit(-1);
    }
}

static inline uint64_t get_cycles()
{
    uint64_t count = 0;
    _mm_lfence();
    if (read(perf_fd, &count, sizeof(count)) != sizeof(count)) {
        return 0;
    }
    _mm_lfence();
    return count;
}

#define LOAD_FENCE_8 \
    _tile_loadd(0, mat_A, 64); _mm_lfence(); \
    _tile_loadd(0, mat_A, 64); _mm_lfence(); \
    _tile_loadd(0, mat_A, 64); _mm_lfence(); \
    _tile_loadd(0, mat_A, 64); _mm_lfence(); \
    _tile_loadd(0, mat_A, 64); _mm_lfence(); \
    _tile_loadd(0, mat_A, 64); _mm_lfence(); \
    _tile_loadd(0, mat_A, 64); _mm_lfence(); \
    _tile_loadd(0, mat_A, 64); _mm_lfence();

#define UNROLL_40_LOAD_FENCES \
    LOAD_FENCE_8 LOAD_FENCE_8 LOAD_FENCE_8 LOAD_FENCE_8 LOAD_FENCE_8

#define UNROLL_200_LOAD_FENCES \
    UNROLL_40_LOAD_FENCES UNROLL_40_LOAD_FENCES UNROLL_40_LOAD_FENCES UNROLL_40_LOAD_FENCES UNROLL_40_LOAD_FENCES

#define UNROLL_1000_LOAD_FENCES \
    UNROLL_200_LOAD_FENCES UNROLL_200_LOAD_FENCES UNROLL_200_LOAD_FENCES UNROLL_200_LOAD_FENCES UNROLL_200_LOAD_FENCES

// 8 LFENCE instructions alone
#define FENCE_8 \
    _mm_lfence(); _mm_lfence(); _mm_lfence(); _mm_lfence(); \
    _mm_lfence(); _mm_lfence(); _mm_lfence(); _mm_lfence();

#define UNROLL_40_FENCES \
    FENCE_8 FENCE_8 FENCE_8 FENCE_8 FENCE_8

#define UNROLL_200_FENCES \
    UNROLL_40_FENCES UNROLL_40_FENCES UNROLL_40_FENCES UNROLL_40_FENCES UNROLL_40_FENCES

#define UNROLL_1000_FENCES \
    UNROLL_200_FENCES UNROLL_200_FENCES UNROLL_200_FENCES UNROLL_200_FENCES UNROLL_200_FENCES

int main() {

    if (!set_tiledata_use()) {
        fprintf(stderr, "Failed to enable AMX (XTILEDATA) state.\n");
        exit(-1);
    }

    init_amx();
    init_perf_counter();

    for (int i = 0; i < 16 * 64; i++) {
        mat_A[i] = (i % 127) + 1;
    }

    // Warm-up cache and pipeline
    for (int i = 0; i < 100; i++) {
        LOAD_FENCE_8
    }

    uint64_t start, end;
    uint64_t fence_cycles, load_fence_cycles;

    // measure baseline overhead of 1000 LFENCE instructions
    start = get_cycles();
    UNROLL_1000_FENCES
    end = get_cycles();
    fence_cycles = end - start;

    // measure tileload + mm lefence instructions
    start = get_cycles();
    _tile_loadd(0, mat_A, 64);
    end = get_cycles();
    load_fence_cycles = end - start;

    _tile_release();

    int64_t load_cycles = (int64_t)load_fence_cycles - (int64_t)fence_cycles;
    double load_latency = (double)load_cycles / 1000.0;
    double load_fence_latency = (double)load_fence_cycles / 1000.0;

    printf("Total Cycles Taken: %ld \n", load_fence_cycles);
    printf("Latency (Cycles Per Load & Fence Instruction): %.2f \n", load_fence_latency);
    printf("Total Load Cycles Taken: %ld \n", load_cycles);
    printf("Latency (Cycles Per Load Instruction): %.2f \n", load_latency);

    close(perf_fd);
    return 0;
}
