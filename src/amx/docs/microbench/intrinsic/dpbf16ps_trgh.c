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

// Compile:
// gcc -O3 -mamx-tile -mamx-bf16 dpbf16ps_trgh.c -o dpbf16ps_trgh

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

// 16 rows by 32 BF16 elements (64 bytes per row)
alignas(64) uint16_t mat_A[16 * 32];
alignas(64) uint16_t mat_B[16 * 32];

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

    for (int i = 0; i < 8; i++) {
        cfg.rows[i] = 16;
        cfg.colsb[i] = 64;
    }

    _tile_loadconfig(&cfg);
}

int perf_fd = -1;

void init_perf_counter()
{
    struct perf_event_attr pe = {0};
    pe.size = sizeof(struct perf_event_attr);
    pe.disabled = 0;

    pe.type = PERF_TYPE_HARDWARE; 
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    perf_fd = syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);

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

#define DP_6 \
    _tile_dpbf16ps(0, 6, 7); \
    _tile_dpbf16ps(1, 6, 7); \
    _tile_dpbf16ps(2, 6, 7); \
    _tile_dpbf16ps(3, 6, 7); \
    _tile_dpbf16ps(4, 6, 7); \
    _tile_dpbf16ps(5, 6, 7);

#define UNROLL_60_DPS \
    DP_6 DP_6 DP_6 DP_6 DP_6 \
    DP_6 DP_6 DP_6 DP_6 DP_6

#define UNROLL_240_DPS \
    UNROLL_60_DPS UNROLL_60_DPS UNROLL_60_DPS UNROLL_60_DPS

#define UNROLL_1200_DPS \
    UNROLL_240_DPS UNROLL_240_DPS UNROLL_240_DPS UNROLL_240_DPS UNROLL_240_DPS

int main() {
    if (!set_tiledata_use()) {
        fprintf(stderr, "Failed to enable AMX (XTILEDATA) state.\n");
        exit(-1);
    }

    init_amx();
    init_perf_counter();

    for (int i = 0; i < 16 * 32; i++) {
        mat_A[i] = 0x4000;
        mat_B[i] = 0x4080;
    }

    _tile_loadd(6, mat_A, 64);
    _tile_loadd(7, mat_B, 64);
    
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);
    _tile_zero(4);
    _tile_zero(5);

    // Warm-up
    for (int i = 0; i < 100; i++) {
        DP_6
    }

    uint64_t start, end, cycles, offset;

    start = get_cycles();
    end = get_cycles();
    offset = end - start;

    start = get_cycles();
    UNROLL_1200_DPS
    end = get_cycles();

    _tile_release();

    cycles = end - start - offset;
    double cycles_per_inst = (double)cycles / 1200.0;

    printf("Total Cycles Taken: %lu \n", cycles);
    printf("Throughput (Cycles Per Instruction): %.2f \n", cycles_per_inst);

    close(perf_fd);
    return 0;
}
