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
// gcc -O3 -mamx-tile store_trgh.c -o store_trgh

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

// Intel AMX setup 
alignas(64) int8_t mat_A[16 * 64];

// 8 distinct 1 KB memory buffers (one per tile register).
// 8 * 1024 bytes = 8 KB, which easily fits within L1D cache (32-48 KB).
// Storing each tile register to its own separate buffer eliminates Store Buffer
// address collisions and L1D cache line write-serialization, allowing TILESTORED
// to achieve its peak hardware issue throughput of ~16 cycles/instruction.
alignas(64) int8_t mat_dst[8][16 * 64];

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

#define STORE_8 \
    _tile_stored(0, mat_dst[0], 64); \
    _tile_stored(1, mat_dst[1], 64); \
    _tile_stored(2, mat_dst[2], 64); \
    _tile_stored(3, mat_dst[3], 64); \
    _tile_stored(4, mat_dst[4], 64); \
    _tile_stored(5, mat_dst[5], 64); \
    _tile_stored(6, mat_dst[6], 64); \
    _tile_stored(7, mat_dst[7], 64);

#define UNROLL_40_STORES \
    STORE_8 STORE_8 STORE_8 STORE_8 STORE_8

#define UNROLL_200_STORES \
    UNROLL_40_STORES UNROLL_40_STORES UNROLL_40_STORES UNROLL_40_STORES UNROLL_40_STORES

#define UNROLL_1000_STORES \
    UNROLL_200_STORES UNROLL_200_STORES UNROLL_200_STORES UNROLL_200_STORES UNROLL_200_STORES

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

    for (int t = 0; t < 8; t++) {
        for (int i = 0; i < 16 * 64; i++) {
            mat_dst[t][i] = 0;
        }
    }

    _tile_loadd(0, mat_A, 64);
    _tile_loadd(1, mat_A, 64);
    _tile_loadd(2, mat_A, 64);
    _tile_loadd(3, mat_A, 64);
    _tile_loadd(4, mat_A, 64);
    _tile_loadd(5, mat_A, 64);
    _tile_loadd(6, mat_A, 64);
    _tile_loadd(7, mat_A, 64);

    // Warm-up
    for (int i = 0; i < 100; i++) {
        STORE_8
    }

    uint64_t start, end, cycles, offset;

    start = get_cycles();
    end = get_cycles();
    offset = end - start;

    start = get_cycles();
    UNROLL_1000_STORES
    end = get_cycles();

    _tile_release();

    cycles = end - start - offset;
    double cycles_per_inst = (double)cycles / 1000.0;

    printf("Total Cycles Taken: %lu \n", cycles);
    printf("Throughput (Cycles Per Instruction): %.2f \n", cycles_per_inst);

    close(perf_fd);
    return 0;
}
