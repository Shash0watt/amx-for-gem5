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

// intel amx setup 
alignas(64) int8_t mat_A[16 * 64];
// alignas(64) int8_t mat_B[16 * 64];
// alignas(64) int8_t mat_dst[16 * 64];

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
    // we also have exclude_user, exclude_kernel, exclue_idele

    perf_fd = syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
    // [syscall, perf_event_attr, pid (0), CPU (-1 call on any cpu), group_fd (-1 this event is the leader?), any other flags ]

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

// 8 independent tile loads 
#define LOAD_8 \
    _tile_loadd(0, mat_A, 64); \
    _tile_loadd(1, mat_A, 64); \
    _tile_loadd(2, mat_A, 64); \
    _tile_loadd(3, mat_A, 64); \
    _tile_loadd(4, mat_A, 64); \
    _tile_loadd(5, mat_A, 64); \
    _tile_loadd(6, mat_A, 64); \
    _tile_loadd(7, mat_A, 64);

// Unroll 1000 loads: 8 * 5 * 5 * 5 = 1000
#define UNROLL_40_LOADS \
    LOAD_8 LOAD_8 LOAD_8 LOAD_8 LOAD_8

#define UNROLL_200_LOADS \
    UNROLL_40_LOADS UNROLL_40_LOADS UNROLL_40_LOADS UNROLL_40_LOADS UNROLL_40_LOADS

#define UNROLL_1000_LOADS \
    UNROLL_200_LOADS UNROLL_200_LOADS UNROLL_200_LOADS UNROLL_200_LOADS UNROLL_200_LOADS


int main() {

    if (!set_tiledata_use()) {
        fprintf(stderr, "Failed to enable AMX (XTILEDATA) state.\n");
        exit(-1);
    }

    init_amx();
    init_perf_counter();

    for (int i = 0; i < 16 * 64; i++) {
        mat_A[i] = (i % 127) + 1;
        // mat_B[i] = (i % 127) + 1;
        // mat_dst[i] = ((i + 13) % 127) + 1;
    }

    // Warm-up
    for (int i = 0; i < 100; i++) {
        LOAD_8
    }

    uint64_t start, end, cycles;

    start = get_cycles();
    UNROLL_1000_LOADS
    end = get_cycles();

    _tile_release();

    cycles = end - start;
    double cycles_per_inst = (double)cycles / 1000.0;

    printf("Total Cycles Taken: %lu \n", cycles);
    printf("Throughput (Cycles Per Instruction): %.2f \n", cycles_per_inst);

    close(perf_fd);
    return 0;
}

